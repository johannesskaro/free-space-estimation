"""
Loader for the milliAmpere 2 dataset from 2023-07-11 (ZED .svo files + ROS 2 bags).

Expected layout under the data root:

    <data_root>/
        ZED camera svo files/2023-07-11_..._28170706_HD1080_FPS15.svo
        bags/scen1/, bags/scen4_2/, ...
"""
import os
from dataclasses import dataclass
from typing import List, Union

import cv2
import numpy as np
import pyzed.sl as sl
import yaml
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

from free_space import utils as ut
from free_space.ma2.calibration import H_POINTS_ZED_FROM_LIDAR, H_POINTS_PIREN_FROM_PIREN_ENU
from free_space.zed import SVOCamera

LIDAR_TOPIC = "/lidar_aft/points"
GNSS_TOPIC = "/senti_parser/SentiPose"


def _clap_values(entries):
    # An entry [a, b] means a clap seen over two frames; use the midpoint
    return np.array([(e[0] + e[1]) / 2 if isinstance(e, list) else e for e in entries])


@dataclass
class Scenario:
    name: str
    description: str
    svo_file: str
    rosbag: str
    start_timestamp: int
    ma2_clap_timestamps: List[Union[int, List[int]]]
    svo_clap_timestamps: List[Union[int, List[int]]]

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(
            name=cfg["name"],
            description=cfg.get("description", ""),
            svo_file=cfg["svo_file"],
            rosbag=cfg["rosbag"],
            start_timestamp=cfg["start_timestamp"],
            ma2_clap_timestamps=cfg["clap_timestamps"]["ma2"],
            svo_clap_timestamps=cfg["clap_timestamps"]["svo"],
        )

    @property
    def gnss_minus_zed_time_ns(self):
        """Clock offset between the MA2 (GNSS) and ZED recordings, from hand claps."""
        return np.mean(_clap_values(self.ma2_clap_timestamps) - _clap_values(self.svo_clap_timestamps))


@dataclass
class Frame:
    timestamp: float            # ns, GNSS clock
    left_img: np.ndarray        # (H, W, 3) BGR, rectified
    disparity_img: np.ndarray   # (H, W) px
    depth_img: np.ndarray       # (H, W) m
    lidar_uv: np.ndarray        # (N, 2) lidar points projected into the image
    lidar_intensity: np.ndarray # (N, 1)
    lidar_xyz: np.ndarray       # (N, 3) lidar points in the camera frame
    pose: np.ndarray            # [x, y, z, qx, qy, qz, qw], NED


class MA2Sequence:
    """Iterates over synchronised camera, lidar and GNSS data for one scenario.

    Lidar and GNSS are read into memory up front; camera frames are read lazily.
    Each frame is paired with the closest lidar scan and GNSS pose in time.
    """

    image_size = (1080, 1920)

    def __init__(self, scenario: Scenario, data_root: str, num_frames: int = 200):
        self.scenario = scenario
        self.num_frames = num_frames
        self.rosbag_path = os.path.join(data_root, scenario.rosbag)
        self.time_offset_ns = scenario.gnss_minus_zed_time_ns

        self.camera = SVOCamera(os.path.join(data_root, scenario.svo_file))
        self.camera.set_svo_position_timestamp(scenario.start_timestamp)
        self.K, _ = self.camera.get_left_parameters()
        self.baseline = np.linalg.norm(self.camera.T)

        self.typestore = get_typestore(Stores.ROS2_FOXY)
        self.lidar_data = self._read_lidar()
        self.gnss_data = self._read_gnss()
        self.lidar_timestamps = np.array([entry[0] for entry in self.lidar_data])
        self.gnss_timestamps = np.array([entry[0] for entry in self.gnss_data])

    def _read_lidar(self):
        """Read all lidar scans after the start time, transformed to the camera frame
        and projected into the image. Only points in front of the camera are kept."""
        lidar_data = []
        with Reader(self.rosbag_path) as reader:
            connections = [c for c in reader.connections if c.topic == LIDAR_TOPIC]
            assert len(connections) == 1
            for connection, timestamp, rawdata in reader.messages(connections):
                if timestamp > self.scenario.start_timestamp:
                    msg = self.typestore.deserialize_cdr(rawdata, connection.msgtype)
                    xyz = msg.data.reshape(-1, msg.point_step)[:,:12].view(dtype=np.float32)
                    intensity = msg.data.reshape(-1, msg.point_step)[:,16:20].view(dtype=np.float32)

                    intensity_clipped = np.clip(intensity, 0, 100)

                    xyz_c = H_POINTS_ZED_FROM_LIDAR.dot(np.r_[xyz.T, np.ones((1, xyz.shape[0]))])[0:3, :].T

                    rvec = np.zeros((1,3), dtype=np.float32)
                    tvec = np.zeros((1,3), dtype=np.float32)
                    distCoeff = np.zeros((1,5), dtype=np.float32)
                    image_points, _ = cv2.projectPoints(xyz_c, rvec, tvec, self.K, distCoeff)

                    forward = xyz_c[:,2] > 0
                    xyz_c_forward = xyz_c[forward]
                    image_points_forward = np.squeeze(image_points[forward], axis=1)
                    intensity_clipped_forward = intensity_clipped[forward]

                    lidar_data.append([timestamp, image_points_forward, intensity_clipped_forward, xyz_c_forward])

        return lidar_data

    def _read_gnss(self):
        """Read all GNSS poses as [timestamp, position (NED), quaternion]."""
        t_pos_ori = []
        with Reader(self.rosbag_path) as reader:
            connections = [c for c in reader.connections if c.topic == GNSS_TOPIC]
            for connection, timestamp, rawdata in reader.messages(connections):
                msg = self.typestore.deserialize_cdr(rawdata, connection.msgtype)
                timestamp_msg = msg.header.stamp.sec * (10**9) + msg.header.stamp.nanosec

                pos_ros = msg.pose.position
                pos = np.array([pos_ros.x, pos_ros.y, pos_ros.z])
                ori_ros = msg.pose.orientation
                ori_quat = np.array([ori_ros.x, ori_ros.y, ori_ros.z, ori_ros.w])

                pos = H_POINTS_PIREN_FROM_PIREN_ENU.dot(np.r_[pos, 1])[:3].T

                t_pos_ori.append([timestamp_msg, pos, ori_quat])
        return t_pos_ori

    def __iter__(self):
        curr_frame = 0
        while self.camera.grab() == sl.ERROR_CODE.SUCCESS and curr_frame < self.num_frames:
            left_img = self.camera.get_left_image(should_rectify=True)
            timestamp = self.camera.get_timestamp() + self.time_offset_ns
            disparity_img = self.camera.get_neural_disp()
            depth_img = self.camera.get_depth_image()
            curr_frame += 1

            lidar_idx, _ = ut.find_closest_timestamp(self.lidar_timestamps, timestamp)
            gnss_idx, _ = ut.find_closest_timestamp(self.gnss_timestamps, timestamp)
            _, lidar_uv, lidar_intensity, lidar_xyz = self.lidar_data[lidar_idx]
            _, pos_ned, ori_quat = self.gnss_data[gnss_idx]

            yield Frame(
                timestamp=timestamp,
                left_img=left_img,
                disparity_img=disparity_img,
                depth_img=depth_img,
                lidar_uv=lidar_uv,
                lidar_intensity=lidar_intensity,
                lidar_xyz=lidar_xyz,
                pose=np.concatenate([pos_ned, ori_quat]),
            )
