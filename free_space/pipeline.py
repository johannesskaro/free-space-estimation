import time
from dataclasses import dataclass

import numpy as np

from free_space import utils as ut
from free_space.fastsam import FastSAMSeg
from free_space.rwps import RWPS
from free_space.stixels import Stixels
from free_space.temporal_filtering import TemporalFiltering
from free_space.yolo import YoloSeg


@dataclass
class FrameResult:
    stixel_footprints: np.ndarray   # (num_stixels, 2) (forward, right) in meters, camera frame
    water_mask: np.ndarray          # final water mask used for the stixels
    water_mask_failure: bool        # True if the segmentation was rejected this frame
    rwps_mask: np.ndarray           # water mask from the plane fit alone (None if it failed)
    boat_mask: np.ndarray           # YOLO boat mask
    lidar_uv: np.ndarray            # lidar points inside the image
    lidar_xyz: np.ndarray
    runtime_ms: float


class FreeSpacePipeline:
    """
    Free-space estimation for one camera, one frame at a time:

      1. RWPS: fit the water plane to the stereo depth -> rough water mask
      2. FastSAM: segment the image, keep segments that overlap the rough mask -> water mask
      3. YOLO: detect boats (used to mark dynamic stixels and cut boats out of the water)
      4. Temporal filtering over the last frames, plus a sanity check of the result
      5. Stixels: obstacle columns on the water edge, with depth fused from lidar,
         stereo and the previous frame

    Per-stixel state (validity, dynamic flags, depth variance, ...) is available on
    `pipeline.stixels` after each call to `process`.
    """

    def __init__(self, K, baseline, image_size, t_body_to_cam, R_body_to_cam,
                 fastsam_weights="weights/FastSAM-x.pt",
                 yolo_weights="weights/yolo11n-seg.pt",
                 rwps_config="configs/rwps.json",
                 num_stixels=192):
        self.height, self.width = image_size
        cam_params = {"cx": K[0,2], "cy": K[1,2], "fx": K[0,0], "fy": K[1,1], "b": baseline}
        P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))

        self.yolo = YoloSeg(model_path=yolo_weights)
        self.fastsam = FastSAMSeg(model_path=fastsam_weights)
        self.rwps = RWPS(config_file=rwps_config)
        self.rwps.set_camera_params(cam_params, P1)
        self.temporal_filtering = TemporalFiltering(N=3)
        self.stixels = Stixels(num_stixels=num_stixels, img_shape=image_size, cam_params=cam_params,
                               t_body_to_cam=t_body_to_cam, R_body_to_cam=R_body_to_cam)

        self.prev_timestamp = 0
        self.prev_pose = np.array([0, 0, 0, 0, 0, 0, 1], dtype=float)  # identity quaternion

    def process(self, timestamp, left_img, disparity_img, depth_img, lidar_uv, lidar_xyz, pose):
        """
        timestamp: ns. left_img: (H, W, 3) BGR. disparity_img: px. depth_img: m.
        lidar_uv: (N, 2) lidar points projected into the image, lidar_xyz: (N, 3)
        the same points in the camera frame. pose: [x, y, z, qx, qy, qz, qw] in NED.
        """
        dt = (timestamp - self.prev_timestamp) / (10 ** 9)
        pose_prev = self.prev_pose
        self.prev_timestamp = timestamp
        self.prev_pose = pose.copy()

        left_img = np.ascontiguousarray(left_img, dtype=np.uint8)
        disparity_img = np.ascontiguousarray(disparity_img, dtype=np.float32)
        depth_img = np.ascontiguousarray(depth_img, dtype=np.float32)
        depth_img[depth_img > 100] = np.nan

        start_time = time.time()

        # Water segmentation
        rwps_mask, plane_params, rwps_succeeded = self.rwps.segment_water_plane_using_point_cloud(depth_img)
        contour_mask, upper_contour_mask, water_mask = self.fastsam.get_contours_and_water_mask(left_img, rwps_mask)
        if not rwps_succeeded:
            water_mask = ut.get_water_mask_from_contour_mask(contour_mask)

        # Boat detection
        boat_mask = self.yolo.get_boat_mask(left_img)

        # Temporal filtering
        water_mask_filtered = self.temporal_filtering.filter(water_mask)
        water_mask_failure = self.temporal_filtering.detect_segmentation_failure(water_mask_filtered, plane_params)
        water_mask_refined = self.yolo.refine_water_mask(boat_mask, water_mask_filtered)

        # Stixels
        lidar_uv, lidar_xyz = ut.filter_point_cloud_by_image(lidar_uv, lidar_xyz, self.height, self.width)

        stixel_footprints = self.stixels.run_stixel_pipeline(
            left_img=left_img,
            water_mask=water_mask_refined,
            water_mask_failure=water_mask_failure,
            disparity_img=disparity_img,
            depth_img=depth_img,
            upper_contours=upper_contour_mask,
            xyz_proj=lidar_uv,
            xyz_c=lidar_xyz,
            pose_prev=pose_prev,
            pose_curr=pose,
            dt=dt,
            boat_mask=boat_mask,
        )

        runtime_ms = (time.time() - start_time) * 1000

        return FrameResult(
            stixel_footprints=stixel_footprints,
            water_mask=water_mask_refined,
            water_mask_failure=water_mask_failure,
            rwps_mask=rwps_mask,
            boat_mask=boat_mask,
            lidar_uv=lidar_uv,
            lidar_xyz=lidar_xyz,
            runtime_ms=runtime_ms,
        )
