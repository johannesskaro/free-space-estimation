"""
Play back the raw lidar point clouds of a scenario in an Open3D window.

Usage: python tools/view_lidar.py --scenario scen4_2 --data-root /path/to/data
"""
import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from free_space.ma2.dataset import LIDAR_TOPIC  # noqa: E402

typestore = get_typestore(Stores.ROS2_FOXY)

def gen_ma2_lidar_points(rosbag_path):
    lidar_data = []
    with Reader(rosbag_path) as reader:
        connections = [c for c in reader.connections if c.topic == LIDAR_TOPIC]
        assert len(connections) == 1
        for connection, timestamp, rawdata in reader.messages(connections):
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            xyz = msg.data.reshape(-1, msg.point_step)[:,:12].view(dtype=np.float32)
            intensity = msg.data.reshape(-1, msg.point_step)[:,16:20].view(dtype=np.float32)
            intensity_clipped = np.clip(intensity, 0, 100)

            lidar_data.append([timestamp, xyz, intensity_clipped])

    return lidar_data 

def vizualize_lidar_points(lidar_data, frame_delay=0.2):

    def make_wire_cube(half):
        # 8 corners
        c = half
        pts = np.array([
            [-c,-c,-c],[ c,-c,-c],[ c, c,-c],[-c, c,-c],
            [-c,-c, c],[ c,-c, c],[ c, c, c],[-c, c, c]
        ], dtype=np.float64)
        # 12 edges
        edges = np.array([
            [0,1],[1,2],[2,3],[3,0],
            [4,5],[5,6],[6,7],[7,4],
            [0,4],[1,5],[2,6],[3,7]
        ], dtype=np.int32)
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts),
            lines=o3d.utility.Vector2iVector(edges)
        )
        ls.colors = o3d.utility.Vector3dVector(np.tile([[1,1,1]], (edges.shape[0],1)))  # nearly invisible black
        return ls

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="LiDAR Stream", width=1280, height=720)

    bbox = make_wire_cube(200)
    vis.add_geometry(bbox)

    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0])
    vis.add_geometry(axes)

    vis.poll_events(); vis.update_renderer()


    ctr = vis.get_view_control()
    ctr.set_lookat([0, 0, 0])
    ctr.set_front([-1, 0, 0])  # looking along +X
    ctr.set_up([0, 0, 1])     # Z is up
    ctr.set_zoom(0.7)

    opt = vis.get_render_option()
    opt.point_size = 2.0

    cmap = plt.get_cmap('jet')

    for timestamp, xyz, intensity in lidar_data:

        xyz = np.asarray(xyz, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"xyz must be (N,3), got {xyz.shape}")
        N = xyz.shape[0]
        pcd.points = o3d.utility.Vector3dVector(xyz)

        if intensity is None:
            intensity_norm = np.ones(N)
        else:
            intensity = np.asarray(intensity).flatten()
            intensity_norm = np.clip(intensity / np.max(intensity), 0, 1)

        colors = cmap(intensity_norm)[:, :3]  # drop alpha channel
        colors = np.asarray(colors, dtype=np.float64)
        if colors.shape != (N, 3):
            raise ValueError(f"colors must be (N,3), got {colors.shape}")

        pcd.colors = o3d.utility.Vector3dVector(colors)
    
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        time.sleep(frame_delay)

    vis.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="scen4_2", help="Bag folder name, e.g. scen1, scen2_2, scen4_2, scen5, scen6")
    parser.add_argument("--data-root", default=os.environ.get("MA2_DATA_ROOT"), required=os.environ.get("MA2_DATA_ROOT") is None)
    parser.add_argument("--frame-delay", type=float, default=0.2, help="Seconds between scans")
    args = parser.parse_args()
    lidar_data = gen_ma2_lidar_points(os.path.join(args.data_root, "bags", args.scenario))
    vizualize_lidar_points(lidar_data, frame_delay=args.frame_delay)
