import json

import cv2
import numpy as np
import open3d as o3d


class RWPS:
    """Water plane segmentation: fits a plane to the lower half of the stereo point
    cloud with RANSAC and marks pixels close to that plane as water.

    The plane is checked against the previous and initial planes; if it jumps too
    much, the older plane is used instead.
    """

    def __init__(self, config_file=None) -> None:
        if config_file is not None:
            # Load configuration file
            self.set_config(config_file)

        else:
            # Use default configuration
            self.config_file = None
            self.distance_threshold = 0.01
            self.ransac_n = 3
            self.num_iterations = 1000
            self.probability = 0.99999999

            self.validity_height_thr = 0.1  # m
            self.validity_angle_thr = 5  # deg
            self.validity_min_inliers = 100

        self.prev_planemodel = None
        self.prev_height = 0
        self.prev_unitnormal = np.array([0, 1, 0])
        self.prev_mask = None

    def set_config(self, config_file):
        """Set RANSAC and plane validation parameters from a JSON file."""
        self.config_file = config_file
        config_data = json.load(open(config_file))
        self.distance_threshold = config_data["RANSAC"]["distance_threshold"]
        self.ransac_n = config_data["RANSAC"]["ransac_n"]
        self.num_iterations = config_data["RANSAC"]["num_iterations"]
        self.probability = config_data["RANSAC"]["probability"]
        self.validity_height_thr = config_data["plane_validation"]["height_thr"]
        self.validity_angle_thr = config_data["plane_validation"]["angle_thr"]
        self.validity_min_inliers = config_data["plane_validation"]["min_inliers"]

    def set_camera_params(self, cam_params, P1, camera_height=None):
        self.cam_params = cam_params
        self.P1 = P1
        self.camera_height = camera_height

    def segment_water_plane_using_point_cloud(
        self,
        depth: np.array,
    ) -> np.array:
        """
        Returns (water_mask, plane_model, succeeded). plane_model is [a, b, c, d] with
        ax + by + cz + d = 0 in the camera frame. On failure water_mask is None and a
        dummy plane is returned.
        """
        valid = True
        assert self.cam_params is not None, "Camera parameters are not provided."
        if self.config_file is None:
            print(
                "Warning: Configuration file is not provided. Using default parameters."
            )
        (H, W) = depth.shape
        self.shape = (H, W)

        inlier_mask = np.zeros((H, W))
        inlier_mask[H // 2 :, :] = 1

        masked_depth = np.where(inlier_mask, depth, 0).astype(np.float32)

        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            W, H, self.cam_params["fx"], self.cam_params["fy"], self.cam_params["cx"], self.cam_params["cy"]
        )

        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(masked_depth),
            intrinsics,
            stride=10,
            project_valid_depth_only=True,
            depth_scale=1.0,
            depth_trunc=200.0
        )

        points_3d = np.asarray(pcd.points)

        if len(pcd.points) < self.ransac_n:
            print("RWPS failed. Not enough points to segment plane")
            self.prev_planemodel = None
            valid = False
            return None, np.array([0, 1, 0, 1]), valid

        plane_model, _ = pcd.segment_plane(
            distance_threshold=self.distance_threshold,
            ransac_n=self.ransac_n,
            num_iterations=self.num_iterations,
        )

        if not plane_model.any():
            print("RWPS failed. No plane found in RANSAC")
            valid = False
            return None, np.array([0, 1, 0, 1]), valid

        normal = plane_model[:3]
        d = plane_model[3]
        normal_length = np.linalg.norm(normal)
        unit_normal = normal / normal_length
        height = d / normal_length

        if self.prev_planemodel is None:
            self.init_planemodel = plane_model
            self.init_height = height
            self.init_unitnormal = unit_normal

        mask = self.get_water_mask_from_plane_model(points_3d, plane_model)

        if self.prev_planemodel is not None:
            prev_valid = self.validity_check(
                self.prev_height, self.prev_unitnormal, height, unit_normal
            )
            init_valid = self.validity_check(
                self.init_height, self.init_unitnormal, height, unit_normal
            )

            if prev_valid and not init_valid:
                mask = self.get_water_mask_from_plane_model(
                    points_3d, self.prev_planemodel
                )

            elif not prev_valid and not init_valid:
                mask = self.get_water_mask_from_plane_model(
                    points_3d, self.init_planemodel
                )

        self.prev_planemodel = plane_model
        self.prev_height = height
        self.prev_unitnormal = unit_normal
        self.prev_mask = mask
        return mask.astype(np.uint8), plane_model, valid

    def plot_3d_pcd(self, left_img, depth_img, stride=1):
        H_full, W_full = depth_img.shape

        # Keep only bottom 2/3 of the image
        start_row = H_full // 3
        left_img_cropped = left_img[start_row:, :]
        depth_img_cropped = depth_img[start_row:, :]

        H, W = depth_img_cropped.shape

        # Convert BGR to RGB
        left_img_rgb = cv2.cvtColor(left_img_cropped, cv2.COLOR_BGR2RGB)

        # Create Open3D images
        img_o3d = o3d.geometry.Image(left_img_rgb)
        depth_o3d = o3d.geometry.Image(depth_img_cropped.astype(np.float32))

        # Adjust intrinsics based on cropping
        fx = self.cam_params["fx"]
        fy = self.cam_params["fy"]
        cx = self.cam_params["cx"]
        cy = self.cam_params["cy"] - start_row  # Shift principal point

        intrinsics = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            img_o3d,
            depth_o3d,
            depth_scale=1.0,
            depth_trunc=200.0,
            convert_rgb_to_intensity=False
        )

        # Generate point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            intrinsics,
            project_valid_depth_only=True
        )

        # Optional: downsample point cloud
        if stride > 1:
            pcd = pcd.voxel_down_sample(voxel_size=stride * 0.01)

        # Create a coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

        plane_pcd = self.create_plane_pointcloud(
            x_range=(-20, 20), y_range=(-1, 2), resolution=80
        )

        # Visualize
        o3d.visualization.draw_geometries([pcd, plane_pcd, coord_frame])

    def create_plane_pointcloud(self, x_range, y_range, resolution=10):
        """
        Create a point cloud of points lying on a plane model (a, b, c, d).
        """

        a, b, c, d = self.prev_planemodel

        # Generate grid in X-Y
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        xx, yy = np.meshgrid(x, y)
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)

        # Calculate corresponding Z
        if c == 0:
            raise ValueError("Plane normal z-component is zero, cannot solve for z.")
        zz = (-a * xx - b * yy - d) / c

        # Stack into (N, 3) points
        points = np.vstack((xx, yy, zz)).T

        # Create PointCloud object
        plane_pcd = o3d.geometry.PointCloud()
        plane_pcd.points = o3d.utility.Vector3dVector(points)

        # Color all points pink
        pink_color = np.array([[1.0, 0.4, 0.7]] * points.shape[0])
        plane_pcd.colors = o3d.utility.Vector3dVector(pink_color)

        return plane_pcd

    def get_image_mask(self, xyz, cam_params, shape):
        H, W = shape
        cx, cy = cam_params["cx"], cam_params["cy"]
        fx, fy = cam_params["fx"], cam_params["fy"]

        X, Y, Z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        Z = np.where(Z == 0, 1e-6, Z)  # Avoid divide-by-zero

        u = np.round((fx * X / Z) + cx).astype(int)
        v = np.round((fy * Y / Z) + cy).astype(int)

        # Only keep valid indices
        valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v = u[valid], v[valid]

        mask = np.zeros((H, W), dtype=np.uint8)
        mask[v, u] = 1
        return mask

    def validity_check(self, prev_height, prev_normal, current_height, current_normal):
        if abs(prev_height - current_height) > self.validity_height_thr:
            return False
        if np.dot(prev_normal, current_normal) < np.cos(np.deg2rad(self.validity_angle_thr)):
            return False
        return True

    def get_water_mask_from_plane_model(self, points_3d, plane_model):
        normal = plane_model[:3]
        d = plane_model[3]
        H, W = self.shape
        normal_length = np.linalg.norm(normal)
        unit_normal = normal / normal_length
        height = d / normal_length
        distances = np.dot(points_3d, unit_normal) + height

        inlier_indices_1d = np.where(np.abs(distances) < self.distance_threshold)[0]
        inlier_points = points_3d[inlier_indices_1d]
        mask = self.get_image_mask(inlier_points, self.cam_params, (H, W))

        return mask

    def get_plane_model(self):
        return self.prev_planemodel
