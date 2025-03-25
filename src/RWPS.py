import numpy as np
from sklearn.linear_model import (
    LinearRegression,
    RANSACRegressor,
)
import json
import cv2
import open3d as o3d

class RWPS:
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
        self.prev_planemodel_disp = None
        self.prev_height = 0
        self.prev_unitnormal = np.array([0, 1, 0])
        self.prev_mask = None
        self.prev_residual_threshold = None
        self.prev_mask_ds = None
        self.invalid = None
        self.counter = 0
        self.sigma_e = 1 / 2

    def set_invalid(self, p1, p2, shape):
        self.invalid = invalid_mask(p1, p2, shape)
        return self.invalid
    
    def set_config(self, config_file):
        self.set_config_xyz(config_file)

    def set_config_xyz(self, config_file):
        """
        Set parameters for RANSAC plane segmentation using 3D point cloud
        """
        self.config_file = config_file
        config_data = json.load(open(config_file))
        self.distance_threshold = config_data["RANSAC"]["distance_threshold"]
        self.ransac_n = config_data["RANSAC"]["ransac_n"]
        self.num_iterations = config_data["RANSAC"]["num_iterations"]
        self.probability = config_data["RANSAC"]["probability"]
        self.validity_height_thr = config_data["plane_validation"]["height_thr"]
        self.validity_angle_thr = config_data["plane_validation"]["angle_thr"]
        self.validity_min_inliers = config_data["plane_validation"]["min_inliers"]
        self.initial_roll = None
        self.initial_pitch = None
        self.disp_deviations_inliers_array = {}
        self.counter = 0

    def set_initial_pitch(self, pitch):
        self.initial_pitch = pitch

    def set_initial_roll(self, roll):
        self.initial_roll = roll

    def set_camera_params(self, cam_params, P1, camera_height=None):
        self.cam_params = cam_params
        self.P1 = P1
        self.camera_height = camera_height

    def get_image_coords(self, points_3d):
        H, W = self.shape
        cam_params = self.cam_params
        cx, cy = cam_params["cx"], cam_params["cy"]
        fx, fy = cam_params["fx"], cam_params["fy"]

        X, Y, Z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]

        Z = np.where(Z == 0, 1e-6, Z)
        u = (fx * X) / Z + cx
        v = (fy * Y) / Z + cy

        u_px = np.round(u).astype(int)
        v_px = np.round(v).astype(int)

        valid_mask = (u_px >= 0) & (u_px < W) & (v_px >= 0) & (v_px < H)
        image_coords_px = np.stack((u_px[valid_mask], v_px[valid_mask]), axis=-1)

        return image_coords_px

    def segment_water_plane_using_point_cloud(
        self,
        depth: np.array,
    ) -> np.array:

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
        
        #print("Points in PCD:", len(pcd.points))


        #coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        #    size=1.0, origin=[0, 0, 0]
        #)
        #o3d.visualization.draw_geometries([pcd, coord_frame])

        # Check if there are enough points to segment plane
        if len(pcd.points) < self.ransac_n:
            print("Not enough points to segment plane")
            self.prev_planemodel = None
            valid = False
            return None, np.array([0, 1, 0, 1]), valid

        plane_model, _ = pcd.segment_plane(
            distance_threshold=self.distance_threshold,
            ransac_n=self.ransac_n,
            num_iterations=self.num_iterations,
            probability=self.probability,
        )

        if not plane_model.any():  # if plane_model is empty
            print("No plane found in RANSAC")
            valid = False
            return None, np.array([0, 1, 0, 1]), valid

        normal = plane_model[:3]
        d = plane_model[3]
        normal_length = np.linalg.norm(normal)
        unit_normal = normal / normal_length
        height = d / normal_length  # abs(d) #/ normal_length

        # if self.prev_planemodel is not np.array([0,1,0]):
        if self.prev_planemodel is not None:
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
                # if not prev_valid and not init_valid:
                mask = self.get_water_mask_from_plane_model(
                    points_3d, self.init_planemodel
                )

        self.prev_planemodel = plane_model
        self.prev_height = height
        self.prev_unitnormal = unit_normal
        self.prev_mask = mask
        return mask.astype(np.uint8), plane_model, valid
    
    def get_segmentation_mask_from_plane_model(self, points_3d, plane_model):

        normal = plane_model[:3]
        d = plane_model[3]
        normal_length = np.linalg.norm(normal)
        unit_normal = normal / normal_length
        height = d / normal_length

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
                # if not prev_valid and not init_valid:
                mask = self.get_water_mask_from_plane_model(
                    points_3d, self.init_planemodel
                )

        return mask

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
        if np.dot(prev_normal, current_normal) < np.cos(self.validity_angle_thr):
            return False
        return True

    def get_pitch(self, normal_vec=None):
        if normal_vec is None:
            normal_vec = self.prev_unitnormal
        normal_vec = normal_vec / np.linalg.norm(normal_vec)
        [a, b, c] = normal_vec
        pitch = np.arctan2(c, b)
        return pitch

    def get_roll(self, normal_vec=None):
        if normal_vec is None:
            normal_vec = self.prev_unitnormal
        normal_vec = normal_vec / np.linalg.norm(normal_vec)
        [a, b, c] = normal_vec
        roll = np.arctan2(a, b)
        return roll
    
    def get_water_mask_from_plane_model(self, points_3d, plane_model):
        normal = plane_model[:3]
        d = plane_model[3]
        H, W = self.shape
        normal_length = np.linalg.norm(normal)
        unit_normal = normal / normal_length
        height = d / normal_length
        distances = np.dot(points_3d, unit_normal) + height

        # distances = normal.dot(points_3d.T) + d
        inlier_indices_1d = np.where(np.abs(distances) < self.distance_threshold)[0]
        inlier_points = points_3d[inlier_indices_1d]
        mask = self.get_image_mask(inlier_points, self.cam_params, (H, W))

        #inlier_indices = np.unravel_index(inlier_indices_1d, (H, W))
        #mask = np.zeros((H, W))
        #mask[inlier_indices] = 1


        return mask
    


    def get_plane_model(self):
        return self.prev_planemodel
    
    def get_horizon(self, normal_vec=None):
        if normal_vec is None and self.prev_unitnormal is not None:
            normal_vec = self.prev_unitnormal
        elif normal_vec is None and self.prev_unitnormal is None:
            print("No plane parameters.")
            return None, None

        [a, b, c] = normal_vec
        fy = self.cam_params["fy"]
        cx = self.cam_params["cx"]
        cy = self.cam_params["cy"]
        W = self.shape[1]
        H = self.shape[0]

        if np.abs(b) < 1e-5:
            print("[WARNING] Horizon estimation skipped: b ~ 0 (horizontal normal)")
            return np.array([0, H // 2]), np.array([W, H // 2]), H // 2  # fallback to middle


        k = a * cx + b * cy - c * fy
        y0 = (1 / b) * (k - a * 0)
        yW = (1 / b) * (k - a * W)

        y0 = np.clip(y0, 0, H - 1)
        yW = np.clip(yW, 0, H - 1)

        p1 = np.array([0, int(round(y0))])
        p2 = np.array([W - 1, int(round(yW))])

        # Check slope sanity
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if dx == 0 or np.abs(dy / dx) > 10:
            print("[WARNING] Horizon slope too steep or invalid — fallback to flat horizon")
            return np.array([0, H // 2]), np.array([W, H // 2]), H // 2

        horizon_cutoff = int(min(p1[1], p2[1]) - 50)
        horizon_cutoff = np.clip(horizon_cutoff, 0, H - 1)

        return p1, p2, horizon_cutoff

    

def invalid_mask(p1, p2, shape):
    # Define the points
    H, W = shape
    x1, y1 = p1
    x2, y2 = p2

    # Calculate the slope (m) and intercept (b) of the line
    if x2 - x1 == 0:
        m = 99999999999999
        b = x1
    else:
        m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1

    # Create the invalid_mask
    invalid_mask = np.zeros((H, W), dtype=bool)

    # Generate a grid of coordinates
    x_coords, y_coords = np.meshgrid(np.arange(W), np.arange(H))

    # Calculate the y values of the line at each x coordinate
    y_line = m * x_coords + b

    # Mark everything under the line as True
    invalid_mask = y_coords >= y_line

    return invalid_mask