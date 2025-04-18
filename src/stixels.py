import numpy as np
import cupy as cp
from shapely.geometry import Polygon
from collections import deque
import utilities as ut
import cv2
from numba import njit, prange
import matplotlib.pyplot as plt
import time
from scipy.spatial.transform import Rotation as R
import math


class Stixels:
        
    def __init__(self, num_stixels, img_shape, cam_params, t_body_to_cam, R_body_to_cam, min_stixel_height=20, max_range=60):

        self.num_stixels = num_stixels
        self.img_shape = img_shape
        self.cam_params = cam_params
        self.stixel_width  = int(img_shape[1] // self.num_stixels)
        self.min_stixel_height = min_stixel_height
        self.max_range = max_range

        self.height_base_list = np.zeros((self.num_stixels, 2), dtype=int)
        self.stixel_lidar_depths = np.zeros(self.num_stixels, dtype=float)
        self.stixel_stereo_depths = np.zeros(self.num_stixels, dtype=float)
        self.stixel_fused_depths = np.zeros(self.num_stixels, dtype=float)
        self.stixel_fused_depths_var = np.full(self.num_stixels, np.inf ,dtype=float)
        self.stixel_footprints = np.full((num_stixels, 2), 0, dtype=float)
        self.stixel_has_measurement = np.full(num_stixels, False, dtype=bool)
        self.stixel_validity = np.full(self.num_stixels, False, dtype=bool)
        self.dynamic_stixel_list = np.full(num_stixels, False, dtype=bool)
        self.using_prop_depth = np.full(self.num_stixels, False, dtype=bool)

        self.prev_height_base_list = np.zeros((self.num_stixels, 2), dtype=int)
        self.prev_stixel_lidar_depths = np.zeros(self.num_stixels, dtype=float)
        self.prev_stixel_stereo_depths = np.zeros(self.num_stixels, dtype=float)
        self.prev_stixel_fused_depths_var = np.full(self.num_stixels, np.inf, dtype=float)
        self.prev_stixel_footprints = np.full((num_stixels, 2), 0, dtype=float)
        self.prev_stixel_footprints_curr_frame = np.full((num_stixels, 2), 0, dtype=float)
        self.prev_stixel_has_measurement = np.full(num_stixels, False, dtype=bool)
        self.prev_stixel_validity = np.full(self.num_stixels, False, dtype=bool)
        self.prev_dynamic_stixel_list = np.full(num_stixels, False, dtype=bool)
        self.prev_using_prop_depth = np.full(self.num_stixels, False, dtype=bool)

        self.association_depth = np.full(num_stixels, -1, dtype=int)
        self.association_height = np.full(num_stixels, -1, dtype=int)

        self.R_body_to_cam = np.array(R_body_to_cam)
        self.t_body_to_cam = np.array(t_body_to_cam)

        self.projection_rays = self.get_projection_rays()

    def run_stixel_pipeline(self, water_mask, disparity_img, depth_img, upper_contours, xyz_proj, xyz_c, pose_prev, pose_curr, boat_mask=None):

        self.prev_stixel_footprints = self.stixel_footprints.copy()
        self.prev_stixel_lidar_depths = self.stixel_lidar_depths.copy()
        self.prev_stixel_fused_depths_var = self.stixel_fused_depths_var.copy()
        self.prev_stixel_has_measurement = self.stixel_has_measurement.copy()
        self.prev_using_prop_depth = self.using_prop_depth.copy()

        delta_heading = ut.get_delta_heading(pose_prev, pose_curr)

        self.create_stixels_in_image(water_mask, disparity_img, depth_img, upper_contours, boat_mask)

        self.get_stixel_depths_from_lidar(xyz_proj, xyz_c)

        self.transform_prev_stixels_into_curr_frame(pose_prev, pose_curr)

        self.associate_prev_stixels(delta_heading)

        self.recursive_height_filter()

        self.recursive_depth_filter(delta_heading)

        self.get_stixel_BEV_footprints()




        return self.stixel_footprints.copy()


    def create_stixels_in_image(self, water_mask, disparity_img, depth_img, upper_contours, boat_mask=None):

        free_space_boundary = get_free_space_boundary(water_mask)
        #start_time = time.time()
        SSM, stereo_depth = self.create_segmentation_score_map(disparity_img, depth_img, free_space_boundary, upper_contours)
        #end_time = time.time()
        #runtime_ms = (end_time - start_time) * 1000
        #print(f"SSM total: {runtime_ms:.2f} ms")
        
        top_boundary = get_optimal_height_numba(SSM, stereo_depth, free_space_boundary, self.num_stixels)
        self.get_stixel_height_base_list(free_space_boundary, top_boundary, boat_mask)

    
    def get_prev_stixel_footprint(self):
        return self.prev_stixel_footprints
    
    def get_projection_rays(self):

        fx = self.cam_params["fx"]
        cx = self.cam_params["cx"]

        projection_rays = np.zeros((self.num_stixels, 3))

        for n in range(self.num_stixels):
            u = (n + 1) * self.stixel_width - self.stixel_width // 2
            x0 = (u - cx) / fx
            ray = np.array([-x0, 1, 0])
            projection_rays[n] = ray

        return projection_rays


    def associate_prev_stixels(self, delta_heading, ang_thres_deg=1, z_diff_thres=1):
        ang_thres = np.deg2rad(ang_thres_deg)
        N = self.num_stixels
        association_depth = np.full(N, -1, dtype=int)
        association_heigth = np.full(N, -1, dtype=int)

        cam_fov = 110
        RAY_SPACING = cam_fov / N
        DELTA_IDX = int(round(delta_heading) / RAY_SPACING)

        relevant_prev_points_list = self.prev_stixel_has_measurement.copy()
        pts = self.prev_stixel_footprints_curr_frame.copy()
        indices = np.where(relevant_prev_points_list)[0]

        z_lidar = self.stixel_lidar_depths.copy()

        for n, ray in enumerate(self.projection_rays):

            #if self.dynamic_stixel_list[n] == True:
             #   continue

            if pts.size == 0 or indices.size == 0:
                continue

            idx_pred = n + DELTA_IDX
            pos = np.abs(indices - idx_pred).argmin()
            pos_minus = np.clip(pos - 1, 0, len(indices) - 1)
            pos_plus = np.clip(pos + 1, 0, len(indices) - 1)

            idx = indices[pos]
            idx_minus = np.clip(indices[pos_minus], 0, N - 1)
            idx_plus = np.clip(indices[pos_plus], 0, N - 1)

            d_n_minus_1 = ut.angular_error(pts[idx_minus], ray)
            d_n = ut.angular_error(pts[idx], ray)
            d_n_plus_1 = ut.angular_error(pts[idx_plus], ray)

            #print("d_n_minus_1: ", d_n_minus_1)
            #print("d_n: ", d_n)
            #print("d_n_plus_1: ", d_n_plus_1)
            #print(" ----- ")

            iterating = True

            if d_n <= d_n_minus_1 and d_n <= d_n_plus_1:
                if d_n < ang_thres:
                    association_depth[n] = idx
                    if self.prev_dynamic_stixel_list[idx] == False or self.dynamic_stixel_list[n] == False:
                        z_diff = np.abs(pts[idx, 0] - z_lidar[n])
                        if z_diff < z_diff_thres:
                            association_heigth[n] = idx

            elif d_n_minus_1 < d_n and d_n_minus_1 < d_n_plus_1:
                i = 2
                d_n_minus_i_prev = d_n_minus_1
                while iterating:
                    #print("minus", n)
                    pos_minus = pos - i
                    if pos_minus < 0:
                        idx = indices[0]
                        if d_n_minus_i_prev < ang_thres:
                            association_depth[n] = idx
                            if self.prev_dynamic_stixel_list[idx] == False or self.dynamic_stixel_list[n] == False:
                                z_diff = np.abs(pts[idx, 0] - z_lidar[n])
                                if z_diff < z_diff_thres:
                                    association_heigth[n] = idx
                                    
                        iterating = False
                        break
                    else:
                        idx = indices[pos_minus]

                    d_n_minus_i = ut.angular_error(pts[idx], ray)
                    if d_n_minus_i < d_n_minus_i_prev:
                        d_n_minus_i_prev = d_n_minus_i
                        i += 1
                        continue
                    else:
                        if d_n_minus_i_prev < ang_thres:
                            idx = indices[pos_minus + 1]
                            association_depth[n] = idx
                            if self.prev_dynamic_stixel_list[idx] == False or self.dynamic_stixel_list[n] == False:
                                z_diff = np.abs(pts[idx, 0] - z_lidar[n])
                                if z_diff < z_diff_thres:
                                    association_heigth[n] = idx
                        iterating = False
                        break

            elif d_n_plus_1 < d_n and d_n_plus_1 < d_n_minus_1:
                i = 2
                d_n_plus_i_prev = d_n_plus_1
                while iterating:
                    #print("plus", n)
                    pos_plus = pos + i
                    if pos_plus >= len(indices):
                        idx = indices[-1]
                        if d_n_plus_i_prev < ang_thres:
                            association_depth[n] = idx
                            if self.prev_dynamic_stixel_list[idx] == False or self.dynamic_stixel_list[n] == False:
                                z_diff = np.abs(pts[idx, 0] - z_lidar[n])
                                if z_diff < z_diff_thres:
                                    association_heigth[n] = idx
                        iterating = False
                        break
                    else:
                        idx = indices[pos_plus]

                    d_n_plus_i = ut.angular_error(pts[idx], ray)
                    if d_n_plus_i < d_n_plus_i_prev:
                        d_n_plus_i_prev = d_n_plus_i
                        i += 1
                        continue
                    else:
                        if d_n_plus_i_prev < ang_thres:
                            idx = indices[pos_plus - 1]
                            association_depth[n] = idx
                            if self.prev_dynamic_stixel_list[idx] == False or self.dynamic_stixel_list[n] == False:
                                z_diff = np.abs(pts[idx, 0] - z_lidar[n])
                                if z_diff < z_diff_thres:
                                    association_heigth[n] = idx
                        iterating = False
                        break
            else:
                print("Error associating previous Stixels for index", n)

        #print(association_list)

        self.association_depth = association_depth
        self.association_height = association_heigth

    

    def recursive_height_filter(self, alpha=0.7):

        for n in range(self.num_stixels):
            if self.association_height[n] != -1:
                idx = self.association_height[n]

                v_top_curr = self.height_base_list[n, 0]
                v_top_prev = self.prev_height_base_list[idx, 0]

                v_top_lp = alpha * v_top_prev + (1 - alpha) * v_top_curr

                #print("v_top_curr: ", v_top_curr)
                #print("v_top_lp: ", v_top_lp)

                self.height_base_list[n, 0] = v_top_lp

    def recursive_depth_filter(self, delta_heading):

        self.using_prop_depth = np.full(self.num_stixels, False, dtype=bool)
        self.stixel_validity = np.full(self.num_stixels, False, dtype=bool)

        C_pose = np.array([[0.01, 0, 0], 
                           [0, 0.01, 0],
                           [0, 0, 0.001225]])

        fx = self.cam_params["fx"]
        b = self.cam_params["b"]

        potential_prop_list = []

        for n in range(self.num_stixels):
            z_lidar = self.stixel_lidar_depths[n]
            z_stereo = self.stixel_stereo_depths[n]
            
            if np.isnan(z_lidar) or np.isinf(z_lidar):
                z_lidar = 0
                var_lidar = np.inf
                has_lidar = False
            else:
                sigma_lidar = 0.01
                var_lidar = sigma_lidar**2
                has_lidar = True

            if np.isnan(z_stereo) or np.isinf(z_stereo):
                z_stereo = 0
                var_stereo = np.inf
            else:
                sigma_px = 0.5 
                sigma_stereo = sigma_px * z_stereo**2 / (fx * b)
                var_stereo = sigma_stereo**2
            
            # Prediction step

            if self.association_depth[n] != -1:
                idx = self.association_depth[n]

                if not has_lidar and self.prev_using_prop_depth[idx]:
                    potential_prop_list.extend([n, idx])

                var_fused_prev = self.prev_stixel_fused_depths_var[idx]

                p = self.prev_stixel_footprints[idx]
                J_z_motion = np.array([1, 0, -p[0]*np.sin(delta_heading) + p[1]*np.cos(delta_heading)])
                var_motion = J_z_motion @ C_pose @ J_z_motion.T 

                z_prop = self.prev_stixel_footprints_curr_frame[idx, 0]
                var_prop = var_fused_prev + var_motion
                #print("var_motion: ", var_motion)

            else:
                z_prop = 0
                var_prop = np.inf

            # Update step

            if z_stereo < 10:

                var_fused = 1 / ((1 / var_lidar) + (1 / var_stereo) + (1 / var_prop) + 1e-10)
                z_fused = var_fused * ((z_lidar / var_lidar) + (z_stereo / var_stereo) + (z_prop / var_prop))

            else:

                var_fused = 1 / ((1 / var_lidar) + (1 / var_prop) + 1e-10)
                z_fused = var_fused * ((z_lidar / var_lidar) + (z_prop / var_prop))


            # If all depths are invalid
            if z_lidar <= 0:
                #self.stixel_has_measurement[n] = False

                if z_prop > 0:
                    #print("using prop lidar depth")
                    self.using_prop_depth[n] = True

            #else:
                #self.stixel_has_measurement[n] = True
                #print(z_fused)

            if z_fused <= 0:
                self.stixel_validity[n] = False
                self.stixel_has_measurement[n] = False
                z_fused = self.max_range

            else:
                self.stixel_validity[n] = True
                self.stixel_has_measurement[n] = True


            self.stixel_fused_depths[n] = z_fused
            self.stixel_fused_depths_var[n] = var_fused


            #print("var_lidar: ", var_lidar)
            #print("var_stereo: ", var_stereo)
            #print("var_prop: ", var_prop)
            #print("var_fused: ", var_fused)
            #print("----------")



    
    def transform_prev_stixels_into_curr_frame_old(self, prev_pose, curr_pose):

        p_cam_prev = self.prev_stixel_footprints

        if p_cam_prev.size == 0:
            return np.empty((0, 2))

        #pos_prev = np.array([prev_pose[1], prev_pose[0]]) # Convert from NED to XZ-plane
        #pos_curr = np.array([curr_pose[1], curr_pose[0]]) # Convert from NED to XZ-plane
        pos_prev = np.array([prev_pose[0], prev_pose[1]]) 
        pos_curr = np.array([curr_pose[0], curr_pose[1]]) 
        ori_prev = prev_pose[3:]
        ori_curr = curr_pose[3:]

        # + np.pi is only for MA2, since it is driving backwards, remove for BlueBoat
        heading_prev = R.from_quat(ori_prev).as_euler('xyz', degrees=False)[2] + np.pi
        heading_curr = R.from_quat(ori_curr).as_euler('xyz', degrees=False)[2] + np.pi

        #print("heading_curr: ", heading_curr * 180/np.pi)
        #print("Direction: ", heading_curr * 180/np.pi - heading_prev * 180/np.pi)

        R_prev = ut.rotation_matrix(heading_prev)
        R_curr = ut.rotation_matrix(heading_curr)

        t_world_to_cam_prev = R_prev.dot(self.t_body_to_cam) + pos_prev
        t_world_to_cam_curr = R_curr.dot(self.t_body_to_cam) + pos_curr

        #p_cam_prev = np.asarray(prev_stixel_points)
        p_world_prev = (R_prev.dot(p_cam_prev.T)).T + t_world_to_cam_prev
        p_cam_curr = (R_curr.T.dot((p_world_prev - t_world_to_cam_curr).T)).T

        self.prev_stixel_footprints_curr_frame = p_cam_curr.copy()


    def transform_prev_stixels_into_curr_frame(self, prev_pose, curr_pose):
        """
        prev_pose, curr_pose: [x, y, z?, qx, qy, qz, qw]
        we only use x,y and yaw around z.
        """

        pts_prev = self.prev_stixel_footprints  # shape (N, 2) in camera coords
        if pts_prev.size == 0:
            self.prev_stixel_footprints_curr_frame = np.empty((0, 2))
            return

        # 1) Extract planar poses
        x_prev, y_prev = prev_pose[0], prev_pose[1]
        x_curr, y_curr = curr_pose[0], curr_pose[1]

        # SciPy expects quaternion [x, y, z, w]
        yaw_prev = R.from_quat(prev_pose[3:]).as_euler('xyz', degrees=False)[2] + np.pi
        yaw_curr = R.from_quat(curr_pose[3:]).as_euler('xyz', degrees=False)[2] + np.pi

        # 2) Build 3×3 homogeneous transforms for each camera in world coords
        def make_homog(x, y, yaw):
            c, s = np.cos(yaw), np.sin(yaw)
            return np.array([
                [   c, s,   x],
                [   -s, c,   y],
                [   0,  0,   1],
            ])

        T_prev = make_homog(x_prev, y_prev, yaw_prev)
        T_curr = make_homog(x_curr, y_curr, yaw_curr)

        # 3) Compute relative transform from prev_cam to curr_cam:
        #    we want  T_rel so that  [p]₍curr₎ = T_rel @ [p]₍prev₎
        T_rel = np.linalg.inv(T_curr) @ T_prev

        # 4) Apply to all points (in homogeneous coordinates)
        N = pts_prev.shape[0]
        pts_h = np.hstack([pts_prev, np.ones((N,1))])       # shape (N,3)
        pts_curr_h = (T_rel @ pts_h.T).T                   # shape (N,3)

        # 5) Store just the x,y back
        self.prev_stixel_footprints_curr_frame = pts_curr_h[:, :2]

    

    def get_stixel_height_base_list(self, free_space_boundary, top_boundary, boat_mask=None):

        self.prev_height_base_list = self.height_base_list.copy()
        self.prev_dynamic_stixel_list = self.dynamic_stixel_list.copy()

        fsb_2d = free_space_boundary.reshape(self.num_stixels, self.stixel_width)
        v_f_array = np.median(fsb_2d, axis=1).astype(int) 
        
        v_top_array = top_boundary.astype(int).copy()
        
        dist = v_f_array - v_top_array
        mask = dist < self.min_stixel_height
        v_top_array[mask] = v_f_array[mask] - self.min_stixel_height

        height_base_list = np.column_stack((v_top_array, v_f_array))

        self.height_base_list[:] = height_base_list

        if boat_mask is not None:
            self.dynamic_stixel_list = compute_dynamic_stixels(v_top_array, v_f_array, boat_mask, self.stixel_width, self.num_stixels)

        return self.height_base_list
    
    
    def get_stixel_depths_from_lidar(self, xyz_proj, xyz_c):

        

        stixel_tops = self.height_base_list[:, 0]
        stixel_bases = self.height_base_list[:, 1]

        stixel_depths, count = assign_points_to_stixels_numba(
        xyz_proj, xyz_c,
        stixel_tops, stixel_bases,
        self.num_stixels, self.stixel_width
    )

        depths = get_percentile_all(stixel_depths, self.num_stixels, count)

        self.stixel_lidar_depths = depths




    def get_stixel_BEV_footprints(self):

        X = np.full(self.num_stixels, np.nan)
        Y = np.full(self.num_stixels, np.nan)
        Z = np.full(self.num_stixels, np.nan)
        Z_invalid = np.array([], dtype=int)

        for n, stixel in enumerate(self.height_base_list):
            
            X[n] = (n + 1) * self.stixel_width - self.stixel_width // 2
            Y[n] = stixel[1]
            Z[n] = self.stixel_fused_depths[n]

        #X = np.delete(X, Z_invalid)
        #Y = np.delete(Y, Z_invalid)
        #Z = np.delete(Z, Z_invalid)

        points_3d = ut.calculate_3d_points(X, Y, Z, self.cam_params)
        #footprint_enu = points_3d[:, [0, 2]]
        footprint_ned = points_3d[:, [2, 0]]

        #angles = np.arctan2(footprint_enu[:, 1], footprint_enu[:, 0])
        #sorted_indices = np.argsort(angles)

        #footprint_enu_sorted = footprint_enu[sorted_indices]

        self.stixel_footprints = footprint_ned

    
    def get_horizontal_disp_edges(self, disparity_img, threshold=0.3):
    
        
        normalized_disparity = normalize_image(disparity_img)
        #normalized_disparity = cv2.normalize(disparity_img, None, 0, 1, cv2.NORM_MINMAX)

        blurred_image = cv2.GaussianBlur(normalized_disparity, (3, 3), 0)
        grad_y = cv2.Sobel(blurred_image, cv2.CV_32F, 0, 1, ksize=5)
        #grad_y = cv2.convertScaleAbs(grad_y)
        grad_y = (grad_y > threshold).astype(np.uint8)
        #_, grad_y = cv2.threshold(grad_y, threshold, 1, cv2.THRESH_BINARY)

        return grad_y
    
    def create_segmentation_score_map(self, disparity_img, depth_img, free_space_boundary, upper_contours):
        H, W = disparity_img.shape
        self.prev_stixel_stereo_depths = self.stixel_stereo_depths.copy()

        #start_time = time.time()
        grad_y = self.get_horizontal_disp_edges(disparity_img)

        #end_time = time.time()
        #runtime_ms = (end_time - start_time) * 1000
        #print(f"Disp edges: {runtime_ms:.2f} ms")

        grad_y = ut.filter_mask_by_boundary(grad_y, free_space_boundary, offset=10)
        grad_y = ut.get_bottommost_line(grad_y, thickness=5)
        upper_contours = ut.filter_mask_by_boundary(upper_contours, free_space_boundary, offset=20)
        upper_contours = ut.get_bottommost_line(upper_contours)

        #cv2.imshow("grad_y", grad_y.astype(np.uint8)*255)
        #cv2.imshow("upper contours", upper_contours*255)
        
        v_f_array = get_v_f_array(free_space_boundary, self.stixel_width, self.num_stixels, H)
        SSM, free_space_boundary_depth = create_SSM_numba(
            disparity_img,
            depth_img,
            grad_y,
            upper_contours,
            v_f_array,
            self.num_stixels,
            self.stixel_width
        )

        self.stixel_stereo_depths = free_space_boundary_depth.copy()


        # 4. Normalize, resize, show
        #SSM_normalized = cv2.normalize(SSM, None, 0, 255, cv2.NORM_MINMAX)
        #H, W = disparity_img.shape
        #SSM_resized = cv2.resize(SSM_normalized, (W, H), interpolation=cv2.INTER_NEAREST)
        #cv2.imshow("SSM", SSM_resized.astype(np.uint8))

        return SSM, free_space_boundary_depth
    

    def get_filterd_lidar_points(self, xyz_proj, xyz_c):
    
        filtered_image_points = []
        filtered_3d_points = []

        for n, stixel in enumerate(self.height_base_list):
            stixel_top = stixel[0]
            stixel_base = stixel[1]
            left_bound = n * self.stixel_width
            right_bound = (n + 1) * self.stixel_width

            mask = (
                (xyz_proj[:, 1] >= stixel_top) &
                (xyz_proj[:, 1] <= stixel_base) &
                (xyz_proj[:, 0] >= left_bound) &
                (xyz_proj[:, 0] <= right_bound)
            )

            stixel_points_2d = xyz_proj[mask]   
            stixel_points_3d = xyz_c[mask]

            filtered_image_points.extend(stixel_points_2d)
            filtered_3d_points.extend(stixel_points_3d)

        filtered_image_points = np.array(filtered_image_points)
        filtered_3d_points = np.array(filtered_3d_points)


        return filtered_image_points, filtered_3d_points


    def create_free_space_plygon(self, points):

        if len(points) < 2:
            print("Cannot create a polygon with less than 2 points.")
            return Polygon()
        #sorted_indices = points[:, 0].argsort()
        #points = points[sorted_indices]
        origin = np.array([0, 0])
        polygon_points = np.vstack([origin, points])

        return Polygon(polygon_points)

    def overlay_stixels_on_image(self, image, min_depth=0, max_depth=60):

        overlay = np.zeros_like(image)

        cmap = plt.get_cmap('gist_earth')

        for n, (stixel_top, stixel_base) in enumerate(self.height_base_list):

            if stixel_base > stixel_top and self.stixel_width > 0:

                stixel_depth = self.stixel_fused_depths[n]
                norm_depth = (stixel_depth - min_depth) / (max_depth - min_depth)
                norm_depth = np.clip(norm_depth, 0, 1)
                rgba = cmap(norm_depth)
                #rgba = cmap(1.0 - norm_depth)
                color = (int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255))

                colored_stixel = np.full((stixel_base - stixel_top, self.stixel_width, 3), color, dtype=np.uint8)
                #green_stixel = np.full((stixel_base - stixel_top, self.stixel_width, 3), (0, 80, 0), dtype=np.uint8) #(0, 50, 0)

                overlay[stixel_top:stixel_base, n * self.stixel_width:(n + 1) * self.stixel_width] = colored_stixel

                cv2.rectangle(overlay, 
                        (n * self.stixel_width, stixel_top),  # Top-left corner
                        ((n + 1) * self.stixel_width, stixel_base),  # Bottom-right corner
                        color,  # Color of the border (BGR)
                        2)  # Thickness of the border

        alpha = 1 # 0.8  # Weight of the original image
        beta = 0.8 #1  # Weight of the overlay
        gamma = 0.0  # Scalar added to each sum

        blended_image = cv2.addWeighted(image, alpha, overlay, beta, gamma)
        return blended_image
    

    def plot_stixel_footprints(self, footprints):

        plt.figure(figsize=(8,8))

        origin = np.array([0, 0])

        if len(footprints) == 0:
            return 
        
        valid_footprints = footprints[self.stixel_has_measurement]

        footprints = np.vstack([origin, valid_footprints])

        zs, xs = zip(*footprints)
        plt.fill(xs, zs, color='cyan', alpha=0.3, label="Free Space")

        first = True
        for (z, x) in footprints:
            plt.scatter(x, z, color='blue', marker='o', s=50, label='Stixel Footprints' if first else "")
            first = False

        plt.xlabel("X [m]")
        plt.ylabel("Z [m]")

        plt.legend()
        plt.show(block=False)
        plt.pause(1)  # Display the plot for a short period
        plt.close()

    def plot_projection_rays_and_associated_points(self, association_list):
        points = self.prev_stixel_footprints_curr_frame.copy()

        plt.figure(figsize=(8,8))

        colors = ut.get_high_contrast_colors(self.num_stixels, cmap_name='jet')

        p1 = np.array([0, 0])
        first = True
        for n, ray in enumerate(self.projection_rays):
            a, b, c, = ray
            z = self.stixel_fused_depths[n]
            if np.isnan(z):
                z = self.max_range

            p2 = np.array([-a*z, z])


            associated_ray = association_list[n]
            if associated_ray == -1:
                color = "black"
            else:
                color = colors[associated_ray]

            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=1, label=f"Rays" if first else "")
            first = False

        first = True
        for n, (z, x) in enumerate(points):
            plt.scatter(x, z, color=colors[n], marker='o', s=20, label='Stixel Footprints' if first else "")
            first = False

        
        plt.legend()
        plt.xlabel("X [m]")
        plt.ylabel("Z [m]")
        plt.show()

    def plot_projection_rays_and_points_world(self, rays, points, pose, points_validity, association_list):

        plt.figure(figsize=(8,8))

        colors = ut.get_high_contrast_colors(self.num_stixels, cmap_name='jet')

        origin = np.array([0, 0])
        p1 = self.transform_points_into_world(origin, pose)

        first = True
        for n, ray in enumerate(rays):
            a, b, c, = ray
            direction = np.array([b, -a])
            direction = direction / np.linalg.norm(direction)

            z = self.stixel_lidar_depths[n]
            if np.isnan(z):
                z = self.max_range

            p_cam = self.stixel_footprints[n]
            length = np.linalg.norm(p_cam)

            p2 = p1 - direction * length

            associated_ray = association_list[n]
            if associated_ray == -1:
                color = "black"
            else:
                color = colors[associated_ray]

            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=1, label=f"Rays" if first else "")
            first = False

        first = True
        for n, (x, y) in enumerate(points):

            plt.scatter(x, y, color=colors[n], marker='o', s=20, label='Stixel Footprints' if first else "")
            first = False

        ax = plt.gca()
        ax.invert_yaxis()
        ax.invert_xaxis()
        plt.legend()
        plt.xlabel("North [m]")
        plt.ylabel("East [m]")
        plt.show()



    def plot_prev_and_curr_stixel_footprints(self, prev_stixels, curr_stixels):
        if prev_stixels.size == 0:
            return 
        
        plt.figure(figsize=(8,8))
        
        first = True
        for (x, y) in prev_stixels:
            plt.scatter(x, y, color='red', marker='o', s=50, label='Prev Stixels' if first else "")
            first = False
        
        first = True
        for (x, y) in curr_stixels:
            plt.scatter(x, y, color='blue', marker='o', s=50, label='Curr Stixels' if first else "")
            first = False

        plt.xlabel("X [m]")
        plt.ylabel("Z [m]")

        plt.legend()
        plt.show()
        
        


@njit(parallel=True)
def get_free_space_boundary(water_mask):
    H, W = water_mask.shape
    search_height = H - 50
    free_space_boundary = np.full(W, H, dtype=np.int32)

    # For each column, search from the bottom of the search region upward 
    # to find the first zero in reversed_mask.
    for col in prange(W):
        for i in range(search_height):
            # Because reversed_mask == 0 is the same as water_mask[search_height-1-i, col] == 0
            if water_mask[search_height - 1 - i, col] == 0:
                free_space_boundary[col] = (search_height - 1 - i)
                break

    if free_space_boundary[11] < H:
        free_space_boundary[:10] = free_space_boundary[11]

    return free_space_boundary

@njit
def get_v_f_array(free_space_boundary, stixel_width, num_stixels, H):
    v_f_array = np.empty(num_stixels, dtype=np.int32)
    for n in range(num_stixels):
        start = n * stixel_width
        end = (n + 1) * stixel_width
        stixel_vals = free_space_boundary[start:end]
        stixel_vals.sort()  # inplace sort
        mid = len(stixel_vals) // 2
        if len(stixel_vals) % 2 == 0:
            v_f = (stixel_vals[mid - 1] + stixel_vals[mid]) // 2
        else:
            v_f = stixel_vals[mid]
        v_f_array[n] = min(max(v_f, 0), H - 1)
    return v_f_array


@njit(parallel=True)
def create_SSM_numba(disparity_img, depth_img,
                               grad_y, upper_contours,
                               v_f_array,
                               num_stixels, stixel_width):

    H, W = disparity_img.shape
    SSM = np.zeros((H, num_stixels), dtype=np.float32)
    #free_space_boundary_depth = np.zeros((H, num_stixels), dtype=np.float32)
    free_space_boundary_depth = np.zeros(num_stixels, dtype=np.float32)

    for n in prange(num_stixels):  # Parallel loop
        stixel_start = n * stixel_width
        stixel_end   = (n + 1) * stixel_width
        stixel_range = slice(stixel_start, stixel_end)

        v_f = v_f_array[n]
        if v_f >= H:
            v_f = H - 1
        elif v_f < 0:
            v_f = 0

        # Depth around free space boundary
        v_start = max(0, v_f - 20)
        depth_window = depth_img[v_start:v_f+1, stixel_range]
        median_val = nanmedian_2d(depth_window)
        free_space_boundary_depth[n] = median_val

        # Binary means
        gy_stixel    = grad_y[:, stixel_range]
        uc_stixel    = upper_contours[:, stixel_range]
        grad_y_means = (rowwise_mean(gy_stixel) > 0.5).astype(np.uint8)
        uc_means     = (rowwise_mean(uc_stixel) > 0.5).astype(np.uint8)

        # Reverse median base row for partial cumsum
        v_f_plus_1 = v_f + 1
        stixel_disparity = disparity_img[:, stixel_range]
        rev_row_medians = nanmedian_rowwise_reversed(stixel_disparity, v_f_plus_1)

        cumsum = np.cumsum(rev_row_medians)
        cumsum_sq = np.cumsum(rev_row_medians**2)
        counts = np.arange(1, v_f_plus_1 + 1)

        means = cumsum / counts
        variances = (cumsum_sq / counts) - (means**2)
        stds = np.sqrt(np.maximum(variances, 0.0))

        zero_mask = rev_row_medians == 0.0
        grad_part, contour_part, fg_scores = compute_scores(zero_mask, grad_y_means, uc_means, stds, v_f_plus_1)

        w1, w2, w3 = 100.0, 100.0, 150.0 #100.0, 100.0, 200.0
        scores = w1 * grad_part + w2 * fg_scores + w3 * contour_part

        # Flip back
        SSM[:v_f_plus_1, n] = scores[::-1]

    return SSM, free_space_boundary_depth

@njit
def compute_scores(zero_mask, grad_y_means, uc_means, stds, v_f_plus_1):
    grad_part = np.empty(v_f_plus_1, dtype=np.float32)
    contour_part = np.empty(v_f_plus_1, dtype=np.float32)
    fg_scores = np.empty(v_f_plus_1, dtype=np.float32)

    for i in range(v_f_plus_1):
        j = v_f_plus_1 - 1 - i  # reversed index

        if zero_mask[i]:
            grad_part[i] = 0.0
            contour_part[i] = 0.0
            fg_scores[i] = -1.0
        else:
            grad_part[i] = grad_y_means[j]
            contour_part[i] = uc_means[j]
            fg_scores[i] = 2.0**(1.0 - 2.0 * (stds[i] ** 2)) - 1.0

    return grad_part, contour_part, fg_scores

@njit
def compute_scores_new(zero_mask, grad_y_means, uc_means, stds, v_f_plus_1):
    grad_part = np.empty(v_f_plus_1, dtype=np.float32)
    contour_part = np.empty(v_f_plus_1, dtype=np.float32)
    fg_scores = np.empty(v_f_plus_1, dtype=np.float32)

    for i in range(v_f_plus_1):
        j = v_f_plus_1 - 1 - i  # reversed index

        if zero_mask[i]:
            fg_scores[i] = -1.0
        else:
            fg_scores[i] = 2.0**(1.0 - 2.0 * (stds[i] ** 2)) - 1.0
        
        grad_part[i] = grad_y_means[j]
        contour_part[i] = uc_means[j]


    return grad_part, contour_part, fg_scores


@njit
def nanmedian_rowwise(arr):
    H, W = arr.shape
    out = np.empty(H, dtype=np.float32)
    for i in range(H):
        row = arr[i]
        # Count valid (non-nan) entries
        valid = row[~np.isnan(row)]
        if valid.size == 0:
            out[i] = np.nan  # or 0.0 if you prefer a fallback
        else:
            sorted_row = np.sort(valid)
            mid = len(sorted_row) // 2
            if len(sorted_row) % 2 == 0:
                out[i] = 0.5 * (sorted_row[mid - 1] + sorted_row[mid])
            else:
                out[i] = sorted_row[mid]
    return out

@njit
def nanmedian_rowwise_reversed(arr, v_f_plus_1):
    W = arr.shape[1]
    out = np.empty(v_f_plus_1, dtype=np.float32)
    for i in range(v_f_plus_1):
        row = arr[i]
        valid = row[~np.isnan(row)]
        if valid.size == 0:
            out[v_f_plus_1 - 1 - i] = np.nan
        else:
            sorted_row = np.sort(valid)
            mid = len(sorted_row) // 2
            if len(sorted_row) % 2 == 0:
                out[v_f_plus_1 - 1 - i] = 0.5 * (sorted_row[mid - 1] + sorted_row[mid])
            else:
                out[v_f_plus_1 - 1 - i] = sorted_row[mid]
    return out

@njit
def nanmedian_2d(arr):
    flat = arr.ravel()
    valid = flat[~np.isnan(flat)]
    if valid.size == 0:
        return np.nan
    valid.sort()
    mid = valid.size // 2
    if valid.size % 2 == 0:
        return 0.5 * (valid[mid - 1] + valid[mid])
    else:
        return valid[mid]

@njit
def rowwise_mean(arr):
    H, W = arr.shape
    out = np.empty(H, dtype=np.float32)
    for i in range(H):
        s = 0.0
        for j in range(W):
            s += arr[i, j]
        out[i] = s / W
    return out


@njit
def assign_points_to_stixels_numba(xyz_proj, xyz_c,
                                stixel_tops, stixel_bases,
                                num_stixels, stixel_width):
    n_points = xyz_proj.shape[0]
    count = np.zeros(num_stixels, dtype=np.int32)

    for i in range(n_points):
        px = xyz_proj[i, 0]
        py = xyz_proj[i, 1]
        stixel_idx = int(px // stixel_width)
        if 0 <= stixel_idx < num_stixels:
            top_height = stixel_tops[stixel_idx]
            base_height = stixel_bases[stixel_idx]
            if top_height <= py <= base_height:
                count[stixel_idx] += 1

    # Prepare array to hold depths
    max_points = np.max(count)
    stixel_depths = np.full((max_points, num_stixels), np.nan, dtype=np.float32)

    # We'll track the offset for each stixel to know where to place the next point
    offset = np.zeros(num_stixels, dtype=np.int32)

    # Second pass: store the points
    for i in range(n_points):
        px = xyz_proj[i, 0]
        py = xyz_proj[i, 1]
        stixel_idx = int(px // stixel_width)
        if 0 <= stixel_idx < num_stixels:
            top_height = stixel_tops[stixel_idx]
            base_height = stixel_bases[stixel_idx]
            if top_height <= py <= base_height:
                idx = offset[stixel_idx]
                offset[stixel_idx] += 1
                stixel_depths[idx, stixel_idx] = xyz_c[i, 2]

    return stixel_depths, count

@njit
def get_percentile_all(stixel_depths, num_stixels, count, percent=0.3):

    out = np.full(num_stixels, np.nan, dtype=np.float32)

    for j in range(num_stixels):
        c = count[j]
        if c == 0:
            continue
        # Extract just the rows that have data for stixel j
        # stixel_depths[:c, j] is the subset, but let's copy to sort in nopython
        subset = stixel_depths[:c, j].copy()
        subset.sort()  # in-place sort
        # index at ~30% of length (0-based)
        idx = int(percent * (c - 1))
        out[j] = subset[idx]

    return out


@njit
def distance_transform_1d_numba(DP_col, penalty):
    n = DP_col.size
    dt = DP_col.copy()  # in-place modifications
    argmin = np.arange(n, dtype=np.int32)

    # Forward pass
    for i in range(1, n):
        alt = dt[i-1] + penalty
        if alt < dt[i]:
            dt[i] = alt
            argmin[i] = argmin[i-1]
    # Backward pass
    for i in range(n-2, -1, -1):
        alt = dt[i+1] + penalty
        if alt < dt[i]:
            dt[i] = alt
            argmin[i] = argmin[i+1]

    return dt, argmin

@njit
def get_optimal_height_numba(SSM, depth_map, free_space_boundary, num_stixels, NZ=5, Cs=3):
    cost_map = - SSM
    H, _ = cost_map.shape
    DP = np.full((H, num_stixels), np.inf, dtype=np.float32)
    parent = np.full((H, num_stixels), -1, dtype=np.int32)

    # Initialize
    DP[:, 0] = cost_map[:, 0]

    # DP loop
    for u in range(num_stixels - 1):
        v_f  = int(free_space_boundary[u])
        v_f1 = int(free_space_boundary[u+1])

        z_u  = depth_map[u]
        z_u1 = depth_map[u+1]
        relax_factor = max(0, 1 - abs(z_u - z_u1) / NZ)
        penalty = Cs * relax_factor

        dt, argmin = distance_transform_1d_numba(DP[:, u], penalty)
        DP[:, u+1] = dt + cost_map[:, u+1]
        parent[:, u+1] = argmin

    # best end
    best_end_v = np.argmin(DP[:, num_stixels-1])
    boundary = np.zeros(num_stixels, dtype=np.int32)
    boundary[-1] = best_end_v
    for u in range(num_stixels - 1, 0, -1):
        boundary[u-1] = parent[boundary[u], u]
    return boundary


@njit(parallel=True, fastmath=True)
def normalize_image(img, out_min=0.0, out_max=1.0):
    H, W = img.shape
    out = np.empty((H, W), dtype=np.float32)
    min_val = np.min(img)
    max_val = np.max(img)
    scale = (out_max - out_min) / (max_val - min_val + 1e-5)

    for i in prange(H):
        for j in range(W):
            val = (img[i, j] - min_val) * scale + out_min
            val = min(max(val, out_min), out_max)
            out[i, j] = val

    return out


@njit(parallel=True)
def compute_dynamic_stixels(v_top_array, v_f_array, boat_mask, stixel_width, num_stixels):
    # Use bool: True means dynamic, False means static.
    dynamic = np.empty(num_stixels, dtype=np.bool_)
    mask_height = boat_mask.shape[0]
    mask_width  = boat_mask.shape[1]
    
    for n in prange(num_stixels):  # parallelized outer loop
        v_top = v_top_array[n]
        v_f = v_f_array[n]
        
        # If the region is invalid, mark as static.
        if v_f <= v_top or v_top < 0 or v_f > mask_height:
            dynamic[n] = False
            continue
        
        u_start = n * stixel_width
        u_end = (n + 1) * stixel_width
        if u_end > mask_width:
            u_end = mask_width
        
        boat_pixels = 0
        total_pixels = 0
        
        # Loop over the ROI of the stixel.
        for i in range(v_top, v_f):
            for j in range(u_start, u_end):
                total_pixels += 1
                if boat_mask[i, j] > 0:
                    boat_pixels += 1
        
        # Mark stixel as dynamic if boat pixels exceed half the ROI.
        dynamic[n] = (total_pixels > 0 and boat_pixels > total_pixels / 2)
        
    return dynamic