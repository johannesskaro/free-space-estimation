import numpy as np
import cv2
from numba import njit, prange
from scipy.spatial.transform import Rotation as R

from free_space import utils as ut


class Stixels:
    """Stixel world: splits the image into `num_stixels` vertical columns and, for
    each, estimates where the obstacle starts (free-space boundary), how tall it is,
    and how far away it is. Depths from lidar, stereo and the previous frame are
    fused with a per-stixel recursive filter.

    Poses are [x, y, z, qx, qy, qz, qw] in NED. Footprints are returned as
    (forward, right) in the camera frame, one row per stixel.
    """

    def __init__(self, num_stixels, img_shape, cam_params, t_body_to_cam, R_body_to_cam, min_stixel_height=20, max_range=60, cam_fov=110):
        self.num_stixels = num_stixels
        self.img_shape = img_shape
        self.cam_params = cam_params
        self.stixel_width  = int(img_shape[1] // self.num_stixels)
        self.min_stixel_height = min_stixel_height
        self.max_range = max_range
        self.cam_fov = cam_fov
        self.ray_spacing = cam_fov / num_stixels

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
        self.prop_set = set()

        self.R_body_to_cam = np.array(R_body_to_cam)
        self.t_body_to_cam = np.array(t_body_to_cam)

        self.projection_rays = self.get_projection_rays()

    def run_stixel_pipeline(self, left_img, water_mask, water_mask_failure, disparity_img, depth_img, upper_contours, xyz_proj, xyz_c, pose_prev, pose_curr, dt, boat_mask=None):
        """Update the stixels with a new frame and return the (N, 2) footprints."""

        self.prev_stixel_footprints = self.stixel_footprints.copy()
        self.prev_stixel_lidar_depths = self.stixel_lidar_depths.copy()
        self.prev_stixel_fused_depths_var = self.stixel_fused_depths_var.copy()
        self.prev_stixel_has_measurement = self.stixel_has_measurement.copy()
        self.prev_stixel_validity = self.stixel_validity.copy()
        self.prev_using_prop_depth = self.using_prop_depth.copy()

        if water_mask_failure:
            print("Segmentation failure detected! Skipping stixel update.")
            self.stixel_validity = np.full(self.num_stixels, False, dtype=bool)
            return self.prev_stixel_footprints.copy()

        delta_heading = ut.get_delta_heading(pose_prev, pose_curr)

        self.create_stixels_in_image(water_mask, disparity_img, depth_img, upper_contours, boat_mask)

        self.get_stixel_depths_from_lidar(xyz_proj, xyz_c)

        self.transform_prev_stixels_into_curr_frame(pose_prev, pose_curr)

        self.associate_prev_stixels(delta_heading)

        self.handle_propagated_depths(delta_heading)

        self.recursive_height_filter()

        self.recursive_depth_filter(delta_heading)

        self.get_stixel_BEV_footprints()

        return self.stixel_footprints.copy()

    def create_stixels_in_image(self, water_mask, disparity_img, depth_img, upper_contours, boat_mask=None):
        free_space_boundary = get_free_space_boundary(water_mask)
        SSM, stereo_depth = self.create_segmentation_score_map(disparity_img, depth_img, free_space_boundary, upper_contours)

        top_boundary = get_optimal_height_numba(SSM, stereo_depth, free_space_boundary, self.num_stixels)
        self.get_stixel_height_base_list(free_space_boundary, top_boundary, boat_mask)

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

    def associate_prev_stixels(self, delta_heading,
                            ang_thres_deg: float = 0.5,
                            z_diff_thres: float = 1):

        N = self.num_stixels
        ang_thres = np.deg2rad(ang_thres_deg)
        max_off = int(np.ceil(ang_thres_deg / (self.cam_fov / N)))

        delta_idx = int(round(np.rad2deg(delta_heading) / self.ray_spacing))

        assoc_height = -np.ones(N, dtype=int)
        best_ray_idx = -np.ones(N, dtype=int)
        best_ray_err = np.full(N, np.inf)

        valid_prev = np.flatnonzero(self.prev_stixel_validity)
        pts = self.prev_stixel_footprints_curr_frame
        z_lidar_curr = self.stixel_lidar_depths
        prev_dyn = self.prev_dynamic_stixel_list
        curr_dyn = self.dynamic_stixel_list

        if valid_prev.size == 0 or pts.size == 0:
            self.association_height = assoc_height
            self.association_depth = -np.ones(N, dtype=int)
            return

        for n, ray in enumerate(self.projection_rays):

            idx_pred = np.clip(n + delta_idx, 0, N - 1)
            pos = np.searchsorted(valid_prev, idx_pred)
            lo = max(pos - max_off, 0)
            hi = min(pos + max_off + 1, valid_prev.size)
            cands = valid_prev[lo:hi]

            errs = np.array([ut.angular_error(pts[i], ray) for i in cands])
            good = errs < ang_thres
            if not good.any():
                continue

            j = np.argmin(np.where(good, errs, np.inf))
            idx = cands[j]
            err = errs[j]

            # depth‐association only if dynamic‐flag matches
            if prev_dyn[idx] == curr_dyn[n]:
                best_ray_idx[n] = idx
                best_ray_err[n] = err

            # height‐association whenever both prev and curr stixel is static
            if (not prev_dyn[idx]) or (not curr_dyn[n]):
                if abs(pts[idx, 0] - z_lidar_curr[n]) < z_diff_thres:
                    assoc_height[n] = idx

        # now resolve “one prev → many rays” by picking the ray with minimal error
        assoc_depth = -np.ones(N, dtype=int)
        rays_with_candidate = best_ray_idx >= 0
        for prev_idx in np.unique(best_ray_idx[rays_with_candidate]):
            rays = np.nonzero(best_ray_idx == prev_idx)[0]
            best_ray = rays[np.argmin(best_ray_err[rays])]
            assoc_depth[best_ray] = prev_idx

        self.association_height = assoc_height
        self.association_depth = assoc_depth

    def handle_propagated_depths(self, delta_heading):
        DELTA_IDX = int(round(delta_heading / self.ray_spacing))

        assoc = self.association_depth.copy()
        z_lidar_curr = self.stixel_lidar_depths.copy()
        z_lidar_prev = self.prev_stixel_lidar_depths.copy()

        has_lidar_curr = ~np.isnan(z_lidar_curr) & ~np.isinf(z_lidar_curr)
        has_lidar_prev = ~np.isnan(z_lidar_prev) & ~np.isinf(z_lidar_prev)  

        for n in range(self.num_stixels):
            idx = assoc[n]

            if idx == -1:
                continue

            if has_lidar_curr[n]:
                # if we now have a direct lidar read, drop any old‐prop entry
                self.prop_set.discard(idx)

            elif has_lidar_prev[idx]:
                # we lost the current lidar but had it before → start propagating
                self.prop_set.add(n)

            else:
                # neither old nor new has lidar → check if we should shift the propagation
                if idx in self.prop_set:
                    idx_pred = idx + DELTA_IDX
                    if 0 <= idx_pred < self.num_stixels and n == idx_pred:
                        self.prop_set.discard(idx)
                        self.prop_set.add(n)
                    else:
                        assoc[n] = -1

        self.association_depth = assoc

    def recursive_height_filter(self, alpha=0.7):
        for n in range(self.num_stixels):
            if self.association_height[n] != -1:
                idx = self.association_height[n]

                v_top_curr = self.height_base_list[n, 0]
                v_top_prev = self.prev_height_base_list[idx, 0]

                v_top_lp = alpha * v_top_prev + (1 - alpha) * v_top_curr

                self.height_base_list[n, 0] = v_top_lp

    def recursive_depth_filter(self, delta_heading):
        self.using_prop_depth = np.full(self.num_stixels, False, dtype=bool)
        self.stixel_validity = np.full(self.num_stixels, False, dtype=bool)

        C_pose = np.array([[0.01, 0, 0], 
                           [0, 0.01, 0],
                           [0, 0, 0.001225]])

        fx = self.cam_params["fx"]
        b = self.cam_params["b"]

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

            if np.isnan(z_stereo) or np.isinf(z_stereo) or z_stereo > 10:
                z_stereo = 0
                var_stereo = np.inf
                has_stereo = False
            else:
                sigma_px = 0.5 
                sigma_stereo = sigma_px * z_stereo**2 / (fx * b)
                var_stereo = sigma_stereo**2
                has_stereo = True

            # Prediction step

            if self.association_depth[n] != -1:
                idx = self.association_depth[n]

                var_fused_prev = self.prev_stixel_fused_depths_var[idx]

                p = self.prev_stixel_footprints[idx]
                J_z_motion = np.array([1, 0, -p[0]*np.sin(delta_heading) - p[1]*np.cos(delta_heading)])
                var_motion = J_z_motion @ C_pose @ J_z_motion.T 

                z_prop = self.prev_stixel_footprints_curr_frame[idx, 0]
                var_prop = var_fused_prev + var_motion
                has_prop = True

            else:
                z_prop = 0
                var_prop = np.inf
                has_prop = False

            # Update step

            var_fused = 1 / ((1 / var_lidar) + (1 / var_stereo) + (1 / var_prop) + 1e-10)
            z_fused = var_fused * ((z_lidar / var_lidar) + (z_stereo / var_stereo) + (z_prop / var_prop))

            if not has_lidar:
                self.stixel_has_measurement[n] = False

                if has_prop:
                    self.using_prop_depth[n] = True

            else:
                self.stixel_has_measurement[n] = True

            # If all depths are invalid

            if not (has_lidar or has_stereo or has_prop):
                self.stixel_validity[n] = False

            else:
                self.stixel_validity[n] = True

            self.stixel_fused_depths[n] = z_fused
            self.stixel_fused_depths_var[n] = var_fused

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

        # SciPy expects quaternion [x, y, z, w]. + np.pi because MA2 drives "backwards" relative to the camera
        yaw_prev = R.from_quat(prev_pose[3:]).as_euler('xyz', degrees=False)[2] + np.pi #- np.deg2rad(3)
        yaw_curr = R.from_quat(curr_pose[3:]).as_euler('xyz', degrees=False)[2] + np.pi #- np.deg2rad(3)

        T_prev = ut.make_homog(x_prev, y_prev, yaw_prev)
        T_curr = ut.make_homog(x_curr, y_curr, yaw_curr)

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
        else:
            self.dynamic_stixel_list = np.full(self.num_stixels, False, dtype=bool)

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

        for n, stixel in enumerate(self.height_base_list):

            X[n] = (n + 1) * self.stixel_width - self.stixel_width // 2
            Y[n] = stixel[1]
            Z[n] = self.stixel_fused_depths[n]

        points_3d = ut.calculate_3d_points(X, Y, Z, self.cam_params)
        footprint_ned = points_3d[:, [2, 0]]

        self.stixel_footprints = footprint_ned

    def get_horizontal_disp_edges(self, disparity_img, threshold=0.3):
        
        normalized_disparity = normalize_image(disparity_img)

        blurred_image = cv2.GaussianBlur(normalized_disparity, (3, 3), 0)
        grad_y = cv2.Sobel(blurred_image, cv2.CV_32F, 0, 1, ksize=5)
        grad_y = (grad_y > threshold).astype(np.uint8)

        return grad_y

    def create_segmentation_score_map(self, disparity_img, depth_img, free_space_boundary, upper_contours):
        H, W = disparity_img.shape
        self.prev_stixel_stereo_depths = self.stixel_stereo_depths.copy()

        grad_y = self.get_horizontal_disp_edges(disparity_img)
        grad_y = ut.filter_mask_by_boundary(grad_y, free_space_boundary, offset=10)
        grad_y = ut.get_bottommost_line(grad_y, thickness=5)

        upper_contours = ut.filter_mask_by_boundary(upper_contours, free_space_boundary, offset=20)
        upper_contours = ut.get_bottommost_line(upper_contours)

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

        return SSM, free_space_boundary_depth


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

        w1, w2, w3 = 100.0, 100.0, 200.0 #100.0, 100.0, 200.0
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