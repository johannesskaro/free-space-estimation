import numpy as np
import cupy as cp
from shapely.geometry import Polygon
from collections import deque
import utilities as ut
import cv2
from scipy.interpolate import griddata, Rbf
from scipy import stats
from fastSAM import FastSAMSeg
from numba import njit, prange
import matplotlib.pyplot as plt
import time




class Stixels:
        
    def __init__(self, num_stixels, img_shape, cam_params, min_stixel_height=20):
        self.num_stixels = num_stixels
        self.img_shape = img_shape
        self.cam_params = cam_params
        self.stixel_width = self.stixel_width = int(img_shape[1] // self.num_stixels)
        self.min_stixel_height = min_stixel_height
        self.stixel_list = np.zeros((self.num_stixels, 2), dtype=int)
        self.stixel_lidar_depths = np.zeros(self.num_stixels)


    def create_stixels(self, water_mask, disparity_img, depth_img, upper_contours, xyz_proj, xyz_c):

        self.shape = disparity_img.shape

        free_space_boundary = get_free_space_boundary(water_mask)

        start_time = time.time()
        SSM, free_space_boundary_depth = self.create_segmentation_score_map(disparity_img, depth_img, free_space_boundary, upper_contours)
        end_time = time.time()
        runtime_ms = (end_time - start_time) * 1000
        print(f"Processing time: {runtime_ms:.2f} ms")

        
        top_boundary, top_boundary_mask = self.get_optimal_height(SSM, free_space_boundary_depth, free_space_boundary)
        self.get_stixel_img_pos_list(free_space_boundary, top_boundary)
        filtered_lidar_points = self.get_stixel_depths_from_lidar(xyz_proj, xyz_c)


        
        stixel_footprints = self.get_stixel_BEV_footprints()



        return stixel_footprints, filtered_lidar_points

    

    def get_stixel_img_pos_list(self, free_space_boundary, top_boundary):

        fsb_2d = free_space_boundary.reshape(self.num_stixels, self.stixel_width)
        v_f_array = np.median(fsb_2d, axis=1).astype(int) 
        
        v_top_array = top_boundary.astype(int).copy()
        
        dist = v_f_array - v_top_array
        mask = dist < self.min_stixel_height
        v_top_array[mask] = v_f_array[mask] - self.min_stixel_height

        stixel_list = np.column_stack((v_top_array, v_f_array))

        self.stixel_list[:] = stixel_list

        return self.stixel_list
    
    def get_stixel_depths_from_lidar(self, xyz_proj, xyz_c):

        stixel_xyz_proj_list = []

        for n, (top_height, base_height) in enumerate(self.stixel_list):
            left_bound = n * self.stixel_width
            right_bound = (n + 1) * self.stixel_width

            mask = (
                (xyz_proj[:, 1] >= top_height) &
                (xyz_proj[:, 1] <= base_height) &
                (xyz_proj[:, 0] >= left_bound) &
                (xyz_proj[:, 0] <= right_bound)
            )

            stixel_xyc_c = xyz_c[mask]
            stixel_xyz_proj = xyz_proj[mask]

            stixel_xyz_proj_list.extend(stixel_xyz_proj)

            if len(stixel_xyc_c) > 0:
                distances = stixel_xyc_c[:, 2]
                lidar_depth = np.percentile(distances, 30)

            else:
                lidar_depth = np.nan

            self.stixel_lidar_depths[n] = lidar_depth

        return stixel_xyz_proj_list


    def get_stixel_BEV_footprints(self):

        X = np.full(self.num_stixels, np.nan)
        Y = np.full(self.num_stixels, np.nan)
        Z = np.full(self.num_stixels, np.nan)
        Z_invalid = np.array([], dtype=int)

        for n, stixel in enumerate(self.stixel_list):

            if np.isnan(self.stixel_lidar_depths[n]):
                Z_invalid = np.append(Z_invalid, n)
            else:
                X[n] = n * self.stixel_width + self.stixel_width // 2
                Y[n] = stixel[1]
                Z[n] = self.stixel_lidar_depths[n]

        X = np.delete(X, Z_invalid)
        Y = np.delete(Y, Z_invalid)
        Z = np.delete(Z, Z_invalid)

        points_3d = ut.calculate_3d_points(X, Y, Z, self.cam_params)
        footprint_enu = points_3d[:, [0, 2]]

        angles = np.arctan2(footprint_enu[:, 1], footprint_enu[:, 0])
        sorted_indices = np.argsort(angles)

        footprint_enu_sorted = footprint_enu[sorted_indices]

        return footprint_enu_sorted
    
    def get_horizontal_disp_edges(self, disparity_img, free_space_boundary):
        

        normalized_disparity = cv2.normalize(disparity_img, None, 0, 1, cv2.NORM_MINMAX)
        blurred_image = cv2.GaussianBlur(normalized_disparity, (5, 5), 0)
        grad_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=5)
        grad_y = cv2.convertScaleAbs(grad_y)
        _, grad_y = cv2.threshold(grad_y, 0.3, 1, cv2.THRESH_BINARY)
        grad_y = ut.filter_mask_by_boundary(grad_y, free_space_boundary, offset=10)
        grad_y = ut.get_bottommost_line(grad_y, thickness=10)
        #cv2.imshow("grad_y", grad_y.astype(np.uint8)*255)

        return grad_y
    
    def create_segmentation_score_map(self, disparity_img, depth_img, free_space_boundary, upper_contours):
        H, W = disparity_img.shape

        start_time = time.time()
        grad_y = self.get_horizontal_disp_edges(disparity_img, free_space_boundary)
        end_time = time.time()
        runtime_ms = (end_time - start_time) * 1000
        #print(f"Processing time: {runtime_ms:.2f} ms")

        upper_contours = ut.filter_mask_by_boundary(upper_contours, free_space_boundary, offset=15)

        
        upper_contours = ut.get_bottommost_line(upper_contours)
 
        #cv2.imshow("upper contours", upper_contours*255)


        disparity_img_f32 = np.ascontiguousarray(disparity_img.astype(np.float32))
        depth_img_f32     = np.ascontiguousarray(depth_img.astype(np.float32))
        grad_y_f32        = np.ascontiguousarray(grad_y.astype(np.float32))
        upper_contours_f32 = np.ascontiguousarray(upper_contours.astype(np.float32))
        free_space_boundary_f32 = np.ascontiguousarray(free_space_boundary.astype(np.float32))

        # 3. Call the numba function
        SSM, free_space_boundary_depth = create_seg_score_map_numba(
            disparity_img_f32,
            depth_img_f32,
            grad_y_f32,
            upper_contours_f32,
            free_space_boundary_f32, 
            self.num_stixels,
            self.stixel_width
        )

        # 4. Normalize, resize, show
        #SSM = cv2.normalize(SSM, None, 0, 255, cv2.NORM_MINMAX)
        #H, W = disparity_img.shape
        #SSM_resized = cv2.resize(SSM, (W, H), interpolation=cv2.INTER_NEAREST)
        #cv2.imshow("SSM", SSM_resized.astype(np.uint8))

        return SSM, free_space_boundary_depth

    
    @staticmethod
    def distance_transform_1d(f_gpu: cp.ndarray, penalty: float):
        n = f_gpu.size

        # Copy f into dt for in-place modification
        dt = cp.copy(f_gpu)

        # argmin[i] starts as i
        argmin = cp.arange(n, dtype=cp.int32)

        # Currently launching a single-threaded kernel,
        # because we do a naive forward/back pass in order.
        grid_size = 1
        block_size = 1
        Stixels.distance_transform_1d_kernel((grid_size,), (block_size,), 
            (f_gpu, dt, argmin, n, penalty))

        return dt, argmin


    def get_optimal_height(self, SSM, depth_map, free_space_boundary):
        _, W = self.img_shape

        cost_map = -SSM
        cost_map_gpu = cp.asarray(cost_map, dtype=cp.float64)   # (H, N)
        depth_map_gpu  = cp.asarray(depth_map,  dtype=cp.float64)   # (H, N)
        fsb_gpu        = cp.asarray(free_space_boundary, dtype=cp.float64)  # (N)
        H, _ = cost_map_gpu.shape
    
        # ---------- DP and parent pointers ----------
        DP_gpu     = cp.full((H, self.num_stixels), cp.inf, dtype=cp.float64)
        parent_gpu = cp.full((H, self.num_stixels), -1,    dtype=cp.int32)

        # Initialize DP in column 0
        DP_gpu[:, 0] = cost_map_gpu[:, 0]

        # ---------- Constants you had before ----------
        NZ = 5
        Cs = 2

        # ---------- Main DP loop over columns ----------
        for u in range(self.num_stixels - 1):
            # free-space boundary in current + next column
            v_f  = fsb_gpu[u]
            v_f1 = fsb_gpu[u + 1]

            # Depth difference for penalty
            z_u  = depth_map_gpu[cp.asarray(v_f, dtype=cp.int32),   u]
            z_u1 = depth_map_gpu[cp.asarray(v_f1, dtype=cp.int32), (u + 1)]

            relax_factor = cp.maximum(0, 1 - cp.abs(z_u - z_u1) / NZ)

            # Penalty for a 1-row jump
            penalty = Cs * relax_factor

            # Run GPU-based distance transform for DP[:, u]
            dt, argmin = self.distance_transform_1d(DP_gpu[:, u], float(penalty))

            # Update DP for column u+1
            DP_gpu[:, u + 1]     = dt + cost_map_gpu[:, u + 1]
            parent_gpu[:, u + 1] = argmin

        # ---------- Find best ending row in last col ----------
        last_col = DP_gpu[:, self.num_stixels - 1]
        best_end_v_gpu = cp.argmin(last_col)

        # ---------- Backtrack boundary ----------
        boundary_gpu = cp.empty((self.num_stixels), dtype=cp.int32)
        boundary_gpu[-1] = best_end_v_gpu

        for u in range(self.num_stixels - 1, 0, -1):
            boundary_gpu[u - 1] = parent_gpu[boundary_gpu[u], u]


        # Copy back to host
        boundary = boundary_gpu.get()

        boundary_mask = np.zeros((H, self.num_stixels), dtype=int)
        boundary_mask[boundary, np.arange(self.num_stixels)] = 1
        boundary_mask = cv2.resize(boundary_mask, (W, H), interpolation=cv2.INTER_NEAREST)

        return boundary, boundary_mask


    def create_free_space_plygon(self, points):

        if len(points) < 2:
            print("Cannot create a polygon with less than 2 points.")
            return Polygon()
        #sorted_indices = points[:, 0].argsort()
        #points = points[sorted_indices]
        origin = np.array([0, 0])
        polygon_points = np.vstack([origin, points])

        return Polygon(polygon_points)

    def overlay_stixels_on_image(self, image):

        overlay = np.zeros_like(image)

        for n, (stixel_top, stixel_base) in enumerate(self.stixel_list):

            if stixel_base > stixel_top and self.stixel_width > 0:

                green_stixel = np.full((stixel_base - stixel_top, self.stixel_width, 3), (0, 80, 0), dtype=np.uint8) #(0, 50, 0)
                overlay[stixel_top:stixel_base, n * self.stixel_width:(n + 1) * self.stixel_width] = green_stixel

                cv2.rectangle(overlay, 
                        (n * self.stixel_width, stixel_top),  # Top-left corner
                        ((n + 1) * self.stixel_width, stixel_base),  # Bottom-right corner
                        (0,255,0),  # Color of the border (BGR)
                        2)  # Thickness of the border

        alpha = 1 # 0.8  # Weight of the original image
        beta = 0.8 #1  # Weight of the overlay
        gamma = 0.0  # Scalar added to each sum

        blended_image = cv2.addWeighted(image, alpha, overlay, beta, gamma)
        return blended_image
    

    def plot_stixel_footprint(self, footprints):

        plt.figure(figsize=(8,8))

        origin = np.array([0, 0])
        footprints = np.vstack([origin, footprints])

        xs, ys = zip(*footprints)
        plt.fill(xs, ys, color='cyan', alpha=0.3, label="Free Space")

        first = True
        for (x, y) in footprints:
            plt.scatter(x, y, color='blue', marker='o', s=50, label='Stixel Footprints' if first else "")
            first = False

        plt.xlabel("X [m]")
        plt.ylabel("Z [m]")

        plt.legend()
        plt.show(block=False)
        plt.pause(1)  # Display the plot for a short period
        plt.close()



    distance_transform_1d_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void distance_transform_1d(const double* __restrict__ f,
                            double* __restrict__ dt,
                            int* __restrict__ argmin,
                            const int n,
                            const double penalty)
    {
        // Forward pass
        for (int i = 1; i < n; i++) {
            double alt = dt[i - 1] + penalty;
            if (alt < dt[i]) {
                dt[i] = alt;
                argmin[i] = argmin[i - 1];
            }
        }
        // Backward pass
        for (int i = n - 2; i >= 0; i--) {
            double alt = dt[i + 1] + penalty;
            if (alt < dt[i]) {
                dt[i] = alt;
                argmin[i] = argmin[i + 1];
            }
        }
    }
    ''', 'distance_transform_1d')


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


@njit(parallel=True)
def create_seg_score_map_numba(disparity_img, depth_img,
                               grad_y, upper_contours,
                               free_space_boundary,
                               num_stixels, stixel_width):

    H, W = disparity_img.shape
    SSM = np.zeros((H, num_stixels), dtype=np.float32)
    free_space_boundary_depth = np.zeros((H, num_stixels), dtype=np.float32)

    for n in prange(num_stixels):  # Parallel loop
        stixel_start = n * stixel_width
        stixel_end   = (n + 1) * stixel_width
        stixel_range = slice(stixel_start, stixel_end)

        # median base row
        v_f = int(np.median(free_space_boundary[stixel_range]))
        if v_f >= H:
            v_f = H - 1
        elif v_f < 0:
            v_f = 0

        stixel_disparity = disparity_img[:, stixel_range]
        #row_medians = np.nanmedian(stixel_disparity, axis=1)
        row_medians = nanmedian_rowwise(stixel_disparity)

        # Depth around free space boundary
        v_start = max(0, v_f - 10)
        depth_window = depth_img[v_start:v_f+1, stixel_range]
        median_val = np.nanmedian(depth_window)
        free_space_boundary_depth[v_f, n] = median_val

        # Binary means
        gy_stixel    = grad_y[:, stixel_range]
        uc_stixel    = upper_contours[:, stixel_range]
        grad_y_means = (rowwise_mean(gy_stixel) > 0.5).astype(np.uint8)
        uc_means     = (rowwise_mean(uc_stixel) > 0.5).astype(np.uint8)

        # Reverse for partial cumsum
        v_f_plus_1 = v_f + 1
        rev_row_medians = row_medians[:v_f_plus_1][::-1]

        cumsum = np.cumsum(rev_row_medians)
        cumsum_sq = np.cumsum(rev_row_medians**2)
        counts = np.arange(1, v_f_plus_1 + 1)

        means = cumsum / counts
        variances = (cumsum_sq / counts) - (means**2)
        stds = np.sqrt(np.maximum(variances, 0.0))

        zero_mask = rev_row_medians == 0.0
        grad_part = np.where(zero_mask, 0, grad_y_means[:v_f_plus_1][::-1])
        contour_part = np.where(zero_mask, 0, uc_means[:v_f_plus_1][::-1])
        fg_scores = np.where(zero_mask, -1, 2.0**(1.0 - 2.0 * (stds**2)) - 1.0)

        w1, w2, w3 = 100.0, 100.0, 200.0
        scores = w1 * grad_part + w2 * fg_scores + w3 * contour_part

        # Flip back
        SSM[:v_f_plus_1, n] = scores[::-1]

    return SSM, free_space_boundary_depth


@njit
def nanmedian_rowwise(arr):
    """
    Compute nanmedian across axis=1 for a 2D array (Numba-compatible).
    Returns a 1D array with the median of each row, skipping NaNs.
    """
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
def rowwise_mean(arr):
    H, W = arr.shape
    out = np.empty(H, dtype=np.float32)
    for i in range(H):
        s = 0.0
        for j in range(W):
            s += arr[i, j]
        out[i] = s / W
    return out