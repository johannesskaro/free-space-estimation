import json

import numpy as np
from numba import njit, prange
from scipy.spatial.transform import Rotation as R


def make_homog(x, y, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        [   c, s,   x],
        [   -s, c,   y],
        [   0,  0,   1],
    ])


def get_delta_heading(pose_prev, pose_curr):
    ori_prev = pose_prev[3:]
    ori_curr = pose_curr[3:]

    # + np.pi because MA2 drives "backwards" relative to the camera
    heading_prev = R.from_quat(ori_prev).as_euler('xyz', degrees=False)[2] + np.pi
    heading_curr = R.from_quat(ori_curr).as_euler('xyz', degrees=False)[2] + np.pi

    delta_heading = heading_curr - heading_prev
    return delta_heading


def calculate_iou(mask1, mask2):
    mask1 = (mask1 > 0)
    mask2 = (mask2 > 0)

    intersection = np.count_nonzero(mask1 & mask2)
    union = np.count_nonzero(mask1 | mask2)

    if union == 0:
        return 0.0  # Avoid division by zero

    return intersection / union


def angular_error(point, line, min_r=1e-3):
    """Angle between the ray from the origin to `point` and the line (a, b, c)."""
    a, b, c = line
    z, x = point

    p = np.array([z, x], dtype=float)
    r = np.linalg.norm(p)
    if r < min_r:
        return np.inf

    # Any vector perpendicular to the normal (a, b) is along the line
    v = np.array([-b, a], dtype=float)

    p_hat = p / r
    v_hat = v / np.linalg.norm(v)

    # Acute angle to the infinite line (ignore ray direction)
    cos_theta = np.dot(p_hat, v_hat)
    cos_theta = np.abs(cos_theta)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    theta = np.arccos(cos_theta)
    return theta


def calculate_3d_points(X, Y, d, cam_params):
    cx, cy = cam_params["cx"], cam_params["cy"]
    fx, fy = cam_params["fx"], cam_params["fy"]

    X_o = d * (X - cx) / fx
    Y_o = d * (Y - cy) / fy
    Z_o = d

    return np.array([X_o, Y_o, Z_o]).T


def get_water_mask_from_contour_mask(contour_mask, offset=30):
    """Fallback water mask: everything below the lowest contour in each column."""
    height, _ = contour_mask.shape

    if offset > 0:
        search_region = contour_mask[:-offset, :]
    else:
        search_region = contour_mask

    reversed_mask = (search_region[::-1, :] > 0)

    first_positive = np.argmax(reversed_mask, axis=0)
    has_positive = np.any(reversed_mask, axis=0)

    bottom_indices = np.where(has_positive, height - 1 - first_positive, -1) - offset

    rows = np.arange(height).reshape(-1, 1)  # shape: (H, 1)
    boundary = bottom_indices.reshape(1, -1)    # shape: (1, W)

    water_mask = (rows >= boundary).astype(np.uint8)

    return water_mask


@njit(parallel=True)
def get_bottommost_line(mask, thickness=5):
    height, width = mask.shape
    output = np.zeros_like(mask, dtype=np.uint8)

    for x in prange(width):
        # Search from bottom to top for the first non-zero pixel
        for y in range(height - 1, -1, -1):
            if mask[y, x] > 0:
                y_bottom = y
                y_start = max(0, y_bottom - thickness + 1)
                for yy in range(y_start, y_bottom + 1):
                    output[yy, x] = 1
                break  # Found bottom, done with this column

    return output


@njit(parallel=True)
def filter_mask_by_boundary(mask, boundary_indices, offset=30):
    height, width = mask.shape
    output = np.zeros_like(mask)
    for x in prange(width):  # Parallel across columns
        h = max(0, boundary_indices[x] - offset)
        for y in range(h):  # Each column independently
            output[y, x] = mask[y, x]
    return output


def find_closest_timestamp(timestamps, target_timestamp):
    idx = np.abs(timestamps - target_timestamp).argmin()
    return idx, timestamps[idx]


def filter_point_cloud_by_image(xyz_proj, xyz_c, height, width):
    x = xyz_proj[:, 0]
    y = xyz_proj[:, 1]

    valid_mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)

    xyz_proj = xyz_proj[valid_mask]
    xyz_c    = xyz_c[valid_mask]

    return xyz_proj, xyz_c


def write_coordinates_to_file(filename, frame, coordinates, validity=None, dynamic=None, depth_uncertainty=None, curr_pose=None):
    """Append one frame of stixel footprints as a JSON line."""
    coordinates_list = [list(coord) for coord in coordinates]

    validity_list = [int(v) for v in validity] if validity is not None else None
    dynamic_list = [int(d) for d in dynamic] if dynamic is not None else None
    depth_uncertainty_list = [float(d) for d in depth_uncertainty] if depth_uncertainty is not None else None

    data = {
        "frame": frame,
        "points": coordinates_list,
        "validity": validity_list,
        "dynamic": dynamic_list,
        "depth_uncertainty": depth_uncertainty_list,
        "pose": curr_pose.tolist() if curr_pose is not None else None
    }

    with open(filename, 'a') as file:
        json.dump(data, file)
        file.write("\n")
