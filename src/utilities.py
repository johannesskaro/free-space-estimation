import cv2
import numpy as np
import scipy.io
from scipy.interpolate import interp1d
import json
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def blend_image_with_mask(img, mask, color=[0, 0, 255], alpha1=1, alpha2=1):
    # Convert binary mask to a colored mask (e.g., red)
    colored_mask = np.zeros_like(img)
    colored_mask[mask > 0] = color

    # Blend the original image and the colored mask
    blended = cv2.addWeighted(img, alpha1, colored_mask, alpha2, 0)
    return blended


def corresponding_pixels(mask1: np.array, mask2: np.array) -> int:

    if mask1.shape != mask2.shape:
        raise ValueError("mask1 and mask2 must have the same shape")

    corresponding_count = np.sum(mask1 == mask2)

    return corresponding_count


def rotate_point(x, y, image_width, image_height, roll_rad, initial_roll_rad=0):
    """
    Rotate a point (x, y) in the image by the roll angle.

    :param x: x-coordinate of the point
    :param y: y-coordinate of the point
    :param roll_angle_rad: Roll angle in radians
    :param image_width: Width of the image
    :param image_height: Height of the image
    :return: Rotated point (x', y')
    """

    # Translate point to origin-centered coordinates
    x -= image_width / 2
    y -= image_height / 2

    # Rotation matrix
    rotation_matrix = np.array(
        [
            [np.cos(roll_rad - initial_roll_rad), -np.sin(roll_rad - initial_roll_rad)],
            [np.sin(roll_rad - initial_roll_rad), np.cos(roll_rad - initial_roll_rad)],
        ]
    )

    # Rotated point
    x_rotated, y_rotated = rotation_matrix @ np.array([x, y])

    # Translate back to image coordinates
    x_rotated += image_width / 2
    y_rotated += image_height / 2

    return round(x_rotated), round(y_rotated)


def calculate_iou(mask1, mask2):
    """
    Calculate Intersection over Union (IoU) between two binary masks.
    """
    mask1 = (mask1 > 0)
    mask2 = (mask2 > 0)

    intersection = np.count_nonzero(mask1 & mask2)
    union = np.count_nonzero(mask1 | mask2)

    if union == 0:
        return 0.0  # Avoid division by zero

    return intersection / union


def calculate_3d_point(x_o, y_o, d, cam_params):
    cx, cy = cam_params["cx"], cam_params["cy"]
    fx, fy = cam_params["fx"], cam_params["fy"]
    X_o = d * (x_o - cx) / fx
    Y_o = d * (y_o - cy) / fy
    return float(X_o), float(Y_o), d


def calculate_3d_points(X, Y, d, cam_params):
    cx, cy = cam_params["cx"], cam_params["cy"]
    fx, fy = cam_params["fx"], cam_params["fy"]

    X_o = d * (X - cx) / fx
    Y_o = d * (Y - cy) / fy
    Z_o = d

    return np.array([X_o, Y_o, Z_o]).T


def calculate_3d_points_from_mask(mask, depth_map, cam_params):
    if mask.shape != depth_map.shape:
        raise ValueError("Mask and depth map must have the same dimensions")

    Y, X = np.where(np.logical_and(mask > 0, depth_map > 0))
    d = depth_map[Y, X]

    points_3d = calculate_3d_points(X, Y, d, cam_params)

    return points_3d


from scipy.spatial.transform import Rotation


def mods_2_intrinsics_extrinsics(M1, M2, D1, D2, R, T, size):
    intrinsics = {
        "stereo_left": {
            "fx": M1[0, 0],
            "fy": M1[1, 1],
            "cx": M1[0, 2],
            "cy": M1[1, 2],
            "distortion_coefficients": D1,
            "image_width": size[0],
            "image_height": size[1],
        },
        "stereo_right": {
            "fx": M2[0, 0],
            "fy": M2[1, 1],
            "cx": M2[0, 2],
            "cy": M2[1, 2],
            "distortion_coefficients": D2,
            "image_width": size[0],
            "image_height": size[1],
        },
    }

    extrinsics = {"rotation_matrix": R, "translation": T / 1000}
    return intrinsics, extrinsics


def get_fov(fx, fy, W, H, type="rad"):
    fov_x = 2 * np.arctan(W / (2 * fx))
    fov_y = 2 * np.arctan(H / (2 * fy))
    if type == "deg":
        fov_x = np.rad2deg(fov_x)
        fov_y = np.rad2deg(fov_y)
    return fov_x, fov_y


def invert_transformation(H):
    R = H[:3, :3]
    T = H[:3, 3]
    H_transformed = np.block(
        [
            [R.T, -R.T.dot(T)[:, np.newaxis]],
            [np.zeros((1, 3)), np.ones((1, 1))],
        ]
    )
    return H_transformed


def pohang_2_intrinsics_extrinsics(intrinsics, extrinsics):
    fov_x_left, fov_y_left = get_fov(
        intrinsics["stereo_left"]["focal_length"],
        intrinsics["stereo_left"]["focal_length"],
        intrinsics["stereo_left"]["image_width"],
        intrinsics["stereo_left"]["image_height"],
        type="deg",
    )
    fov_x_right, fov_y_right = get_fov(
        intrinsics["stereo_right"]["focal_length"],
        intrinsics["stereo_right"]["focal_length"],
        intrinsics["stereo_right"]["image_width"],
        intrinsics["stereo_right"]["image_height"],
        type="deg",
    )
    intrinsics = {
        "stereo_left": {
            "fx": intrinsics["stereo_left"]["focal_length"],
            "fy": intrinsics["stereo_left"]["focal_length"],
            "cx": intrinsics["stereo_left"]["cc_x"],
            "cy": intrinsics["stereo_left"]["cc_y"],
            "distortion_coefficients": intrinsics["stereo_left"][
                "distortion_coefficients"
            ],
            "image_width": intrinsics["stereo_left"]["image_width"],
            "image_height": intrinsics["stereo_left"]["image_height"],
            "h_fov": fov_x_left,
            "v_fov": fov_y_left,
        },
        "stereo_right": {
            "fx": intrinsics["stereo_right"]["focal_length"],
            "fy": intrinsics["stereo_right"]["focal_length"],
            "cx": intrinsics["stereo_right"]["cc_x"],
            "cy": intrinsics["stereo_right"]["cc_y"],
            "distortion_coefficients": intrinsics["stereo_right"][
                "distortion_coefficients"
            ],
            "image_width": intrinsics["stereo_right"]["image_width"],
            "image_height": intrinsics["stereo_right"]["image_height"],
            "h_fov": fov_x_right,
            "v_fov": fov_y_right,
        },
    }
    R_lidar_quat = extrinsics["lidar_front"]["quaternion"]
    R_lidar = Rotation.from_quat(R_lidar_quat).as_matrix()
    t_lidar = extrinsics["lidar_front"]["translation"]

    qL = extrinsics["stereo_left"]["quaternion"]
    tL = extrinsics["stereo_left"]["translation"]
    qR = extrinsics["stereo_right"]["quaternion"]
    tR = extrinsics["stereo_right"]["translation"]
    RL = Rotation.from_quat(qL).as_matrix()
    RR = Rotation.from_quat(qR).as_matrix()

    T_POINTS_WORLD_FROM_LEFT = np.diag([1.0] * 4)
    T_POINTS_WORLD_FROM_LEFT[:3, :3] = RL
    T_POINTS_WORLD_FROM_LEFT[:3, 3] = tL

    T_POINTS_WORLD_FROM_RIGHT = np.diag([1.0] * 4)
    T_POINTS_WORLD_FROM_RIGHT[:3, :3] = RR
    T_POINTS_WORLD_FROM_RIGHT[:3, 3] = tR

    T_POINTS_RIGHT_FROM_LEFT = (
        invert_transformation(T_POINTS_WORLD_FROM_RIGHT) @ T_POINTS_WORLD_FROM_LEFT
    )
    # T = TL @ invert_transformation(TR)
    R_POINTS_RIGHT_FROM_LEFT = T_POINTS_RIGHT_FROM_LEFT[:3, :3]
    t_POINTS_RIGHT_FROM_LEFT = T_POINTS_RIGHT_FROM_LEFT[:3, 3]

    extrinsics = {
        "rotation_matrix": R_POINTS_RIGHT_FROM_LEFT,
        "translation": t_POINTS_RIGHT_FROM_LEFT,
    }
    return intrinsics, extrinsics, np.array(tL), np.array(RL), np.array(R_lidar), np.array(t_lidar)

def pohang_2_extract_roll_pitch_yaw(file_path):
    roll_pitch_yaw_list = []
    
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            ts_euler = np.zeros(4)
            if len(parts) >= 5:  # Ensuring there are at least 5 values (timestamp, x, y, z, w)
                ts = float(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                w = float(parts[4])
                quat = np.array([x, y, z, w])
                euler = Rotation.from_quat(quat).as_euler('xyz')
                ts_euler[1:] = euler
                ts_euler[0] = ts
                roll_pitch_yaw_list.append(ts_euler)
                
    return np.array(roll_pitch_yaw_list)

def pohang_2_extract_camera_timstamps(file_path):

    timestamps = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            ts_img = [float(parts[0]), parts[1]]  # Second value stays as a string
            timestamps.append(ts_img) 
    
    return np.array(timestamps)

def pohang_2_match_ahrs_timestamps(image_timestamps, ahrs_data):
    closest_indices = []

    # Ensure the timestamps are floats
    ahrs_data[:, 0] = ahrs_data[:, 0].astype(np.float64)
    image_timestamps_float = image_timestamps[:, 0].astype(np.float64)

    for img_ts in image_timestamps_float:
        # Find the index of the closest AHRS timestamp
        closest_index = np.argmin(np.abs(np.array(ahrs_data[:, 0]) - img_ts))
        closest_indices.append(closest_index)
    
    ahrs_data_matched = ahrs_data[closest_indices]

    return ahrs_data_matched

def pohang_2_extract_lidar_timestamps(bin_files):
    """
    Extracts timestamps from LiDAR .bin file names.
    
    Args:
        bin_files: List of LiDAR .bin file paths.
        
    Returns:
        lidar_timestamps: List of timestamps extracted from the file names.
    """
    lidar_timestamps = []
    for file in bin_files:
        # Extract the file name without the extension, which is the timestamp
        timestamp_str = os.path.splitext(os.path.basename(file))[0]
        lidar_timestamps.append(int(timestamp_str))  # Convert to integer or float
    return np.array(lidar_timestamps)

def pohang_2_match_lidar_timestamps(image_timestamps, lidar_data_path):
    unsorted_lidar_data = [
    f
    for f in os.listdir(lidar_data_path)
    if os.path.isfile(os.path.join(lidar_data_path, f))
    ]
    lidar_data = sorted(unsorted_lidar_data, key=lambda x: int(x[:19]))
    print(len(lidar_data))
    lidar_timestamps = pohang_2_extract_lidar_timestamps(lidar_data)
    closest_indices = []
    #image_timestamps_float = image_timestamps[:, 0].astype(np.float64)

    for img_ts in image_timestamps[:, 0]:
        # Find the index of the closest lidar timestamp
        cleaned_img_ts = img_ts.replace('.', '')
        cleaned_img_ts_float = np.float64(cleaned_img_ts)
        #print(cleaned_img_ts_float)
        #print(lidar_timestamps)
        closest_index = np.argmin(np.abs(lidar_timestamps - cleaned_img_ts_float))
        closest_indices.append(closest_index)

    #print(closest_indices)
    lidar_data_matched = lidar_data[closest_indices]

    return lidar_data_matched

def load_intrinsics_and_extrinsics(intrinsics_file, extrinsics_file):
    with open(intrinsics_file, 'r') as f:
        intrinsics = json.load(f)
    
    with open(extrinsics_file, 'r') as f:
        extrinsics = json.load(f)
    
    return intrinsics, extrinsics

def visualize_lidar_points(points):
    """
    Visualize the 3D LiDAR points using matplotlib.
    
    Args:
        points: Nx3 numpy array containing the x, y, z coordinates of the points.
    """
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap='viridis', s=0.5)
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()

def usvinland_extrinsics():
    data = json.load("usvinland.json")
    extrinsics = data["extrinsics"]

    RL = extrinsics["stereo_left"]["rotation_matrix"]
    # tL = extrinsics["stereo_left"]["translation"]
    RR = extrinsics["stereo_right"]["rotation_matrix"]
    # tR = extrinsics["stereo_right"]["translation"]
    PL = extrinsics["stereo_left"]["projection_matrix"]
    PR = extrinsics["stereo_right"]["projection_matrix"]

    T_POINTS_WORLD_FROM_LEFT = np.diag([1.0] * 4)
    T_POINTS_WORLD_FROM_LEFT[:3, :3] = RL
    # T_POINTS_WORLD_FROM_LEFT[:3, 3] = tL

    T_POINTS_WORLD_FROM_RIGHT = np.diag([1.0] * 4)
    T_POINTS_WORLD_FROM_RIGHT[:3, :3] = RR
    # T_POINTS_WORLD_FROM_RIGHT[:3, 3] = tR

    T_POINTS_RIGHT_FROM_LEFT = (
        invert_transformation(T_POINTS_WORLD_FROM_RIGHT) @ T_POINTS_WORLD_FROM_LEFT
    )
    # T = TL @ invert_transformation(TR)
    R_POINTS_RIGHT_FROM_LEFT = T_POINTS_RIGHT_FROM_LEFT[:3, :3]
    t_POINTS_RIGHT_FROM_LEFT = T_POINTS_RIGHT_FROM_LEFT[:3, 3]

    extrinsics = {
        "rotation_matrix": R_POINTS_RIGHT_FROM_LEFT,
        "translation": t_POINTS_RIGHT_FROM_LEFT,
    }


def read_mat_file(file_path):
    data = scipy.io.loadmat(file_path)
    return data


def find_lowest_y(f, x, start, end):
    while start < end:
        mid = start + (end - start) // 2
        if mid >= f(x):
            end = mid
        else:
            start = mid + 1
    return start


def create_wateredge_mask(points, H, W):
    # Sort points by x-coordinate
    points = points[np.argsort(points[:, 0])]

    # Interpolate between points to define the edge
    f = interp1d(
        points[:, 0], points[:, 1], kind="linear", assume_sorted=True
    )  # , fill_value='extrapolate')

    # Create an empty mask array
    mask = np.zeros((H, W), dtype=np.uint8)

    # # Iterate over each pixel in the mask array
    # for x in range(W):
    #     for y in range(H):
    #         # Determine if the pixel is below the water edge
    #         if y >= f(x):
    #             mask[y:, x] = 1
    #             break
    # Iterate over each pixel in the mask array
    for x in range(W):
        # Find the lowest y such that y >= f(x)
        lowest_y = find_lowest_y(f, x, 0, H)
        # Set mask[y:, x] to 1
        mask[lowest_y:, x] = 1
    return mask


def remove_obstacles_from_watermask(water_mask, obstacles):
    # remove bbox of obstacles from water mask
    obstacles_mask = np.zeros_like(water_mask)
    for obstacle in obstacles:
        x, y, w, h = obstacle
        x, y, w, h = int(x), int(y), int(w), int(h)
        obstacles_mask[y : y + h, x : x + w] = 1
    not_obstacles_mask = 1 - obstacles_mask

    water_mask = np.logical_and(water_mask, not_obstacles_mask)

    return water_mask


def sync_timestamps(ts1, ts2):
    """Synchronizes timestamps

    Args:
        ts1 np.array: (N,)
        ts2 np.array: (M,)

    Returns:
        ts1,ts2 indexes
    """
    if max(ts1.shape[0], ts2.shape[0]) == ts1.shape[0]:
        ts1_idx = np.searchsorted(ts1, ts2, side="left")
        ts1_idx = np.clip(ts1_idx - 1, 0, len(ts1) - 1)
        ts2_idx = np.arange(0, ts2.shape[0])
    else:
        ts2_idx = np.searchsorted(ts2, ts1, side="left")
        ts2_idx = np.clip(ts2_idx - 1, 0, len(ts2) - 1)
        ts1_idx = np.arange(0, ts1.shape[0])

    return ts1_idx, ts2_idx


def find_closest_numbers_idx(num, arr):
    closest_left = None
    closest_right = None

    for i, x in enumerate(arr):
        if x < num:
            if closest_left is None or abs(x - num) < abs(closest_left - num):
                closest_left = x
                closest_left_idx = i
        elif x > num:
            if closest_right is None or abs(x - num) < abs(closest_right - num):
                closest_right = x
                closest_right_idx = i

    return closest_left_idx, closest_right_idx


def normalize_img(img, scale = 1):
    img = img.astype(np.float32)
    img -= img.min()
    img /= img.max()
    img *= scale
    img = img.astype(np.uint8)
    return img

def homog(vec):
    return np.concatenate([vec, [1]])

def dehomog(vec):
    return vec[:-1] / vec[-1]


def get_bottommost_line(mask, thickness=5):
    height, width = mask.shape
    reversed_mask = mask[::-1, :] > 0

    first_positive = np.argmax(reversed_mask, axis=0)
    has_positive = np.any(reversed_mask, axis=0)

    bottom_indices = np.full(width, -1, dtype=int)
    bottom_indices[has_positive] = height - 1 - first_positive[has_positive]

    lower_bound = np.clip(bottom_indices - thickness + 1, 0, height - 1)
    upper_bound = bottom_indices

    rows = np.arange(height).reshape(-1, 1)
    mask_band = (rows >= lower_bound) & (rows <= upper_bound) & (bottom_indices >= 0)

    return mask_band.astype(np.uint8)

def get_water_mask_from_contour_mask(contour_mask, offset=30):
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


def filter_mask_by_boundary(mask, boundary_indices, offset=30):
    height, _ = mask.shape
    adjusted_boundary = np.clip(boundary_indices - offset, 0, height - 1)
    row_indices = np.arange(height).reshape(-1, 1)
    keep_mask = row_indices < adjusted_boundary
    return np.where(keep_mask, mask, 0)

def write_coordinates_to_file(filename, frame, coordinates):

    coordinates_list = [list(coord) for coord in coordinates]
    
    data = {
        "frame": frame,
        "points": coordinates_list
    }
    
    # Open the file in append mode; if it doesn't exist, it will be created
    with open(filename, 'a') as file:
        json.dump(data, file)
        file.write("\n")

def find_closest_timestamp(timestamps, target_timestamp):
    idx = np.abs(timestamps - target_timestamp).argmin()
    return idx, timestamps[idx]