import cv2
import numpy as np
import scipy.io
from scipy.interpolate import interp1d
import json
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import njit, prange
from shapely.geometry import Polygon, Point
from scipy.spatial.transform import Rotation as R
import random


def blend_image_with_mask(img, mask, color=[0, 0, 255], alpha1=1, alpha2=1):

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
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

def make_homog(x, y, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        [   c, s,   x],
        [   -s, c,   y],
        [   0,  0,   1],
    ])


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

def get_delta_heading(pose_prev, pose_curr):
        ori_prev = pose_prev[3:]
        ori_curr = pose_curr[3:]

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

def distance_from_point_to_line(point, line):
    a, b, c = line
    z, x = point
    dist = np.abs(a*z +b*x + c) / np.sqrt(a**2 + b**2)
    #r = np.linalg.norm(point)
    r = np.hypot(z, x)
    scaled_dist = dist / max(r, 1e-10)
    return scaled_dist

def distance_from_point_to_line_2(point, line):
    a, b, c = line
    z, x = point
    dist = np.abs(a*z +b*x + c) / np.sqrt(a**2 + b**2)
    return dist

def angular_error(point, line, min_r=1e-3):

    a, b, c = line
    z, x = point

    # 1) form the 2D vector from the origin to the point
    p = np.array([z, x], dtype=float)
    r = np.linalg.norm(p)
    if r < min_r:
        return np.inf

    # 2) get a direction vector along the line by rotating the normal (a,b)
    #    any vector perpendicular to the normal is along the line:
    v = np.array([-b, a], dtype=float)

    # 3) normalize both
    p_hat = p / r
    v_hat = v / np.linalg.norm(v)

    # 4) compute the cosine of the angle and then acos
    cos_theta = np.dot(p_hat, v_hat)
    # if you want the acute angle to the infinite line (no ray‐sign), do:
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


def rotation_matrix(theta):
    # Rotation around down axis
    return np.array([[np.cos(theta), np.sin(theta)],
                     [- np.sin(theta),  np.cos(theta)]])

def transform_matrix_2d(theta, t):
    c, s = np.cos(theta), np.sin(theta)
    T = np.array([
        [c, -s, t[0]],
        [s,  c, t[1]],
        [0,  0,   1 ]
    ])
    return T

def transform_matrix(R, t):
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t
    return H

def invert_transformation(H):
    R = H[:3, :3]
    t = H[:3, 3]
    H_transformed = np.block(
        [
            [R.T, -R.T.dot(t)[:, np.newaxis]],
            [np.zeros((1, 3)), np.ones((1, 1))],
        ]
    )
    return H_transformed


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

def read_mat_file(file_path):
    data = scipy.io.loadmat(file_path)
    return data



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


def get_high_contrast_colors(n, cmap_name='jet'):
    base_colors = plt.cm.get_cmap(cmap_name)(np.linspace(0, 1, n))
    if n <= 1:
        return base_colors
    
    # Golden ratio conjugate
    phi = 0.618033988749895  
    # Compute a permutation index using the fractional part of (index * phi)
    indices = np.argsort((np.arange(n) * phi) % 1)
    
    return base_colors[indices]


def plot_sam_masks_cv2(image, masks):
    """
    Plots all segmentation masks from a Segment Anything Model (SAM) in different bright colors using OpenCV.
    Each mask is overlaid on the original image with transparency and outlined with a contour line.

    :param image: Original image (H, W, 3) in NumPy format.
    :param masks: Binary masks (N, H, W), where N is the number of detected objects.
    """
    # Ensure the image is in BGR format
    if len(image.shape) == 2 or image.shape[-1] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    overlay = image.copy()
    num_masks = masks.shape[0]
    # Generate bright colors (each channel between 150 and 255)
    colors = [tuple(random.randint(50, 255) for _ in range(3)) for _ in range(num_masks)]
    alpha = 0.8  # transparency factor for the mask fill

    for i in range(num_masks):
        mask = masks[i]
        color = colors[i]
        
        # Create a colored overlay for the mask region
        color_mask = np.zeros_like(image, dtype=np.uint8)
        color_mask[mask == 1] = color
        
        # Blend the colored overlay with the original image
        overlay[mask == 1] = cv2.addWeighted(overlay[mask == 1], 1 - alpha,
                                             color_mask[mask == 1], alpha, 0)
        
        # Convert mask to uint8 format (values 0 or 255)
        mask_uint8 = (mask.astype(np.uint8) * 255)
        # Find contours on the binary mask
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw the contour line around the mask region
        cv2.drawContours(overlay, contours, -1, color, 2)
    
    cv2.imshow("SAM Mask Visualization", overlay)

    return overlay


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


def write_coordinates_to_file(filename, frame, coordinates, validity=None, dynamic=None, depth_uncertainty=None):

    coordinates_list = [list(coord) for coord in coordinates]

    validity_list = [int(v) for v in validity] if validity is not None else None
    dynamic_list = [int(d) for d in dynamic] if dynamic is not None else None
    depth_uncertainty_list = [float(d) for d in depth_uncertainty] if depth_uncertainty is not None else None
    
    data = {
        "frame": frame,
        "points": coordinates_list,
        "validity": validity_list,
        "dynamic": dynamic_list,
        "depth_uncertainty": depth_uncertainty_list
    }
    
    # Open the file in append mode; if it doesn't exist, it will be created
    with open(filename, 'a') as file:
        json.dump(data, file)
        file.write("\n")

def find_closest_timestamp(timestamps, target_timestamp):
    idx = np.abs(timestamps - target_timestamp).argmin()
    return idx, timestamps[idx]


def merge_lidar_onto_image(image, lidar_points, lidar_3d_points=None, intensities=None, point_size=2, max_value=60, min_value=0, alpha=1):


    if intensities is not None and len(intensities.shape) == 2:
        intensities = np.squeeze(intensities, axis=1)  # From (N, 1) to (N,)

    image_with_lidar = image.copy()
    height, width = image.shape[:2]

    # Create a separate overlay for the lidar points
    lidar_overlay = np.zeros_like(image_with_lidar)

    if lidar_3d_points is not None:
        if lidar_3d_points.ndim == 1:
            depths = None  # Already 1D: each element is a depth value
        else:
            depths = lidar_3d_points[:, 2]
    else:
        depths = None

    # If intensities are provided, ensure they match the number of points
    if depths is not None:
        if len(depths) != len(lidar_points):
            raise ValueError("The length of intensities must match the number of lidar points.")

        # Normalize intensities
        if max_value is None:
            max_value = np.max(depths)
        if min_value is None:
            min_value = np.min(depths)

        # Avoid division by zero
        if max_value == min_value:
            max_value = min_value + 1

        depths_normalized = (depths - min_value) / (max_value - min_value)
        depths_normalized = np.clip(depths_normalized, 0, 1)
    else:
        # Use a default intensity of 1 for all points if no intensities are provided
        depths_normalized = np.ones(len(lidar_points))
    
    # Use the 'Reds' colormap
    #colormap = plt.get_cmap('Reds')
    colormap = plt.get_cmap('gist_earth')

    # Draw points on the lidar overlay image
    for i, point in enumerate(lidar_points):
        x, y = int(round(point[0])), int(round(point[1]))
        if 0 <= x < width and 0 <= y < height:
            value_norm = depths_normalized[i]
            rgba = colormap(value_norm)  # returns RGBA, take RGB
            #rgba = colormap(1.0 - value_norm)
            color = (int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255))
            #color = (0, 0, 255)  # Red color for the point
            cv2.circle(lidar_overlay, (x, y), point_size, color, -1)

            #color = tuple(int(c * 255) for c in color[::-1])  # convert to BGR
            #color = (0, 0, 255)
            #cv2.circle(lidar_overlay, (x, y), point_size, color, -1)

    # Blend the original image and the lidar overlay
    #alpha = 0.8 #1  # Weight of the original image
    beta = 1 # 0.8   # Weight of the overlay
    gamma = 0.0  # Scalar added to each sum
    image_with_lidar = cv2.addWeighted(image_with_lidar, alpha, lidar_overlay, beta, gamma)

    return image_with_lidar

def filter_point_cloud_by_image(xyz_proj, xyz_c, height, width):

    x = xyz_proj[:, 0]
    y = xyz_proj[:, 1]

    valid_mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)

    xyz_proj = xyz_proj[valid_mask]
    xyz_c    = xyz_c[valid_mask]

    return xyz_proj, xyz_c