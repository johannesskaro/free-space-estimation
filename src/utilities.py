import cv2
import numpy as np
import scipy.io
from scipy.interpolate import interp1d
import json
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import njit, prange
from shapely.geometry import Polygon, Point


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



def calculate_3d_points(X, Y, d, cam_params):
    cx, cy = cam_params["cx"], cam_params["cy"]
    fx, fy = cam_params["fx"], cam_params["fy"]

    X_o = d * (X - cx) / fx
    Y_o = d * (Y - cy) / fy
    Z_o = d

    return np.array([X_o, Y_o, Z_o]).T



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


def merge_lidar_onto_image(image, lidar_points, lidar_3d_points=None, intensities=None, point_size=2, max_value=60, min_value=0):


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
            color = (int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255))
            cv2.circle(lidar_overlay, (x, y), point_size, color, -1)

            #color = tuple(int(c * 255) for c in color[::-1])  # convert to BGR
            #color = (0, 0, 255)
            #cv2.circle(lidar_overlay, (x, y), point_size, color, -1)

    # Blend the original image and the lidar overlay
    alpha = 1  # Weight of the original image
    beta = 0.8   # Weight of the overlay
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