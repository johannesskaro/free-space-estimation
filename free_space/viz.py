"""Visualization helpers. None of these affect the pipeline output."""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def blend_image_with_mask(img, mask, color=[0, 0, 255], alpha1=1, alpha2=1):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    colored_mask = np.zeros_like(img)
    colored_mask[mask > 0] = color
    return cv2.addWeighted(img, alpha1, colored_mask, alpha2, 0)


def overlay_stixels_on_image(image, stixels, min_depth=0, max_depth=60):
    """Draw valid stixels, colored by fused depth."""
    overlay = np.zeros_like(image)
    cmap = plt.get_cmap('gist_earth')
    w = stixels.stixel_width

    for n, (stixel_top, stixel_base) in enumerate(stixels.height_base_list):
        if stixel_base > stixel_top and w > 0:
            if not stixels.stixel_validity[n]:
                continue
            stixel_depth = stixels.stixel_fused_depths[n]
            norm_depth = np.clip((stixel_depth - min_depth) / (max_depth - min_depth), 0, 1)
            rgba = cmap(norm_depth)
            color = (int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255))

            overlay[stixel_top:stixel_base, n * w:(n + 1) * w] = np.full((stixel_base - stixel_top, w, 3), color, dtype=np.uint8)
            cv2.rectangle(overlay, (n * w, stixel_top), ((n + 1) * w, stixel_base), [0, 0, 0], 1)

    return cv2.addWeighted(image, 0.8, overlay, 0.8, 0.0)


def get_lidar_points_in_stixels(stixels, xyz_proj, xyz_c):
    """Lidar points that fall inside a valid stixel."""
    filtered_image_points = []
    filtered_3d_points = []

    for n, (stixel_top, stixel_base) in enumerate(stixels.height_base_list):
        if not stixels.stixel_validity[n]:
            continue
        left_bound = n * stixels.stixel_width
        right_bound = (n + 1) * stixels.stixel_width

        mask = (
            (xyz_proj[:, 1] >= stixel_top) &
            (xyz_proj[:, 1] <= stixel_base) &
            (xyz_proj[:, 0] >= left_bound) &
            (xyz_proj[:, 0] <= right_bound)
        )
        filtered_image_points.extend(xyz_proj[mask])
        filtered_3d_points.extend(xyz_c[mask])

    return np.array(filtered_image_points), np.array(filtered_3d_points)


def draw_lidar_points(image, lidar_points, lidar_3d_points=None, point_size=2, max_value=60, min_value=0, alpha=1):
    """Draw projected lidar points, colored by depth if lidar_3d_points is given."""
    image_with_lidar = image.copy()
    height, width = image.shape[:2]
    lidar_overlay = np.zeros_like(image_with_lidar)

    if lidar_3d_points is not None and len(lidar_3d_points) > 0:
        depths = lidar_3d_points[:, 2]
        depths_normalized = np.clip((depths - min_value) / (max_value - min_value), 0, 1)
    else:
        depths_normalized = np.ones(len(lidar_points))

    colormap = plt.get_cmap('gist_earth')

    for i, point in enumerate(lidar_points):
        x, y = int(round(point[0])), int(round(point[1]))
        if 0 <= x < width and 0 <= y < height:
            rgba = colormap(depths_normalized[i])
            color = (int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255))
            cv2.circle(lidar_overlay, (x, y), point_size, color, -1)

    return cv2.addWeighted(image_with_lidar, alpha, lidar_overlay, 1, 0.0)


def plot_bev(stixels, size_px=1080, xlim=(-35, 35), ylim=(-10, 65)):
    """Bird's-eye view of the free space in the camera frame, as a BGR image.

    Blue: static obstacle, red: dynamic (boat), yellow: depth propagated from
    the previous frame because there was no lidar measurement.
    """
    stixel_points = stixels.stixel_footprints[:, [1, 0]]  # (right, forward)
    validity = stixels.stixel_validity

    dpi = 100
    fig = plt.figure(figsize=(size_px / dpi, size_px / dpi), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)

    poly = np.vstack((stixel_points[validity], [0, 0]))
    poly = np.vstack((poly, poly[0]))
    xs, ys = poly[:, 0], poly[:, 1]
    ax.fill(xs, ys, color='cyan', alpha=0.3, label="Free Space")
    ax.plot(xs, ys, color='cyan', zorder=-10)
    ax.scatter(0, 0, s=50, color='green', label="Camera Center")

    groups = [
        ("Propagated", 'yellow', validity & stixels.using_prop_depth),
        ("Dynamic", 'red', validity & ~stixels.using_prop_depth & stixels.dynamic_stixel_list),
        ("Static", 'blue', validity & ~stixels.using_prop_depth & ~stixels.dynamic_stixel_list),
    ]
    for label, color, sel in groups:
        if sel.any():
            ax.scatter(stixel_points[sel, 0], stixel_points[sel, 1], color=color, marker='o', s=50, label=label)

    ax.set_xlabel("X [m]", fontsize=16)
    ax.set_ylabel("Z [m]", fontsize=16)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(loc='upper right', fontsize=14)

    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

