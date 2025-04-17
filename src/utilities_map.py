import json
from shapely.geometry import LineString
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.ndimage import rotate
from pyproj import Transformer
from scipy.spatial.transform import Rotation as R
import math
import numpy as np
import pymap3d
from transforms import *
import cv2


def transform_to_utm32(lon, lat):
    transformer = Transformer.from_crs("epsg:4326", "epsg:32632", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y

ORIGIN = transform_to_utm32(PIREN_LON, PIREN_LAT)

def get_line_strings_from_file(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)

    # Convert back to LineString objects
    line_strings = [LineString(coords) for coords in data]
    return line_strings

LINE_STRINGS = get_line_strings_from_file("files/linestrings.json")

def plot_line_strings(ax, line_strings, origin=[0, 0]):
    origin_x = origin[0]
    origin_y = origin[1]
    

    plt.figure(figsize=(8, 8))
    for ls in line_strings:
        x, y = ls.xy
        x = [x_ - origin_x for x_ in x]
        y = [y_ - origin_y for y_ in y]
        ax.plot(x, y, color='black', linestyle='-')

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("UTM32 LineStrings")
    #plt.grid(True)
    #plt.show()

    return ax


def plot_gnss_iteration_video(curr_pose, stixel_points, dynamic_list, stixel_validity, using_prop_depth):

    origin = ORIGIN
    line_strings = LINE_STRINGS

    stixel_points = stixel_points[:, [1, 0]]
    stixel_points_poly = stixel_points[stixel_validity]

    gnss_pos = curr_pose[:2]
    gnss_ori = curr_pose[3:]

    #fig = plt.figure(figsize=(8, 8))

    width, height = 1080, 1080
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_subplot(1,1,1)

    for ls in line_strings:
        x, y = ls.xy
        x = [xi - origin[0] for xi in x]
        y = [yi - origin[1] for yi in y]
        ax.plot(x, y, color='black', linestyle='-')

    r = R.from_quat(gnss_ori)
    euler_angles = r.as_euler('xyz', degrees=False)
    yaw = euler_angles[2]
    boat_position = gnss_pos

    boat_img = plt.imread("files/ferry.png")
    rotation_angle = math.degrees(-yaw)
    rotated_boat_img = rotate(boat_img, angle=rotation_angle, reshape=True)
    rotated_boat_img = np.clip(rotated_boat_img, 0, 1)
    img_box = OffsetImage(rotated_boat_img, zoom=0.08)
    ab = AnnotationBbox(img_box, boat_position, frameon=False, box_alignment=(0.5, 0.5))
    ab.set_zorder(-10)



    stixel_points_global_poly = transform_stixel_points(stixel_points_poly, boat_position, yaw)

    #stixel_points_global = stixel_points_global_poly[:len(dynamic_list)]
    stixel_points_global = transform_stixel_points(stixel_points, boat_position, yaw)
    stixel_points_global = stixel_points_global[:len(dynamic_list)]

    xs, ys = zip(*stixel_points_global_poly)
    ax.fill(xs, ys, color='cyan', alpha=0.3, label="Free Space")
    ax.plot(xs, ys, color='cyan', zorder=-10)

    plt.scatter(boat_position[0], boat_position[1], s=50, color='green', label="Ego Vessel")

    for n, (x, y) in enumerate(stixel_points_global):
        if stixel_validity[n]:

            if dynamic_list[n]:
                plt.scatter(x, y, color='red', marker='o', s=50, label="Boat" if 'Boat' not in plt.gca().get_legend_handles_labels()[1] else "")
            elif using_prop_depth[n] == True:
                plt.scatter(x, y, color='yellow', marker='o', s=50, label="Propagated" if 'Propagated' not in plt.gca().get_legend_handles_labels()[1] else "")
            else:
                plt.scatter(x, y, color='blue', marker='o', s=50, label="Static" if 'Static' not in plt.gca().get_legend_handles_labels()[1] else "")

    
    ax.set_xlabel("East [m]", fontsize=16)
    ax.set_ylabel("North [m]", fontsize=16)
    #ax.set_title("Free Space Estimation")
    ax.add_artist(ab)

    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlim(-265, -335)
    ax.set_ylim(-490, -565)
    #plt.grid(True)
    #plt.show(block=False)
    #plt.pause(1)  # Display the plot for a short period
    #plt.close()

    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(fontsize=14)

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    plt.close(fig)
    return img_bgr



def transform_stixel_points(stixel_points, boat_position, heading):
    offset_y = TRANS_FLOOR_TO_LIDAR[0]
    offset_x = TRANS_FLOOR_TO_LIDAR[1]

    transformed_points = []

    heading_offset = 0 # math.radians(1.5)
    corrected_heading = - heading + math.pi + heading_offset
    cos_theta = math.cos(corrected_heading)
    sin_theta = math.sin(corrected_heading)
    for x, y in stixel_points:
        x_rel = x - offset_x
        y_rel = y - offset_y

        global_x = boat_position[0] + (x_rel * cos_theta - y_rel * sin_theta)
        global_y = boat_position[1] + (x_rel * sin_theta + y_rel * cos_theta)
        transformed_points.append((global_x, global_y))

    cam_pos_x = boat_position[0] - (offset_x * cos_theta - offset_y * sin_theta)
    cam_pos_y = boat_position[1] - (offset_x * sin_theta + offset_y * cos_theta)
    cam_pos = (cam_pos_x, cam_pos_y)

    transformed_points.append(cam_pos) # close the polygon
    transformed_points.append(transformed_points[0]) # close the polygon

    return transformed_points





if __name__ == "__main__":
    file_path = "files/linestrings.json"
    line_strings = get_line_strings_from_file(file_path)
    plot_line_strings(line_strings, origin=[400000000, -50000])
    #plot_line_strings(line_strings)