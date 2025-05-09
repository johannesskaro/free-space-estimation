import json
from shapely.geometry import LineString
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.ndimage import rotate
from pyproj import Transformer
from scipy.spatial.transform import Rotation as ROT
import math
import numpy as np
from transforms import *
import cv2
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


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

    #stixel_points = stixel_points[:, [1, 0]]
    #stixel_points_poly = stixel_points[stixel_validity]

    stixel_points = stixel_points[:, [0, 1]]
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

    r = ROT.from_quat(gnss_ori)
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



    #stixel_points_global_poly = transform_stixel_points_global_2(stixel_points_poly, boat_position, yaw)
    stixel_points_global_poly = transform_stixel_points_global(stixel_points_poly, curr_pose[:3], gnss_ori)

    #stixel_points_global = stixel_points_global_poly[:len(dynamic_list)]
    stixel_points_global = transform_stixel_points_global(stixel_points, curr_pose[:3], gnss_ori)
    stixel_points_global = stixel_points_global[:len(dynamic_list)]

    xs, ys = zip(*stixel_points_global_poly)
    ax.fill(ys, xs, color='cyan', alpha=0.3, label="Free Space")
    ax.plot(ys, xs, color='cyan', zorder=-10)

    plt.scatter(boat_position[0], boat_position[1], s=50, color='green', label="Ego Vessel")

    for n, (x, y) in enumerate(stixel_points_global):
        if stixel_validity[n]:
            #print("stixel_validity", stixel_validity[n])
            plt.scatter(y, x, color='blue', marker='o', s=50, label="Stixels" if 'Stixels' not in plt.gca().get_legend_handles_labels()[1] else "")

            #if using_prop_depth[n]:
            #    plt.scatter(x, y, color='yellow', marker='o', s=50, label="Propagated" if 'Propagated' not in plt.gca().get_legend_handles_labels()[1] else "")
            #elif dynamic_list[n]:
            #    plt.scatter(x, y, color='red', marker='o', s=50, label="Boat" if 'Boat' not in plt.gca().get_legend_handles_labels()[1] else "")
            #else:
            #    plt.scatter(x, y, color='blue', marker='o', s=50, label="Static" if 'Static' not in plt.gca().get_legend_handles_labels()[1] else "")

    
    ax.set_xlabel("East [m]", fontsize=16)
    ax.set_ylabel("North [m]", fontsize=16)
    #ax.set_title("Free Space Estimation")
    ax.add_artist(ab)

    ax.invert_yaxis()
    ax.invert_xaxis()
    #ax.set_xlim(-265, -335)
    #ax.set_ylim(-490, -565)
    #plt.grid(True)
    #plt.show(block=False)
    #plt.pause(1)  # Display the plot for a short period
    #plt.close()

    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(fontsize=14)

    #fig.canvas.draw()
    #img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    #img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3].copy()
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    plt.close(fig)
    return img_bgr

def build_H_world_from_cam(boat_pos, boat_quat, H_cam_from_boat):
    # boat → world
    H_world_from_boat           = np.eye(4)
    H_world_from_boat[:3, :3]   = R.from_quat(boat_quat).as_matrix()
    H_world_from_boat[:3, 3]    = boat_pos

    # camera → world  =  boat → world ⋅ (boat ← camera)
    H_world_from_cam = H_world_from_boat @ np.linalg.inv(H_cam_from_boat)
    return H_world_from_cam


def plot_previous_gnss_iterations_local(gnss_pos_list, gnss_ori_list, stixel_points_list):

    plt.figure(figsize=(8, 8))

    curr_boat_pos = gnss_pos_list[-1]
    curr_boat_ori = gnss_ori_list[-1]

    H_WORLD_FROM_RIGHT_ZED_CURR = build_H_world_from_cam(curr_boat_pos, curr_boat_ori, H_POINTS_RIGHT_ZED_FROM_FLOOR)

    cam_pos_list = []

    for i, gnss_pos in enumerate(gnss_pos_list):

        ori = gnss_ori_list[i]

        H_WORLD_FROM_RIGHT_ZED_PREV = build_H_world_from_cam(gnss_pos, ori, H_POINTS_RIGHT_ZED_FROM_FLOOR)

        H_CURR_FROM_PREV = np.linalg.inv(H_WORLD_FROM_RIGHT_ZED_PREV) @ H_WORLD_FROM_RIGHT_ZED_CURR

        cam_pos_local = H_CURR_FROM_PREV @ np.array([0, 0, 0, 1])
        cam_x_local, cam_y_local = cam_pos_local[0], cam_pos_local[2]

        cam_pos_list.append((cam_x_local, cam_y_local))

    
    cmap = plt.get_cmap("Blues")
    num_scans = len(stixel_points_list)


    for i, stixel_points in enumerate(stixel_points_list):

        pos = gnss_pos_list[i]
        ori = gnss_ori_list[i]

        stixel_points_local = transform_stixel_points_local(stixel_points, H_WORLD_FROM_RIGHT_ZED_CURR, pos, ori)
        #stixel_points_local = stixel_points

        xs, ys = zip(*stixel_points_local)
        color = cmap(i/num_scans)

        scan_alpha = 0.005 + 0.2*(i/num_scans)**3
        #plt.fill(xs, ys, color='cyan', alpha=scan_alpha, label=f"Scan {num_scans-i}")
        #plt.scatter(xs, ys, color='blue', s=50, alpha=scan_alpha, label=f"Scan {num_scans-i}")
        label = "Stixels" if i == len(stixel_points_list) - 1 else ""
        #plt.scatter(xs, ys, color='blue', s=10, alpha=scan_alpha, label=label)
        plt.scatter(xs, ys, color=color, s=10, alpha=scan_alpha, label=label)

    norm = Normalize(vmin=0, vmax=num_scans-1)          # 0 = oldest scan, N-1 = newest
    sm   = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])                            # required for older MPL versions

    cbar = plt.colorbar(sm, ax=plt.gca(), pad=0.01, aspect=35)
    cbar.set_label("Older  ←  scan index  →  Newer", fontsize=12)
    cbar.set_ticks([])
    cbar.ax.tick_params(size=0)

    gnss_x = [pt[0] for pt in cam_pos_list]
    gnss_y = [pt[1] for pt in cam_pos_list]
    
    for xi, yi in zip(gnss_x, gnss_y):
        plt.scatter(xi, yi, color='green', s=10)

    plt.scatter(0, 0, color='green', marker='*', s=200, label='Camera position')

        
    plt.xlabel("Z (m)")
    plt.ylabel("X (m)")
    #plt.title("GNSS Data and Free Space")
    ax = plt.gca()

    ax.set_xlabel("X [m]", fontsize=16)
    ax.set_ylabel("Z [m]", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(fontsize=14)



    plt.grid(True)
    plt.savefig("images/bev_consistency_200_frames_v14.png", dpi=300, bbox_inches='tight')


    #plt.ioff()
    #plt.show()
    plt.close()
    #plt.show(block=False)
    #plt.pause(1)  # Display the plot for a short period
    #plt.close()


def plot_gnss_iteration_video_local(curr_pose, stixel_points, dynamic_list, stixel_validity, using_prop_depth):


    stixel_points = stixel_points[:, [1, 0]]
    stixel_points_poly = stixel_points[stixel_validity]

    gnss_pos = curr_pose[:2]
    gnss_ori = curr_pose[3:]

    #fig = plt.figure(figsize=(8, 8))

    width, height = 1080, 1080
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_subplot(1,1,1)


    r = ROT.from_quat(gnss_ori)
    euler_angles = r.as_euler('xyz', degrees=False)
    yaw = euler_angles[2]
    cam_position = [0, 0]
    boat_position = [cam_position[0] + TRANS_FLOOR_TO_LIDAR[1], cam_position[1] + TRANS_FLOOR_TO_LIDAR[0]]

    boat_img = plt.imread("files/ferry.png")
    rotation_angle = math.degrees(0)
    rotated_boat_img = rotate(boat_img, angle=rotation_angle, reshape=True)
    rotated_boat_img = np.clip(rotated_boat_img, 0, 1)
    img_box = OffsetImage(rotated_boat_img, zoom=0.08)  #zoom=0.08
    ab = AnnotationBbox(img_box, boat_position, frameon=False, box_alignment=(0.5, 0.5))
    ab.set_zorder(-10)

    stixel_points_poly = np.vstack((stixel_points_poly, cam_position))
    stixel_points_poly = np.vstack((stixel_points_poly, stixel_points_poly[0])) # closing the the polygon

    xs, ys = zip(*stixel_points_poly)
    ax.fill(xs, ys, color='cyan', alpha=0.3, label="Free Space")
    ax.plot(xs, ys, color='cyan', zorder=-10)

    plt.scatter(boat_position[0], boat_position[1], s=50, color='green', label="Ego Vessel")

    for n, (x, y) in enumerate(stixel_points):
        if stixel_validity[n]:
            
            if using_prop_depth[n]:
                plt.scatter(x, y, color='yellow', marker='o', s=50, label="Propagated" if 'Propagated' not in plt.gca().get_legend_handles_labels()[1] else "")
            elif dynamic_list[n]:
                plt.scatter(x, y, color='red', marker='o', s=50, label="Boat" if 'Boat' not in plt.gca().get_legend_handles_labels()[1] else "")
            else:
                plt.scatter(x, y, color='blue', marker='o', s=50, label="Static" if 'Static' not in plt.gca().get_legend_handles_labels()[1] else "")

    
    ax.set_xlabel("East [m]", fontsize=16)
    ax.set_ylabel("North [m]", fontsize=16)
    #ax.set_title("Free Space Estimation")
    ax.add_artist(ab)

    #ax.set_xlim(-35, 35)
    #ax.set_ylim(-10, 65)
    ax.set_xlim(-12, 12)
    ax.set_ylim(-5, 15)
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


def transform_stixel_points_global_2(stixel_points, boat_position, heading):
    offset_y = TRANS_FLOOR_TO_LIDAR[0]
    offset_x = TRANS_FLOOR_TO_LIDAR[1]

    transformed_points = []

    heading_offset = 0 #np.deg2rad(3) # math.radians(1.5)
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


def transform_stixel_points_global(stixel_points, boat_position, boat_ori):
    rot = ROT.from_quat(boat_ori)

    roll, pitch, yaw = rot.as_euler('xyz')
    yaw = -yaw
    rot_flipped = ROT.from_euler('xyz', [roll, pitch, yaw])

    boat_position = np.array([boat_position[1], boat_position[0], boat_position[2]])

    R_world_to_body = rot.as_matrix()

    H_WORLD_FROM_FLOOR = np.eye(4)
    H_WORLD_FROM_FLOOR[:3, :3] = R_world_to_body 
    H_WORLD_FROM_FLOOR[:3, 3] = boat_position

    H_WORLD_FROM_RIGHT_ZED = H_WORLD_FROM_FLOOR @ np.linalg.inv(H_POINTS_RIGHT_ZED_FROM_FLOOR)

    points_cam = np.array([[x, 0, z, 1] for z, x in stixel_points]) # Convert from ned to cam coordinates
    points_world = (H_WORLD_FROM_RIGHT_ZED @ points_cam.T).T[:, :2]

    cam_origin_world = (H_WORLD_FROM_RIGHT_ZED @ np.array([0, 0, 0, 1]))[:2]
    points_world = np.vstack([points_world, cam_origin_world, points_world[0]])

    #print("points_world", points_world)

    return points_world


def transform_stixel_points_local(stixel_points, H_WORLD_FROM_RIGHT_ZED_CURR, boat_position, boat_ori):

    H_WORLD_FROM_RIGHT_ZED_PREV = build_H_world_from_cam(boat_position, boat_ori, H_POINTS_RIGHT_ZED_FROM_FLOOR)
    H_CURR_FROM_PREV = np.linalg.inv(H_WORLD_FROM_RIGHT_ZED_PREV) @ H_WORLD_FROM_RIGHT_ZED_CURR

    points_cam = np.array([[x, 0, z, 1] for z, x in stixel_points]) # Convert from ned to cam coordinates
    points_local = (H_CURR_FROM_PREV @ points_cam.T).T[:, [0, 2]]


    return points_local







if __name__ == "__main__":
    file_path = "files/linestrings.json"
    line_strings = get_line_strings_from_file(file_path)
    plot_line_strings(line_strings, origin=[400000000, -50000])
    #plot_line_strings(line_strings)