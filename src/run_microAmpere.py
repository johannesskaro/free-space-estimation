from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
import numpy as np
import cv2
import pyzed.sl as sl
from stereo_svo_sdk4 import SVOCamera
from transforms import *
from utilities import find_closest_timestamp, get_water_mask_from_contour_mask, blend_image_with_mask, merge_lidar_onto_image, filter_point_cloud_by_image, plot_sam_masks_cv2
import time
from yolo import YoloSeg
from fastSAM import FastSAMSeg
from RWPS import RWPS
from temporal_filtering import TemporalFiltering
from stixels import Stixels
from optical_flow import OpticalFlow
from utilities_map import plot_gnss_iteration_video, plot_gnss_iteration_video_local
import os
from pyproj import Proj, transform, Transformer
from scipy.spatial.transform import Rotation
import datetime
import struct


# Watertest 1
SVO_FILE_PATH = r"C:\Users\johro\Documents\19-04-25\watertest1\watertest1_prt.svo2"
ROSBAG_NAME = "watertest1/bag"
START_TIMESTAMP = 1745066626514600000 + 70*10**9 

# Watertest 2

ROSBAG_FOLDER = r"C:\Users\johro\Documents\19-04-25"
ROSBAG_PATH = f"{ROSBAG_FOLDER}/{ROSBAG_NAME}"


stereo_cam = SVOCamera(SVO_FILE_PATH)
stereo_cam.set_svo_position_timestamp(START_TIMESTAMP)
K, D = stereo_cam.get_left_parameters()
R = stereo_cam.R
T = stereo_cam.T
focal_length = K[0,0]
baseline = np.linalg.norm(T)
height, width = 1080, 1920

typestore = get_typestore(Stores.ROS2_HUMBLE)

LIDAR_TOPIC = "/ouster/points"
GNSS_TOPIC = '/blueboat/sensors/nav_pvt' #"/microampere/sensors/nav_pvt"


def gen_svo_images():
    num_frames = 200 # 275
    curr_frame = 0
    while stereo_cam.grab() == sl.ERROR_CODE.SUCCESS: # and curr_frame < num_frames:
        image = stereo_cam.get_left_image(should_rectify=True)
        timestamp = stereo_cam.get_timestamp()
        disparity_img = stereo_cam.get_neural_disp()
        curr_frame += 1
        depth_img = stereo_cam.get_depth_image()

        yield timestamp, image, disparity_img, depth_img


def gen_ma2_lidar_points():
    lidar_data = []
    with Reader(ROSBAG_PATH) as reader:
        connections = [c for c in reader.connections if c.topic == LIDAR_TOPIC]
        assert len(connections) == 1
        for connection, timestamp, rawdata in reader.messages(connections):
            if timestamp > START_TIMESTAMP:
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                xyz = msg.data.reshape(-1, msg.point_step)[:,:12].view(dtype=np.float32)
                intensity = msg.data.reshape(-1, msg.point_step)[:,16:20].view(dtype=np.float32)
                rgb = np.tile(np.array([255, 0, 0], dtype=np.uint8)/255, (xyz.shape[0], 1))

                intensity_clipped = np.clip(intensity, 0, 100)
                
                xyz_c = H.dot(np.r_[xyz.T, np.ones((1, xyz.shape[0]))])[0:3, :].T
                
                rvec = np.zeros((1,3), dtype=np.float32)
                tvec = np.zeros((1,3), dtype=np.float32)
                distCoeff = np.zeros((1,5), dtype=np.float32)
                image_points, _ = cv2.projectPoints(xyz_c, rvec, tvec, K, distCoeff)
                
                
                xyz_c_forward = xyz_c[xyz_c[:,2] > 0]
                image_points_forward = image_points[xyz_c[:,2] > 0]
                image_points_forward = np.squeeze(image_points_forward, axis=1)

                intensity_clipped_forward = intensity_clipped[xyz_c[:,2] > 0]

                lidar_data.append([timestamp, image_points_forward, intensity_clipped_forward, xyz_c_forward])

    return lidar_data  


GNSS_MSG = """
std_msgs/Header header
uint32 itow
uint16 year
uint8 month
uint8 day
uint8 hour
uint8 min
uint8 sec
uint8 valid
uint8 VALID_DATE = 1
uint8 VALID_TIME = 2
uint8 VALID_FULLY_RESOLVED = 4
uint8 VALID_MAG = 8
uint32 t_acc
int32 nano
uint8 fix_type
bool gnss_fix_ok
bool diff_soln
uint8 psm_state
bool head_veh_valid
uint8 carr_soln
uint8 num_sv
float32 lat
float32 lon
float32 height
float32 h_msl
float32 h_acc
float32 v_acc
float32 vel_n
float32 vel_e
float32 vel_d
float32 head_veh
float32 g_speed
float32 course
float32 p_dop
"""


types = get_types_from_msg(GNSS_MSG, 'blueboat_interfaces/msg/GNSSNavPvt')
typestore.register(types)
gnss_msg_type = 'blueboat_interfaces/msg/GNSSNavPvt'



def parse_gnss_message(rawdata):
    if len(rawdata) < 120:
        return None  # skip too small messages

    try:
        lat, lon = struct.unpack_from('<ff', rawdata, 56)
        course, = struct.unpack_from('<f', rawdata, 100)

    except struct.error:
        return None

    return {
        'lat': lat,
        'lon': lon,
        'course': course
    }


def gen_ma2_gnss_ned():
    # Define your reference position here (origin of NED frame)
    LAT0 = 63.0  # Example latitude
    LON0 = 10.0  # Example longitude
    transformer = Transformer.from_crs("epsg:4326", "epsg:32632", always_xy=True)  # Example: WGS84 -> UTM Zone 32N
    x0, y0 = transformer.transform(LON0, LAT0)
    t_pos_ori = []
    with Reader(ROSBAG_PATH) as reader:
        connections = [c for c in reader.connections if c.topic == GNSS_TOPIC]
        for connection, timestamp, rawdata in reader.messages(connections):
            msg = typestore.deserialize_cdr(rawdata, gnss_msg_type)

            #timestamp_msg = msg.header.stamp.sec * (10**9) + msg.header.stamp.nanosec
            timestamp_msg = timestamp
        

            lat = msg.lat
            lon = msg.lon
            heading_deg = msg.course

            x, y = transformer.transform(lon, lat)
            north = y - y0
            east = x - x0
            down = 0.0 

            pos = [north, east, down]
            heading_rad = np.deg2rad(heading_deg)

            r = Rotation.from_euler('z', -heading_rad)  # Negative for NED convention
            quat = r.as_quat()  # Returns (x, y, z, w)
            ori_quat = [quat[0], quat[1], quat[2], quat[3]]

            t_pos_ori.append([timestamp_msg, pos, ori_quat])
    # pos is here relative to piren, which is NED
    return t_pos_ori




def main():

    # Iniitalize

    fastsam_model_path = "weights/FastSAM-x.pt"
    yolo_model_path = "weights/yolo11n-seg.pt"
    rwps_config_path = "config/rwps_config.json"

    cam_params = {"cx": K[0,2], "cy": K[1,2], "fx": K[0,0], "fy":K[1,1], "b": baseline}
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))


    t_body_to_cam = np.zeros((3, 1), dtype=np.float32)
    R_body_to_cam = np.eye(3, dtype=np.float32)

    #yolo = YoloSeg(model_path=yolo_model_path)
    fastsam = FastSAMSeg(model_path=fastsam_model_path)
    rwps3d = RWPS(config_file=rwps_config_path)
    temporal_filtering = TemporalFiltering(K, N=5, t_imu_to_cam=t_body_to_cam, R_imu_to_cam=R_body_to_cam)
    stixels = Stixels(num_stixels=192, img_shape=(height, width), cam_params=cam_params, t_body_to_cam=t_body_to_cam, R_body_to_cam=R_body_to_cam)
    #optical_flow = OpticalFlow(cam_params=cam_params, stixel_width=10)

    rwps3d.set_camera_params(cam_params, P1)

    # Extracting data

    gen_svo = gen_svo_images()
    #lidar_data = gen_ma2_lidar_points()
    gnss_data_ned = gen_ma2_gnss_ned()
    #lidar_timestamps = np.array([entry[0] for entry in lidar_data])
    gnss_timestamps = np.array([entry[0] for entry in gnss_data_ned])

    curr_pose = np.zeros(7)
    curr_pose[3:] = [0, 0, 0, 1]  # Identity quaternion


    iterating = True
    timestamp = 0
    prev_timestamp = 0

    while iterating:
        try:
            next_svo = next(gen_svo)
        except StopIteration:
            iterating = False
            print('Reached end of SVO file')
            break

        prev_timestamp = timestamp
        timestamp, left_img, disparity_img, depth_img, = next_svo
        dt = (timestamp - prev_timestamp) / (10 ** 9)
        left_img = np.ascontiguousarray(left_img, dtype=np.uint8)
        disparity_img  = np.ascontiguousarray(disparity_img,  dtype=np.float32)
        depth_img      = np.ascontiguousarray(depth_img,      dtype=np.float32)
        depth_img[depth_img > 100] = np.nan
        
        #lidar_idx, lidar_timestamp = find_closest_timestamp(lidar_timestamps, timestamp)
        gnss_idx, gnss_timestamp = find_closest_timestamp(gnss_timestamps, timestamp)

        #xyz_proj = lidar_data[lidar_idx][1]
        #xyz_intensity = lidar_data[lidar_idx][2]
        #xyz_c = lidar_data[lidar_idx][3]
    
        prev_pose = curr_pose.copy()
        pos_ned = gnss_data_ned[gnss_idx][1]
        ori_quat = gnss_data_ned[gnss_idx][2]
        curr_pose = np.concatenate([pos_ned, ori_quat])




        # Processing

        start_time = time.time()

        # FusedWSS

        rwps_mask_3d, plane_params_3d, rwps_succeded = rwps3d.segment_water_plane_using_point_cloud(depth_img)
        #contour_mask, upper_contour_mask, water_mask = fastsam.get_all_countours_and_best_iou_mask_2(left_img, rwps_mask_3d)
        contour_mask, upper_contour_mask, water_mask = fastsam.get_all_countours_and_best_iou_mask(left_img, rwps_mask_3d)
        
        if not rwps_succeded:
            water_mask = get_water_mask_from_contour_mask(contour_mask)

        # Boat detection

        #boat_mask = yolo.get_boat_mask(left_img)

        # Temporal Filtering
        
        water_mask_filtered = temporal_filtering.get_filtered_frame_no_motion_compensation(water_mask)
        #water_mask_filtered = yolo.refine_water_mask(boat_mask, water_mask_filtered)

        # Stixel pipeline
        
        #xyz_proj, xyz_c = filter_point_cloud_by_image(xyz_proj, xyz_c, height, width)

        stixels.create_stixels_in_image(water_mask_filtered, disparity_img, depth_img, upper_contour_mask, boat_mask=None)

        #stixel_footprints = stixels.run_stixel_pipeline(
        #        left_img=left_img,
        #        water_mask=water_mask_filtered, 
        #        disparity_img=disparity_img, 
        #        depth_img=depth_img, 
        #        upper_contours=upper_contour_mask, 
        #        xyz_proj=xyz_proj, 
        #        xyz_c=xyz_c,
        #        pose_prev=prev_pose,
        #        pose_curr=curr_pose,
        #        dt=dt, 
        #        boat_mask=boat_mask,
        #)

        #optical_flow.plot_residual_flow(left_img, prev_pose, curr_pose)
        #optical_flow.plot_flow(left_img, dt)


        # Display
        end_time = time.time()
        runtime_ms = (end_time - start_time) * 1000
        print(f"Total time: {runtime_ms:.2f} ms")

        pink_color = [255, 0, 255]
        water_img_filtered = blend_image_with_mask(left_img, water_mask_filtered, pink_color, alpha1=1, alpha2=0.5)
        #water_img = blend_image_with_mask(left_img, water_mask, pink_color, alpha1=1, alpha2=0.5)
        stixel_img = stixels.overlay_stixels_on_image(left_img)


        #filtered_lidar_points, filtered_3d_points = stixels.get_filterd_lidar_points(xyz_proj, xyz_c)
        #lidar_stixel_img = merge_lidar_onto_image(image=stixel_img, lidar_points=filtered_lidar_points, lidar_3d_points=filtered_3d_points)
        #lidar_stixel_img = merge_lidar_onto_image(image=stixel_img, lidar_points=xyz_proj, lidar_3d_points=xyz_c)

        #cv2.imshow("left", left_img)
        nan_mask = np.isnan(depth_img) | (depth_img == 0) | np.isinf(depth_img)
        norm_depth = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored_depth = cv2.applyColorMap(norm_depth, cv2.COLORMAP_RAINBOW)
        colored_depth[nan_mask] = [0, 0, 0]  
        cv2.imshow("depth", colored_depth)


        nan_mask = np.isnan(disparity_img) | (disparity_img == 0) | np.isinf(disparity_img)
        norm_disparity = cv2.normalize(disparity_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored_disparity = cv2.applyColorMap(norm_disparity, cv2.COLORMAP_RAINBOW)
        colored_disparity[nan_mask] = [0, 0, 0]  
        cv2.imshow("disparity", colored_disparity)

        cv2.imshow("Filtered_mask", water_img_filtered)
        #cv2.imshow("Lidar stixel image", lidar_stixel_img)
        cv2.imshow("stixel image", stixel_img)
        #cv2.imshow("Contour mask", upper_contour_mask)


        #stixels.plot_stixel_footprints(stixel_footprints)
        #stixels.plot_projection_rays_and_associated_points(stixels.association_height.copy())
        #stixels.plot_projection_rays_and_associated_points(stixels.association_depth.copy())
        #stixels.plot_prev_and_curr_stixel_footprints(prev_stixel_footprints, stixel_footprints)

        cv2.waitKey(1)


        


if __name__ == "__main__":
    print("Starting microAmpere processing...")
    main()
