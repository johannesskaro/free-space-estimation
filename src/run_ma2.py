from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from rosbags.typesys import Stores, get_typestore
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import pyzed.sl as sl
from stereo_svo import SVOCamera
from transforms import *
from utilities import find_closest_timestamp, get_water_mask_from_contour_mask, blend_image_with_mask
import time
from yolo import YoloSeg
from fastSAM import FastSAMSeg
from RWPS import RWPS
from temporal_filtering import TemporalFiltering
from stixels import Stixels

#Scen1 - Into tunnel
#SVO_FILE_PATH = r"C:\Users\johro\Documents\2023-07-11_Multi_ZED_Summer\ZED camera svo files\2023-07-11_11-30-51_28170706_HD1080_FPS15.svo"
#ROSBAG_NAME = "scen1"
#START_TIMESTAMP = 1689067892194593719 #+ 120000000000
#ma2_clap_timestamps = np.array([1689068801634572145, 1689068803035078922, 1689068804635190937, 1689068806436892969, 1689068809235474632]) 
#svo_clap_timestamps = np.array([1689068801796052729, 1689068803135787729, 1689068804743766729, 1689068806686255729, 1689068809298756729]) 


#Scen2_2 - Crossing
#SVO_FILE_PATH = r"C:\Users\johro\Documents\2023-07-11_Multi_ZED_Summer\ZED camera svo files\2023-07-11_11-52-01_28170706_HD1080_FPS15.svo"
#ROSBAG_NAME = "scen2_2"
#START_TIMESTAMP = 1689069183452515635 + 25000000000
#svo_clap_timestamps = np.array([1689069149608491571, 1689069151551222571,1689069153962745571,1689069155369492571, 1689069156977375571, 1689069158785868571]) 
#ma2_clap_timestamps = np.array([1689069149450584626, 1689069151450466830,1689069153851288364,1689069155051366874, 1689069156851162758, 1689069158651091124])


#Scen4_2 - Docking w. boats
SVO_FILE_PATH = r"C:\Users\johro\Documents\2023-07-11_Multi_ZED_Summer\ZED camera svo files\2023-07-11_12-20-43_28170706_HD1080_FPS15.svo" #right zed
#SVO_FILE_PATH = r"C:\Users\johro\Documents\2023-07-11_Multi_ZED_Summer\ZED camera svo files\2023-07-11_12-20-43_5256916_HD1080_FPS15.svo" #left zed
ROSBAG_NAME = "scen4_2"
START_TIMESTAMP = 1689070899731613030 #+ 14000000000
#START_TIMESTAMP = 1689070888907352002# Starting to see kayak
#START_TIMESTAMP = 1689070920831613030 #Docking
ma2_clap_timestamps = np.array([1689070864130009197, 1689070865931143443, 1689070867729428949, 1689070870332243623, 1689070872330384680])
svo_clap_timestamps = np.array([1689070864415441257, 1689070866090016257, 1689070867898886257, 1689070870444290257, 1689070872386914257]) 


#Scen5 - Docking with tube
#SVO_FILE_PATH = r"C:\Users\johro\Documents\2023-07-11_Multi_ZED_Summer\ZED camera svo files\2023-07-11_12-49-30_28170706_HD1080_FPS15.svo" #port side zed
#SVO_FILE_PATH = r"C:\Users\johro\Documents\2023-07-11_Multi_ZED_Summer\ZED camera svo files\2023-07-11_12-49-30_5256916_HD1080_FPS15.svo" #starboard side zed
#ROSBAG_NAME = "scen5"
#START_TIMESTAMP = 1689072610543382582 + 15300000000
#START_TIMESTAMP = 1689072601809970113
#START_TIMESTAMP = 1689072611943528062
#START_TIMESTAMP = 1689072633543528062
#ma2_clap_timestamps = np.array([(1689072578409069874 + 1689072578610974426) / 2,1689072580008516107,1689072581409199315,1689072582609241408,(1689072584209923187 + 1689072584409875973) / 2,1689072585209757240,])
#svo_clap_timestamps = np.array([1689072578584773729,1689072580192777729,1689072581532534729,(1689072582805224729 + 1689072582872153729) / 2,1689072584412922729,(1689072585350708729 + 1689072585417710729) / 2,])


#Scen6 - Docking with tube further away
#SVO_FILE_PATH = r"C:\Users\johro\Documents\2023-07-11_Multi_ZED_Summer\ZED camera svo files\2023-07-11_12-55-58_28170706_HD1080_FPS15.svo" #port side zed
#SVO_FILE_PATH = r"C:\Users\johro\Documents\2023-07-11_Multi_ZED_Summer\ZED camera svo files\2023-07-11_12-55-58_5256916_HD1080_FPS15.svo" # left zed
#ROSBAG_NAME = "scen6"
#START_TIMESTAMP = 1689073008428931880 #+ 2000000000  # Starting to see tube
#START_TIMESTAMP = 1689073018428931880 # tube almost passed
#START_TIMESTAMP = 1689073021428931880 + 1000000000 # tube passed
#ma2_clap_timestamps = np.array([1689072978427718986, 1689072980427686560, 1689072982230896164, 1689072984228220707])
#svo_clap_timestamps = np.array([1689072978666263269, 1689072980675916269, 1689072982484494269, 1689072984360142269])


diffs_s = (ma2_clap_timestamps - svo_clap_timestamps) / (10 ** 9)
GNSS_MINUS_ZED_TIME_NS = np.mean(ma2_clap_timestamps - svo_clap_timestamps)

ROSBAG_FOLDER = r"C:\Users\johro\Documents\2023-07-11_Multi_ZED_Summer\bags"
ROSBAG_PATH = f"{ROSBAG_FOLDER}/{ROSBAG_NAME}"
LIDAR_TOPIC = "/lidar_aft/points"
GNSS_TOPIC = "/senti_parser/SentiPose"


stereo_cam = SVOCamera(SVO_FILE_PATH)
stereo_cam.set_svo_position_timestamp(START_TIMESTAMP)
K, D = stereo_cam.get_left_parameters()
_, _, R, T = stereo_cam.get_right_parameters()
focal_length = K[0,0]
baseline = np.linalg.norm(T)
height, width = 1080, 1920

typestore = get_typestore(Stores.ROS2_FOXY)

def gen_svo_images():
    num_frames = 200 # 275
    curr_frame = 0
    while stereo_cam.grab() == sl.ERROR_CODE.SUCCESS and curr_frame < num_frames:
        image = stereo_cam.get_left_image(should_rectify=True)
        timestamp = stereo_cam.get_timestamp()
        disparity_img = stereo_cam.get_neural_disp()
        curr_frame += 1
        depth_img = stereo_cam.get_depth_image()

        yield timestamp + GNSS_MINUS_ZED_TIME_NS, image, disparity_img, depth_img


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
 

def gen_ma2_gnss_ned():
    t_pos_ori = []
    with Reader(ROSBAG_PATH) as reader:
        connections = [c for c in reader.connections if c.topic == GNSS_TOPIC]
        for connection, timestamp, rawdata in reader.messages(connections):
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            timestamp_msg = msg.header.stamp.sec * (10**9) + msg.header.stamp.nanosec

            pos_ros = msg.pose.position
            pos = np.array([pos_ros.x, pos_ros.y, pos_ros.z])
            ori_ros = msg.pose.orientation
            ori_quat = np.array([ori_ros.x, ori_ros.y, ori_ros.z, ori_ros.w])

            H = H_POINTS_PIREN_FROM_PIREN_ENU
            pos = H.dot(np.r_[pos, 1])[:3].T

            t_pos_ori.append([timestamp_msg, pos, ori_quat])
    # pos is here relative to piren, which is NED
    return t_pos_ori

def main():

    # Iniitalize

    fastsam_model_path = "weights/FastSAM-x.pt"
    yolo_model_path = "weights/yolo11n-seg.pt"
    rwps_config_path = "config/rwps_config.json"

    yolo = YoloSeg(model_path=yolo_model_path)
    fastsam = FastSAMSeg(model_path=fastsam_model_path)
    rwps3d = RWPS(config_file=rwps_config_path)
    temporal_filtering = TemporalFiltering(K, N=3, t_imu_to_cam=TRANS_FLOOR_TO_CAM, R_imu_to_cam=ROT_FLOOR_TO_LIDAR)
    stixels = Stixels(num_stixels=192, img_shape=(height, width))


    cam_params = {"cx": K[0,2], "cy": K[1,2], "fx": K[0,0], "fy":K[1,1], "b": baseline}
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    rwps3d.set_camera_params(cam_params, P1)


    # Extracting data
    gen_svo = gen_svo_images()
    lidar_data = gen_ma2_lidar_points()
    gnss_data_ned = gen_ma2_gnss_ned()
    lidar_timestamps = np.array([entry[0] for entry in lidar_data])
    gnss_timestamps = np.array([entry[0] for entry in gnss_data_ned])

    iterating = True

    while iterating:
        try:
            next_svo = next(gen_svo)
        except StopIteration:
            iterating = False
            print('Reached end of SVO file')
            break

        timestamp, left_img, disparity_img, depth_img, = next_svo
        lidar_idx, lidar_timestamp = find_closest_timestamp(lidar_timestamps, timestamp)
        gnss_idx, gnss_timestamp = find_closest_timestamp(gnss_timestamps, timestamp)

        xyz_proj = lidar_data[lidar_idx][1]
        xyz_intensity = lidar_data[lidar_idx][2]
        xyz_c = lidar_data[lidar_idx][3]

        pos_ned = gnss_data_ned[gnss_idx][1]
        ori_quat = gnss_data_ned[gnss_idx][2]
        curr_pose = np.concatenate([pos_ned, ori_quat])

        
        # Processing
        start_time = time.time()

        # FusedWSS
        rwps_mask_3d, plane_params_3d, rwps_succeded = rwps3d.segment_water_plane_using_point_cloud(depth_img)
        
        contour_mask, upper_contour_mask, water_mask = fastsam.get_all_countours_and_best_iou_mask(left_img, rwps_mask_3d)

        

        rwps_succeded = False
        if not rwps_succeded:
            water_mask = get_water_mask_from_contour_mask(contour_mask)
        
        # Temporal Filtering
        water_mask_filtered = temporal_filtering.get_filtered_frame_no_motion_compensation(water_mask)

        #boat_mask = yolo.get_boat_mask(left_img)
        #water_mask_filtered = yolo.refine_water_mask(boat_mask, water_mask_filtered)

        # Create Stixels
        
        #stixels.create_stixels(water_mask_filtered, disparity_img, depth_img, upper_contour_mask)
        free_space_boundary = stixels.get_free_space_boundary(water_mask_filtered)

        
        temp = stixels.create_segmentation_score_map(disparity_img, free_space_boundary, upper_contour_mask)
        




        # Display
        end_time = time.time()
        runtime_ms = (end_time - start_time) * 1000
        print(f"Processing time: {runtime_ms:.2f} ms")

        pink_color = [255, 0, 255]
        water_img_filtered = blend_image_with_mask(left_img, water_mask_filtered, pink_color, alpha1=1, alpha2=0.5)
        water_img = blend_image_with_mask(left_img, water_mask, pink_color, alpha1=1, alpha2=0.5)


        #cv2.imshow("left", left_img)
        cv2.imshow("Filtered_mask", water_img_filtered)
        cv2.waitKey(1)
        #time.sleep(0.1)
        



if __name__ == "__main__":
    main()
