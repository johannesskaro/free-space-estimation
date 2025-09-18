from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
import cv2
import numpy as npq
import pyzed.sl as sl
from stereo_svo_sdk4 import SVOCamera
from transforms import *
import utilities as ut
import time
from yolo import YoloSeg
from fastSAM import FastSAMSeg
from RWPS import RWPS
from temporal_filtering import TemporalFiltering
from stixels import Stixels, get_free_space_boundary
from optical_flow import OpticalFlow
from utilities_map import plot_gnss_iteration_video, plot_gnss_iteration_video_local, plot_previous_gnss_iterations_local, plot_previous_gnss_iterations_global

#Scen1 - Into tunnel
#SVO_FILE_PATH = r"C:\Users\johro\Documents\2023-07-11_Multi_ZED_Summer\ZED camera svo files\2023-07-11_11-30-51_28170706_HD1080_FPS15.svo"
#ROSBAG_NAME = "scen1"
#START_TIMESTAMP = 1689067892194593719 + 120000000000
#ma2_clap_timestamps = np.array([1689068801634572145, 1689068803035078922, 1689068804635190937, 1689068806436892969, 1689068809235474632]) 
#svo_clap_timestamps = np.array([1689068801796052729, 1689068803135787729, 1689068804743766729, 1689068806686255729, 1689068809298756729]) 


#Scen2_2 - Crossing
#SVO_FILE_PATH = r"C:\Users\johro\Documents\2023-07-11_Multi_ZED_Summer\ZED camera svo files\2023-07-11_11-52-01_28170706_HD1080_FPS15.svo"
#ROSBAG_NAME = "scen2_2"
#START_TIMESTAMP = 1689069183452515635 + 30000000000
#svo_clap_timestamps = np.array([1689069149608491571, 1689069151551222571,1689069153962745571,1689069155369492571, 1689069156977375571, 1689069158785868571]) 
#ma2_clap_timestamps = np.array([1689069149450584626, 1689069151450466830,1689069153851288364,1689069155051366874, 1689069156851162758, 1689069158651091124])


#Scen4_2 - Docking w. boats
#SVO_FILE_PATH = r"C:\Users\johro\Documents\2023-07-11_Multi_ZED_Summer\ZED camera svo files\2023-07-11_12-20-43_28170706_HD1080_FPS15.svo" #right zed
#SVO_FILE_PATH = r"C:\Users\johro\Documents\2023-07-11_Multi_ZED_Summer\ZED camera svo files\2023-07-11_12-20-43_5256916_HD1080_FPS15.svo" #left zed
SVO_FILE_PATH = "/home/johro/datasets/2023-07-11_Multi_ZED_Summer/ZED camera svo files/2023-07-11_12-20-43_28170706_HD1080_FPS15.svo"
ROSBAG_NAME = "scen4_2"
START_TIMESTAMP = 1689070899731613030 + 20000000000
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
#SVO_FILE_PATH = "/home/johro/datasets/2023-07-11_Multi_ZED_Summer/ZED camera svo files/2023-07-11_12-55-58_28170706_HD1080_FPS15.svo" #port side zed
#ROSBAG_NAME = "scen6"
#START_TIMESTAMP = 1689073008428931880   #+ 4000000000  # Starting to see tube
#START_TIMESTAMP = 1689073018428931880 # tube almost passed
#START_TIMESTAMP = 1689073021428931880 + 1000000000 # tube passed
#ma2_clap_timestamps = np.array([1689072978427718986, 1689072980427686560, 1689072982230896164, 1689072984228220707])
#svo_clap_timestamps = np.array([1689072978666263269, 1689072980675916269, 1689072982484494269, 1689072984360142269])


diffs_s = (ma2_clap_timestamps - svo_clap_timestamps) / (10 ** 9)
GNSS_MINUS_ZED_TIME_NS = np.mean(ma2_clap_timestamps - svo_clap_timestamps)

#ROSBAG_FOLDER = r"C:\Users\johro\Documents\2023-07-11_Multi_ZED_Summer\bags"
ROSBAG_FOLDER = "/home/johro/datasets/2023-07-11_Multi_ZED_Summer/bags"
ROSBAG_PATH = f"{ROSBAG_FOLDER}/{ROSBAG_NAME}"
LIDAR_TOPIC = "/lidar_aft/points"
GNSS_TOPIC = "/senti_parser/SentiPose"


stereo_cam = SVOCamera(SVO_FILE_PATH)
stereo_cam.set_svo_position_timestamp(START_TIMESTAMP)
K, D = stereo_cam.get_left_parameters()
R = stereo_cam.R
T = stereo_cam.T
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


save_video = False
save_map_video = False
save_data = False

#src_dir = r"C:\Users\johro\Documents\BB-Perception\free-space-estimation"
src_dir = "projects/free-space-estimation"
FPS = 5.0

if save_video:
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")  # You can also use 'MP4V' for .mp4 format
    out = cv2.VideoWriter(
        f"{src_dir}/results/video_stixel_world_4.mp4",
        fourcc,
        FPS,
        (width, height),
    )

if save_map_video:
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")  # You can also use 'MP4V' for .mp4 format
    out_map = cv2.VideoWriter(
        f"{src_dir}/results/{ROSBAG_NAME}_video_BEV_stixel_world_4.mp4",
        fourcc,
        FPS,
        (height, height),
    )



def main():

    # Iniitalize

    fastsam_model_path = "weights/FastSAM-x.pt"
    yolo_model_path = "weights/yolo11n-seg.pt"
    rwps_config_path = "config/rwps_config.json"

    cam_params = {"cx": K[0,2], "cy": K[1,2], "fx": K[0,0], "fy":K[1,1], "b": baseline}
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))

    #t_body_to_cam = np.array([-TRANS_FLOOR_TO_LIDAR[0], -TRANS_FLOOR_TO_LIDAR[1], TRANS_FLOOR_TO_LIDAR[2]])
    t_body_to_cam = np.array([- TRANS_FLOOR_TO_LIDAR[0], - TRANS_FLOOR_TO_LIDAR[1]])

    #rot_yaw_180 = Rot.from_euler('z', np.pi).as_matrix()
    #R_body_to_cam = rot_yaw_180 @ ROT_FLOOR_TO_LIDAR 
    R_body_to_cam = np.eye(2)

    yolo = YoloSeg(model_path=yolo_model_path)
    fastsam = FastSAMSeg(model_path=fastsam_model_path)
    rwps3d = RWPS(config_file=rwps_config_path)
    temporal_filtering = TemporalFiltering(K, N=3, t_imu_to_cam=t_body_to_cam, R_imu_to_cam=R_body_to_cam)
    stixels = Stixels(num_stixels=192, img_shape=(height, width), cam_params=cam_params, t_body_to_cam=t_body_to_cam, R_body_to_cam=R_body_to_cam)
    optical_flow = OpticalFlow(cam_params=cam_params, stixel_width=10)

    rwps3d.set_camera_params(cam_params, P1)

    # Extracting data

    gen_svo = gen_svo_images()
    lidar_data = gen_ma2_lidar_points()
    gnss_data_ned = gen_ma2_gnss_ned()
    lidar_timestamps = np.array([entry[0] for entry in lidar_data])
    gnss_timestamps = np.array([entry[0] for entry in gnss_data_ned])

    curr_pose = np.zeros(7)
    curr_pose[3:] = [0, 0, 0, 1]  # Identity quaternion

    gnss_pos_list = []
    gnss_ori_list = []
    stixel_points_list = []

    iterating = True
    curr_frame = 0
    timestamp = 0
    prev_timestamp = 0

    while iterating:
        try:
            next_svo = next(gen_svo)
        except StopIteration:
            iterating = False
            print('Reached end of SVO file')
            break
        
        curr_frame += 1
        prev_timestamp = timestamp
        timestamp, left_img, disparity_img, depth_img, = next_svo
        dt = (timestamp - prev_timestamp) / (10 ** 9)
        left_img = np.ascontiguousarray(left_img, dtype=np.uint8)
        disparity_img  = np.ascontiguousarray(disparity_img,  dtype=np.float32)
        depth_img      = np.ascontiguousarray(depth_img,      dtype=np.float32)
        depth_img[depth_img > 100] = np.nan


        lidar_idx, lidar_timestamp = ut.find_closest_timestamp(lidar_timestamps, timestamp)
        gnss_idx, gnss_timestamp = ut.find_closest_timestamp(gnss_timestamps, timestamp)

        xyz_proj = lidar_data[lidar_idx][1]
        xyz_intensity = lidar_data[lidar_idx][2]
        xyz_c = lidar_data[lidar_idx][3]

        prev_pose = curr_pose.copy()
        pos_ned = gnss_data_ned[gnss_idx][1]
        ori_quat = gnss_data_ned[gnss_idx][2]
        curr_pose = np.concatenate([pos_ned, ori_quat])
        gnss_pos_list.append(pos_ned)
        gnss_ori_list.append(ori_quat)
        
        # Processing

        start_time = time.time()

        # FusedWSS

        rwps_mask_3d, plane_params_3d, rwps_succeded = rwps3d.segment_water_plane_using_point_cloud(depth_img)
        
        
        #rwps3d.plot_3d_pcd(left_img=left_img, depth_img=depth_img)

        
        #contour_mask, upper_contour_mask, water_mask = fastsam.get_all_countours_and_best_iou_mask_2(left_img, rwps_mask_3d)
        contour_mask, upper_contour_mask, water_mask = fastsam.get_all_countours_and_best_iou_mask(left_img, rwps_mask_3d)

        
        
        if not rwps_succeded:
            water_mask = ut.get_water_mask_from_contour_mask(contour_mask)

        # Boat detection

        boat_mask = yolo.get_boat_mask(left_img)

        # Temporal Filtering
        
        water_mask_filtered = temporal_filtering.get_filtered_frame_no_motion_compensation(water_mask)
        water_mask_refined = yolo.refine_water_mask(boat_mask, water_mask_filtered)

        # Stixel pipeline
        
        xyz_proj, xyz_c = ut.filter_point_cloud_by_image(xyz_proj, xyz_c, height, width)

        stixel_footprints = stixels.run_stixel_pipeline(
                left_img=left_img,
                water_mask=water_mask_refined, 
                disparity_img=disparity_img, 
                depth_img=depth_img, 
                upper_contours=upper_contour_mask, 
                xyz_proj=xyz_proj, 
                xyz_c=xyz_c,
                pose_prev=prev_pose,
                pose_curr=curr_pose,
                dt=dt, 
                boat_mask=boat_mask,
        )
        


        # Display
        end_time = time.time()
        runtime_ms = (end_time - start_time) * 1000
        print(f"Total time: {runtime_ms:.2f} ms")

        pink_color = [255, 0, 255]
        #left_img_cr = left_img.copy() // 2 + 128
        #water_img_filtered = ut.blend_image_with_mask(left_img, water_mask_filtered, pink_color, alpha1=1, alpha2=0.5)
        water_img_refined = ut.blend_image_with_mask(left_img, water_mask_refined, pink_color, alpha1=1, alpha2=0.5)
        #water_img = blend_image_with_mask(left_img, water_mask, pink_color, alpha1=1, alpha2=0.5)
        stixel_img = stixels.overlay_stixels_on_image(left_img)
        

        filtered_lidar_points, filtered_3d_points = stixels.get_filterd_lidar_points(xyz_proj, xyz_c)
        lidar_stixel_img = ut.merge_lidar_onto_image(image=stixel_img, lidar_points=filtered_lidar_points, lidar_3d_points=filtered_3d_points, alpha=1)
        #lidar_stixel_img = merge_lidar_onto_image(image=stixel_img, lidar_points=xyz_proj, lidar_3d_points=xyz_c)
        #lidar_img = ut.merge_lidar_onto_image(image=left_img, lidar_points=xyz_proj, lidar_3d_points=None, alpha=1)

        rwps_img = ut.blend_image_with_mask(left_img, rwps_mask_3d, pink_color, alpha1=1, alpha2=0.5)
        cv2.imshow("RWPS mask", rwps_img)
        #cv2.imshow("left", left_img)
        #cv2.imshow("Filtered_mask", water_img_filtered)
        cv2.imshow("Refined_mask", water_img_refined)
        cv2.imshow("Lidar stixel image", lidar_stixel_img)
        #cv2.imshow("stixel image", stixel_img)
        #cv2.imshow("Lidar image", lidar_img)
        #cv2.imshow("Contour mask", upper_contour_mask)
        #cv2.imshow("contour mask", contour_mask)
        #cv2.imshow("upper contour mask", upper_contour_mask)
        #cv2.imshow("Boat mask", boat_mask)

        #nan_mask = np.isnan(depth_img) | (depth_img == 0) | np.isinf(depth_img)
        #depth_img[nan_mask] = 0
        #norm_depth = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        #colored_depth = cv2.applyColorMap(norm_depth, cv2.COLORMAP_RAINBOW)
        #colored_depth[nan_mask] = [0, 0, 0]  
        #cv2.imshow("depth", colored_depth)

        #nan_mask = np.isnan(disparity_img) | (disparity_img == 0) | np.isinf(disparity_img)
        #norm_disparity = cv2.normalize(disparity_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        #norm_disparity[nan_mask] = 0
        #colored_disparity = cv2.applyColorMap(norm_disparity, cv2.COLORMAP_RAINBOW)
        #colored_disparity[nan_mask] = [0, 0, 0]
        #cv2.imshow("disparity", norm_disparity)

        #free_space_boundary = get_free_space_boundary(water_mask_refined)
        #grad_y = stixels.get_horizontal_disp_edges(disparity_img)
        #cv2.imshow("grad_y_before", grad_y*255)
        #grad_y = ut.filter_mask_by_boundary(grad_y, free_space_boundary, offset=5)
        #grad_y = ut.get_bottommost_line(grad_y, thickness=10)
        #grad_y_img = ut.blend_image_with_mask(left_img, grad_y, [0, 255, 0], alpha1=1, alpha2=1)
        #cv2.imshow("grad_y_img", grad_y_img)

        #upper_contours = ut.filter_mask_by_boundary(upper_contour_mask, free_space_boundary, offset=5)
        #upper_contours = ut.get_bottommost_line(upper_contours, thickness=10)
        #upper_contours_img = ut.blend_image_with_mask(left_img, upper_contours, [0, 0, 255], alpha1=1, alpha2=1)
        #cv2.imshow("upper_contours_img", upper_contours_img)


        #stixels.plot_stixel_footprints(stixel_footprints)
        #stixels.plot_projection_rays_and_associated_points(stixels.association_height.copy())
        #stixels.plot_projection_rays_and_associated_points(stixels.association_depth.copy())
        #stixels.plot_prev_and_curr_stixel_footprints(prev_stixel_footprints, stixel_footprints)

        #optical_flow.plot_flow(left_img, dt=0)

        #stixel_points_list.append(stixel_footprints)
        #plot_previous_gnss_iterations_local(gnss_pos_list, gnss_ori_list, stixel_points_list)

        cv2.waitKey(1)


        if save_video:
            out.write(lidar_stixel_img)
        if save_map_video:
            map_image = plot_gnss_iteration_video(curr_pose, stixel_footprints, 
                                                  stixels.dynamic_stixel_list.copy(), 
                                                  stixels.stixel_validity.copy(), 
                                                  stixels.using_prop_depth.copy(), 
                                                  stixels.stixel_fused_depths_var.copy())

            #map_image = plot_gnss_iteration_video_local(curr_pose, stixel_footprints, 
            #                                             stixels.dynamic_stixel_list.copy(), 
            #                                            stixels.stixel_validity.copy(), 
             #                                            stixels.using_prop_depth.copy())
            out_map.write(map_image)
        if save_data:
            ut.write_coordinates_to_file("files/perception_data_scen4_2_w_gnss.jsonl", 
                                         frame=curr_frame, 
                                         coordinates=stixel_footprints, 
                                         validity=stixels.stixel_validity.copy(), 
                                         dynamic=stixels.dynamic_stixel_list.copy(),
                                         depth_uncertainty=stixels.stixel_fused_depths_var.copy(),
                                         curr_pose=curr_pose,
            )
    
    if save_video:
        out.release()
    if save_map_video:
        out_map.release()
        



if __name__ == "__main__":
    main()
