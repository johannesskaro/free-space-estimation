import numpy as np
from collections import deque
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import json

class TemporalFiltering:
    def __init__(self, cam_matrix, N, t_imu_to_cam, R_imu_to_cam):

        self.N = N
        self.cam_matrix = cam_matrix
        self.t_imu_to_cam = t_imu_to_cam
        self.R_imu_to_cam = R_imu_to_cam
        self.past_frames = deque(maxlen=self.N)
        self.past_poses = deque(maxlen=self.N)


    def add_frame(self, frame):
        self.past_frames.append(frame)

    def add_pose(self, pose):
        self.past_poses.append(pose)

    def get_rotation_matrix(self, ori_quat):
        r = R.from_quat(ori_quat)
        euler_angles = r.as_euler('zyx', degrees=False)
        rotation = R.from_euler('ZYX', euler_angles, degrees=False)
        rotation_matrix = rotation.as_matrix()
        return rotation_matrix


    def get_filtered_frame_no_motion_compensation(self, water_mask):
        mask_sum = np.sum(self.past_frames, axis=0)
        threshold = self.N/2
        thresholded_mask = (mask_sum > threshold).astype(np.uint8)
        smoothed_water_mask = np.logical_or(water_mask, thresholded_mask).astype(int)
        self.add_frame(water_mask)
        return smoothed_water_mask
    
    def compute_homography(self, R, t, n, d, K):
        """
        Computes the homography matrix H = K * (R - (t * n^T) / d) * K_inv
        """
        K_inv = np.linalg.inv(K)
        Rt = R - (t @ n.T) / d
        H = K @ Rt @ K_inv
        return H
    
    def get_ego_motion_compensated_frames(self, curr_pose, plane_normal=np.array([[0], [0], [1]]), plane_d=1):

        curr_pos = curr_pose[:3]
        curr_ori_quat = curr_pose[3:]

        past_compensated_frames = []
        R_curr = self.get_rotation_matrix(curr_ori_quat)

        for i, (past_frame, past_pose) in enumerate(zip(self.past_frames, self.past_poses)):
            past_pos = past_pose[:3]
            past_ori_quat = past_pose[3:]
            R_past = self.get_rotation_matrix(past_pos)

            R_rel_imu = R_curr @ R_past.T

            t_induced_imu = R_rel_imu @ self.t_imu_to_cam - self.t_imu_to_cam

            R_rel_cam = self.R_imu_to_cam @ R_rel_imu @ self.R_imu_to_cam.T
            t_rel_cam = self.R_imu_to_cam @ t_induced_imu

            H = self.compute_homography(R_rel_cam, t_rel_cam, plane_normal, plane_d, self.cam_matrix)

            warped_mask = cv2.warpPerspective(past_frame.astype(np.uint8), H, (past_frame.shape[1], past_frame.shape[0]), flags=cv2.INTER_NEAREST)
            past_compensated_frames.append(warped_mask)

        past_compensated_frames = np.array(past_compensated_frames)
        return past_compensated_frames
    
    def get_filtered_frame(self, water_mask, curr_pose, plane_normal, plane_d):

        past_compensated_frames = self.get_ego_motion_compensated_frames(curr_pose, plane_normal, plane_d)
        past_compensated_frames = np.sum(past_compensated_frames, axis=0)
        mask_sum = np.sum(self.past_frames, axis=0)
        threshold = self.N/2
        thresholded_mask = (mask_sum > threshold).astype(np.uint8)
        smoothed_water_mask = np.logical_or(water_mask, thresholded_mask).astype(int)
        self.add_frame(water_mask)
        self.add_pose(curr_pose)

        return smoothed_water_mask