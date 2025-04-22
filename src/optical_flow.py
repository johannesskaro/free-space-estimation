import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


class OpticalFlow:

    def __init__(self, cam_params, stixel_width, flow_thr_px: float = 1.5):
        self.stixel_width = stixel_width
        self.cam_params = cam_params
        self.flow_thr_px  = flow_thr_px
        self.prev_gray    = None   # previous greyscale image for flow




    def reset(self):
        self.prev_gray = None

    def process_frame(
            self,
            stixel_list,
            frame_bgr: np.ndarray,
            pose_prev,
            pose_curr
        ) -> np.ndarray:

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        N = len(stixel_list)
        dynamic_mask = np.zeros(N, dtype=bool)

        if self.prev_gray is None:
            self.prev_gray = gray
            return dynamic_mask

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray,
            None,
            pyr_scale=0.5, levels=3,
            winsize=21, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0)

        u = flow[..., 0]                     # horizontal component (px)
        v = flow[..., 1]                    # vertical component (px)

        fx = self.cam_params['fx']          
        fy = self.cam_params['fy']          
        cx = self.cam_params['cx']          
        cy = self.cam_params['cy']    

        euler_prev = R.from_quat(pose_prev[3:]).as_euler('xyz', degrees=False)
        euler_curr = R.from_quat(pose_curr[3:]).as_euler('xyz', degrees=False)

        delta_roll = euler_curr[0] - euler_prev[0]
        delta_pitch = euler_curr[1] - euler_prev[1]
        delta_yaw = euler_curr[2] - euler_prev[2]

        for n, stixel in enumerate(stixel_list):
            u0 = n * self.stixel_width
            u1 = (n + 1) * self.stixel_width
            v0, v1 = stixel[n, 0], stixel[n, 1]

            u_c = (u0 + u1) // 2
            v_c = (v0 + v1) // 2

            u_ego = fx * delta_yaw - (u_c - cx) * delta_roll
            v_ego = fy * delta_roll - (v_c - cy) * delta_yaw + fy * delta_pitch
            
            u_med = np.median(u[v0:v1, u0:u1])
            v_med = np.median(v[v0:v1, u0:u1])

            flow_residual = np.hypot(u_med - u_ego,   v_med - v_ego) 

            if flow_residual > self.flow_thr_px:
                dynamic_mask[n] = True


        self.prev_gray = gray
        return dynamic_mask
    
    def plot_residual_flow(self, frame: np.ndarray, pose_prev, pose_curr, stride: int = 16):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            print("No previous frame to compute flow; call process_frame first.")
            self.prev_gray = gray
            return

        # compute dense flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3,
            winsize=21, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0)
        u_img = flow[..., 0]
        v_img = flow[..., 1]

        # compute pose deltas
        fx, fy = self.cam_params['fx'], self.cam_params['fy']
        cx, cy = self.cam_params['cx'], self.cam_params['cy']
        e_prev = R.from_quat(pose_prev[3:]).as_euler('xyz', degrees=False)
        e_curr = R.from_quat(pose_curr[3:]).as_euler('xyz', degrees=False)
        d_roll  = e_curr[0] - e_prev[0]
        d_pitch = e_curr[1] - e_prev[1]
        d_yaw   = e_curr[2] - e_prev[2]

        # prepare sampling grid
        h, w = gray.shape
        ys, xs = np.mgrid[0:h:stride, 0:w:stride]
        u_s = u_img[ys, xs]
        v_s = v_img[ys, xs]

        # predicted ego flow at sampled points
        u_ego = fx * d_yaw - (xs - cx) * d_roll
        v_ego = fy * d_roll - (ys - cy) * d_yaw + fy * d_pitch

        # residuals
        u_res = u_s - u_ego
        v_res = v_s - v_ego

        # plot
        disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 6))
        plt.imshow(disp)
        plt.quiver(xs, ys, u_res, v_res,
                   angles='xy', scale_units='xy', scale=1, color='r')
        plt.title("Residual Optical Flow (Measured - Predicted)")
        plt.axis('off')
        plt.show()

        self.prev_gray = gray
    


    def plot_flow(self, frame_bgr: np.ndarray, stride: int = 16):
        """
        Compute and plot the dense Farneback flow vectors overlaid on the image.
        
        Parameters
        ----------
        frame_bgr : np.ndarray
            Current frame in BGR.
        stride : int
            Sampling step for quiver arrows (pixels).
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            print("No previous frame to compute flow; call process_frame first.")
            self.prev_gray = gray
            return

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3,
            winsize=21, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0)
        u = flow[..., 0]
        v = flow[..., 1]

        # Prepare sampling grid
        h, w = gray.shape
        ys, xs = np.mgrid[0:h:stride, 0:w:stride]

        u_s = u[ys, xs]
        v_s = v[ys, xs]

        # Plot
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 6))
        plt.imshow(image_rgb)
        plt.quiver(xs, ys, u_s, v_s, color='r', angles='xy', scale_units='xy', scale=1)
        plt.title("Dense Optical Flow (Farneback)")
        plt.axis('off')
        plt.show()

        self.prev_gray = gray

