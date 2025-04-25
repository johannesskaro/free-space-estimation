import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from transforms import TRANS_FLOOR_TO_LIDAR


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

        R_prev = R.from_quat(pose_prev[3:]).as_matrix()
        R_curr = R.from_quat(pose_curr[3:]).as_matrix()

        pos_b_prev = pose_prev[0:3]
        pos_b_curr = pose_curr[0:3]

        t_body_to_cam = np.array([-TRANS_FLOOR_TO_LIDAR[0], -TRANS_FLOOR_TO_LIDAR[1], TRANS_FLOOR_TO_LIDAR[2]])

        p_c_prev = pos_b_prev + R_prev.dot(t_body_to_cam)
        p_c_curr = pos_b_curr + R_curr.dot(t_body_to_cam)

        delta_pc = p_c_curr - p_c_prev

        Vx, Vy, Vz = R_curr.T.dot(delta_pc)

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

    def plot_residual_flow_per_stixel(self, frame: np.ndarray,
                                      pose_prev, pose_curr,
                                      stride=16):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            print("No previous frame to compute flow; call process_frame first.")
            self.prev_gray = gray
            return

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_img_gray, gray,
            None, pyr_scale=0.5, levels=3,
            winsize=21, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0
        )
        u_img = flow[..., 0]
        v_img = flow[..., 1]

        # 3) Unpack camera intrinsics
        fx, fy = self.cam_params['fx'], self.cam_params['fy']
        cx, cy = self.cam_params['cx'], self.cam_params['cy']

        # 4) Compute pose deltas
        e_prev = R.from_quat(pose_prev[3:]).as_euler('xyz', degrees=False)
        e_curr = R.from_quat(pose_curr[3:]).as_euler('xyz', degrees=False)
        d_roll  = e_curr[0] - e_prev[0]
        d_pitch = e_curr[1] - e_prev[1]
        d_yaw   = e_curr[2] - e_prev[2]

        # 5) Compute camera translation in camera frame
        t_body_to_cam = np.array([-TRANS_FLOOR_TO_LIDAR[0],
                                  -TRANS_FLOOR_TO_LIDAR[1],
                                   TRANS_FLOOR_TO_LIDAR[2]])
        R_prev = R.from_quat(pose_prev[3:]).as_matrix()
        R_curr = R.from_quat(pose_curr[3:]).as_matrix()
        p_c_prev = pose_prev[:3] + R_prev.dot(t_body_to_cam)
        p_c_curr = pose_curr[:3] + R_curr.dot(t_body_to_cam)
        delta_pc = p_c_curr - p_c_prev
        Vx, Vy, Vz = R_curr.T.dot(delta_pc)

        # 6) Gather residuals at each stixel center
        xs, ys, us_res, vs_res = [], [], [], []
        for n, stx in enumerate(self.height_base_list):
            # pixel bounds
            u0, u1 = n*self.stixel_width, (n+1)*self.stixel_width
            v0, v1 = stx[0], stx[1]
            # center
            u_c = 0.5*(u0 + u1)
            v_c = 0.5*(v0 + v1)

            # measured median flow
            band_u = u_img[v0:v1, u0:u1]
            band_v = v_img[v0:v1, u0:u1]
            if band_u.size == 0:
                continue
            u_med = np.median(band_u)
            v_med = np.median(band_v)

            # rotational prediction
            u_rot = -fx*d_yaw - (u_c - cx)*d_roll
            v_rot = fy*d_roll - (v_c - cy)*d_yaw + fy*d_pitch
            # translational prediction
            z = self.stixel_stereo_depths[n]
            if np.isfinite(z) and z > 0:
                u_trans = fx*(Vx/z) - (u_c - cx)*(Vz/z)
                v_trans = fy*(Vy/z) - (v_c - cy)*(Vz/z)
            else:
                u_trans = 0
                v_trans = 0

            # total predicted flow
            u_ego = u_rot + u_trans
            v_ego = v_rot + v_trans

            # residual
            us_res.append(u_med - u_ego)
            vs_res.append(v_med - v_ego)
            xs.append(u_c)
            ys.append(v_c)

        disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 6))
        plt.imshow(disp)
        plt.quiver(xs, ys, us_res, vs_res,
                   color='r', angles='xy',
                   scale_units='xy', scale=1)
        plt.title("Residual Optical Flow per Stixel")
        plt.axis('off')
        plt.show()

        # 8) Update prev_gray
        self.prev_img_gray = gray
    


    def plot_flow(self, frame_bgr: np.ndarray, dt, stride: int = 16):
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
        
        if dt <= 0:
            dt = 1

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3,
            winsize=21, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0)
        u = flow[..., 0] / dt
        v = flow[..., 1] / dt

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

