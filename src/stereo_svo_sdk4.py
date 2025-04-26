import os
import pickle
import pyzed.sl as sl
import cv2
import numpy as np


class SVOCamera:
    def __init__(self, svo_file_path) -> None:
        cam_input_type = sl.InputType()
        cam_input_type.set_from_svo_file(svo_file_path)
        init_params = sl.InitParameters(input_t=cam_input_type, svo_real_time_mode=False)
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS #sl.DEPTH_MODE.NEURAL
        init_params.depth_minimum_distance = 0
        init_params.depth_maximum_distance = 60 #100
        #init_params.camera_disable_self_calib = True
        self.cam =  sl.Camera()
        err = self.cam.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED camera open failed with error: {err}")
        #assert self.cam.open(init_params) == sl.ERROR_CODE.SUCCESS

        self.cam.get_camera_information().camera_configuration.fps

        self.runtime = sl.RuntimeParameters()
        #self.runtime.sensing_mode = sl.SENSING_MODE.STANDARD
        #self.runtime.sensing_mode = sl.RuntimeParameters.enable_fill_mode
        self.left_image = sl.Mat()
        self.right_image = sl.Mat()
        self.point_cloud = sl.Mat()
        self.disparity = sl.Mat()
        self.depth = sl.Mat()

        # Short baseline rectification
        cam_resolution = self.cam.get_camera_information().camera_configuration.resolution
        self.image_shape = (cam_resolution.width, cam_resolution.height)
        left_K, left_D = self.get_left_parameters()
        right_K, right_D = left_K.copy(), left_D.copy()
        baseline = self.cam.get_camera_information().camera_configuration.calibration_parameters.get_camera_baseline()
        self.R = np.eye(3)
        self.T = np.array([-baseline, 0, 0])
        R1, R2, self.left_P, self.right_P, self.Q, roi1, roi2 = cv2.stereoRectify(left_K, left_D, right_K, right_D, self.image_shape, self.R, self.T, alpha=0) # alpha = 0 is zoomed to only valid pixels
        self.map_left_x, self.map_left_y = cv2.initUndistortRectifyMap(left_K, left_D, R1, self.left_P, self.image_shape, cv2.CV_32FC1)
        self.map_right_x, self.map_right_y = cv2.initUndistortRectifyMap(right_K, right_D, R2, self.right_P, self.image_shape, cv2.CV_32FC1)

        # For short baseline disparity calculations
        num_disparities = int(self.image_shape[0]*0.04)
        block_size = 4
        self.stereo_matcher = cv2.StereoSGBM_create(
            minDisparity = 1,
            numDisparities = num_disparities,
            blockSize = block_size,
            P1 = 8*3*block_size*block_size,
            P2 = 32*3*block_size*block_size,
            disp12MaxDiff = 0,
            uniquenessRatio = 0,
            speckleWindowSize = 200,
            speckleRange = 1,
            mode = cv2.STEREO_SGBM_MODE_SGBM
        )
    
    @property
    def length(self):
        return self.cam.get_svo_number_of_frames()
    
    def set_svo_position(self, n_frames):
        assert n_frames < self.length
        self.cam.set_svo_position(n_frames)
    
    def grab(self):
        err = self.cam.grab(self.runtime)
        return err
    
    def get_left_image(self, should_rectify = False):
        if should_rectify:
            self.cam.retrieve_image(self.left_image, sl.VIEW.LEFT)
        else:
            self.cam.retrieve_image(self.left_image, sl.VIEW.LEFT_UNRECTIFIED)
        image_bgr = self.left_image.get_data()[:,:,:3]
        return np.ascontiguousarray(image_bgr)
    
    def get_right_image(self, should_rectify = False):
        if should_rectify:
            self.cam.retrieve_image(self.right_image, sl.VIEW.RIGHT)
        else:
            self.cam.retrieve_image(self.right_image, sl.VIEW.RIGHT_UNRECTIFIED)
        image_bgr = self.right_image.get_data()[:,:,:3]
        return np.ascontiguousarray(image_bgr)

    def get_timestamp(self):
        return self.cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).data_ns
        # self.cam.retrieve_image(self.left_image, sl.VIEW.LEFT_UNRECTIFIED)
        # return self.left_image.timestamp.data_ns

    def close(self):
        self.cam.close()
    
    def get_left_parameters(self, should_rectify_zed=False, should_rectify_cv2=False):
        if should_rectify_cv2:
            D = np.zeros((1,5), dtype=np.float32)
            K = self.left_P[:3,:3]
        else:
            cp = self.get_calibration_parameters_(should_rectify_zed)
            K, D = self.get_cam_K_D_(cp)
        return K, D

    def get_calibration_parameters_(self, should_rectify_zed=False):
        cp_all = self.cam.get_camera_information().camera_configuration.calibration_parameters
        return cp_all if should_rectify_zed else cp_all.left_cam
    
    
    def get_cam_K_D_(self, camera_parameters):
        cp = camera_parameters
        K = np.array([
            [cp.fx, 0, cp.cx],
            [0, cp.fy, cp.cy],
            [0, 0, 1]
        ])
        # Distortion from ZED is is [k1, k2, p1, p2, k3], checked using the ZED_Explorer
        # Different things on the website: see here: https://www.stereolabs.com/docs/video/camera-calibration/
        # But something else here: https://www.stereolabs.com/docs/opencv/calibration/
        D = cp.disto
        # Distortion in OpenCV is [k1 k2 p1 p2 (k3 k4 k5 k6 s1 s2 s3 s4 tx ty)]
        D = np.array([D[0], D[1], D[2], D[3], D[4]])
        return K, D
    
    def get_cam_R_T_(self, camera_parameters):
        # It is not clear if it should be one Rodriguez vector or several. 
        # Also, best results are when returning just R and T, but the baseline is positive here
        # I would assume it should be negative, similar to the results from OpenCV calibration
        # We keep the best results although we cannot explain them 
        # The baseline should be negative. Otherwise, the Q matrix from rectify 
        # gets the wrong values and point cloud is wrong
        cp = camera_parameters
        Rx, _ = cv2.Rodrigues(np.array([cp.R[0], 0, 0]))
        Ry, _ = cv2.Rodrigues(np.array([0, cp.R[1], 0]))
        Rz, _ = cv2.Rodrigues(np.array([0, 0, cp.R[2]]))
        R = Rx @ Ry @ Rz
        T = cp.T
        
        return R, -T
        # return R.T, -R.T @ T
    
    def get_zed_rectified_Q(self):
        # Create own Q matrix:
        # OpenCV Q matrix looks like this:
        # See here: https://answers.opencv.org/question/187734/derivation-for-perspective-transformation-matrix-q/
        #  1   0   0       -cx
        #  0   1   0       -cy
        #  0   0   0       f(px)
        #  0   0   -1/Tx   ~0
        K, D = self.get_left_parameters(should_rectify_zed=True, should_rectify_cv2=False)
        _, _, R, T = self.get_right_parameters(should_rectify_zed=True, should_rectify_cv2=False)
        # R should here be identity
        # T should be trivial
        Q = np.array([
            [1, 0, 0, -K[0, 2]],
            [0, 1, 0, -K[1, 2]],
            [0, 0, 0, K[0, 0]],
            [0, 0, - 1 / T[0], 0]
        ])
        # assert np.allclose(Q, self.Q)
        return Q
    
    def get_neural_numpy_pointcloud(self):
        self.cam.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)

        # Get numpy pointcloud
        point_cloud_np = self.point_cloud.get_data()
        point_cloud_np = point_cloud_np.reshape(point_cloud_np.shape[0]*point_cloud_np.shape[1],point_cloud_np.shape[2])
        # Seems like if some value is inf or nan, all are inf or nan. Except for nan color, which is vlid to be a color. Do not remove these. 
        # Improved from Tryms suggesstions
        valid_mask = ~np.all(~np.isfinite(point_cloud_np), axis=1)
        point_cloud_valid = point_cloud_np[valid_mask]
        pc_xyz = point_cloud_valid[:, :3]
        pc_rgba = np.frombuffer(point_cloud_valid[:, 3].tobytes(), dtype=np.uint8).reshape(point_cloud_valid.shape[0], 4)
        pc_rgb = pc_rgba[:, :3] / 255

        return pc_xyz, pc_rgb
    
    def get_neural_pos_image(self):
        self.cam.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)
        pos_image_xyzrgba = self.point_cloud.get_data()
        valid_mask = ~np.all(~np.isfinite(pos_image_xyzrgba), axis=2)
        return pos_image_xyzrgba, valid_mask
    
    def get_neural_disp(self):
        self.cam.retrieve_measure(self.disparity, sl.MEASURE.DISPARITY)
        disp_negative_infs = self.disparity.get_data()
        disp_negative = disp_negative_infs.copy()
        disp_negative[~np.isfinite(disp_negative)] = 0
        disp = -disp_negative
        return disp
    
    def get_depth_image(self):
        self.cam.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
        return np.nan_to_num(self.depth.get_data(), nan=0)
    
    def set_svo_position_timestamp(self, timestamp):
        # self.set_svo_position(0)
        # assert self.grab() == sl.ERROR_CODE.SUCCESS
        # t0 = self.get_timestamp()
        n_frames_end = self.length - 1
        # int(np.ceil((timestamp - t0) / (10**9) * self.fps))

        n_frames = self.find_n_frames(timestamp, 0, n_frames_end)

        self.set_svo_position(n_frames)

    def find_n_frames(self, timestamp, n_frames_start, n_frames_end):
        self.set_svo_position(n_frames_start)
        assert self.grab() == sl.ERROR_CODE.SUCCESS
        t_start = self.get_timestamp()
        self.set_svo_position(n_frames_end)
        assert self.grab() == sl.ERROR_CODE.SUCCESS
        t_end = self.get_timestamp()

        assert t_start < timestamp < t_end

        if n_frames_end - n_frames_start == 1:
            if t_end - timestamp < timestamp - t_start:
                return n_frames_end
            else:
                return n_frames_start
        
        n_frames_mid = int((n_frames_end - n_frames_start)/2 + n_frames_start)
        self.set_svo_position(n_frames_mid)
        assert self.grab() == sl.ERROR_CODE.SUCCESS
        t_mid = self.get_timestamp()
        if t_mid == timestamp:
            return n_frames_mid
        elif t_mid < timestamp:
            return self.find_n_frames(timestamp, n_frames_mid, n_frames_end)
        else:
            return self.find_n_frames(timestamp, n_frames_start, n_frames_mid)
    
    def get_rectified_images(self):
        left_image = self.get_left_image(should_rectify=False)
        right_image = self.get_right_image(should_rectify=False)

        left_image_rectified = cv2.remap(left_image, self.map_left_x, self.map_left_y, cv2.INTER_LINEAR)
        right_image_rectified = cv2.remap(right_image, self.map_right_x, self.map_right_y, cv2.INTER_LINEAR)

        return left_image_rectified, right_image_rectified
    
    def calc_disparity_(self, left_image, right_image, stereo_matcher):
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

        left = cv2.equalizeHist(left_gray)
        right = cv2.equalizeHist(right_gray)

        disparity_sgbm = stereo_matcher.compute(left, right)
        disparity = disparity_sgbm.astype(np.float32) / 16.0

        disparity[disparity > disparity.max()*0.7] = 0

        return disparity
    
    def get_disparity(self):
        left_image, right_image = self.get_rectified_images()
        disparity = self.calc_disparity_(left_image, right_image, self.stereo_matcher)
        return disparity

    
    def get_pc(self):
        left_image, right_image = self.get_rectified_images()
        disparity = self.get_disparity()

        point_image = cv2.reprojectImageTo3D(disparity, self.Q, handleMissingValues=False)

        valid_indeces = np.logical_and(np.all(np.isfinite(point_image), axis=2), point_image[:,:,2] < 150)
        pc_xyz = point_image[valid_indeces]
        pc_bgr = left_image[valid_indeces] / 255
        pc_rgb = np.fliplr(pc_bgr)

        return pc_xyz, pc_rgb

    def project_into_image(self, xyz, should_rectify_zed=False, should_rectify_cv2=False):
        K, D = self.get_left_parameters(should_rectify_zed, should_rectify_cv2)
        image_shape_xy = self.image_shape
        image_points_inside, image_points_xyz = self.project_into_image_with_K_D(xyz, K, D, image_shape_xy)
        return image_points_inside, image_points_xyz
    
    @staticmethod
    def project_into_image_with_K_D(xyz, K, D, image_shape_xy):
        rvec = np.zeros((1,3), dtype=np.float32)
        tvec = np.zeros((1,3), dtype=np.float32)
        image_points, _ = cv2.projectPoints(xyz, rvec, tvec, K, D)
        image_points_squeezed = np.squeeze(image_points, axis=1)
        image_mask = (xyz[:,2] > 0) & (image_points_squeezed[:,0] > 0) & (image_points_squeezed[:,1] > 0) & (image_points_squeezed[:,0] < image_shape_xy[0]) & (image_points_squeezed[:,1] < image_shape_xy[1])
        image_points_inside = image_points_squeezed[image_mask]
        image_points_xyz = xyz[image_mask]
        return image_points_inside, image_points_xyz

class StereoSVOCamera:
    def __init__(self, left_svo_file_path, right_svo_file_path, nanos_to_add_to_left) -> None:
        self.left_cam = SVOCamera(left_svo_file_path)
        self.right_cam = SVOCamera(right_svo_file_path)

        self.fps = self.left_cam.fps
        assert self.fps == self.right_cam.fps

        # Find corresponding frames
        self.nanos_to_add_to_left = nanos_to_add_to_left
        self.n_frames_left_before_right = self.get_n_frames_left_before_right()
        self.set_svo_position(0)
    
    def get_n_frames_left_before_right(self):
        left_time_ns = self.left_cam.get_timestamp()
        right_time_ns = self.right_cam.get_timestamp()
        left_time_ns += self.nanos_to_add_to_left
        n_frames_left_before_right = (right_time_ns - left_time_ns)*(10**(-9)) * self.fps
        n_frames_left_before_right = round(n_frames_left_before_right)
        return n_frames_left_before_right
    
    def set_svo_position(self, n_frames):
        if self.n_frames_left_before_right >= 0:
            self.left_cam.set_svo_position(self.n_frames_left_before_right + n_frames)
            self.right_cam.set_svo_position(n_frames)
        else:
            self.left_cam.set_svo_position(n_frames)
            self.right_cam.set_svo_position(n_frames - self.n_frames_left_before_right)
    
    def get_timestamp(self):
        return self.left_cam.get_timestamp()
    
    def grab(self):
        err = self.left_cam.grab()
        if err != sl.ERROR_CODE.SUCCESS:
            return err
        err = self.right_cam.grab()
        if err != sl.ERROR_CODE.SUCCESS:
            return err
        
        n_frames_left_before_right = self.get_n_frames_left_before_right()
        n = 0
        while n_frames_left_before_right != 0:
            n += 1
            if n_frames_left_before_right < 0:
                err = self.right_cam.grab()
                if err != sl.ERROR_CODE.SUCCESS:
                    return err
            elif n_frames_left_before_right > 0:
                err = self.left_cam.grab()
                if err != sl.ERROR_CODE.SUCCESS:
                    return err
            n_frames_left_before_right = self.get_n_frames_left_before_right()

        if n != 0:
            print(f"Lost {n} frames")

        return err

    def close(self):
        self.left_cam.close()
        self.right_cam.close()

# # Calibrated stereo camera
# CALIBRATED_STARBOARD_SVO_FILE_PATH = "/media/nicholas/T7 Shield/2023-05-10 MA2 ZED/Starboard ZED/HD1080_SN28170706_12-05-52.svo"
# CALIBRATED_PORT_SVO_FILE_PATH = "/media/nicholas/T7 Shield/2023-05-10 MA2 ZED/Port ZED/HD1080_SN5256916_12-05-53.svo"
# CALIBRATED_NANOS_ADD_TO_PORT = 1683713235330602903 - 1683713235548397738

# stereo_R = np.array([
#     [ 9.99950818e-01, -4.89968263e-04,  9.90565319e-03],
#     [ 2.38056752e-04,  9.99676926e-01,  2.54162868e-02],
#     [-9.91490611e-03, -2.54126787e-02,  9.99627876e-01],
# ])

# stereo_T = np.array([
#     [-1.87686526], 
#     [-0.00353127], 
#     [ 0.01586476],
# ])

# TRACKING_RESULTS_FOLDER = "/media/nicholas/T7 Shield/2023-06-26 Elfryd tracking results data"

class CalibratedStereoSVOCamera(StereoSVOCamera):
    def __init__(self, left_svo_file_path, right_svo_file_path, nanos_to_add_to_left, R_wide, t_wide) -> None:
        super().__init__(left_svo_file_path, right_svo_file_path, nanos_to_add_to_left)
        self.R = R_wide
        self.t = t_wide

        cam_resolution = self.left_cam.cam.get_camera_information().camera_resolution
        image_shape = (cam_resolution.width, cam_resolution.height)

        # Wide baseline rectification
        left_leftcam_K, left_leftcam_D = self.left_cam.get_left_parameters()
        left_rightcam_K, left_rightcam_D = self.right_cam.get_left_parameters()
        R1, R2, self.left_P, self.right_P, self.Q_wide, roi1_wide, roi2_wide = cv2.stereoRectify(left_leftcam_K, left_leftcam_D, left_rightcam_K, left_rightcam_D, image_shape, R_wide, t_wide, alpha=0) # alpha = 0 is zoomed to only valid pixels
        self.map_wide_left_leftcam_x, self.map_wide_left_leftcam_y = cv2.initUndistortRectifyMap(left_leftcam_K, left_leftcam_D, R1, self.left_P, image_shape, cv2.CV_32FC1)
        self.map_wide_left_rightcam_x, self.map_wide_left_rightcam_y = cv2.initUndistortRectifyMap(left_rightcam_K, left_rightcam_D, R2, self.right_P, image_shape, cv2.CV_32FC1)

        K_leftcam_rectified, D_leftcam_rectified = self.left_cam.get_left_parameters(should_rectify_zed=True, should_rectify_cv2=False)
        K_rightcam_rectified, D_rightcam_rectified = self.right_cam.get_left_parameters(should_rectify_zed=True, should_rectify_cv2=False)
        self.rectifier_wide_from_rectified_short_zed = ImageRectifier(R_wide, t_wide, K_leftcam_rectified, D_leftcam_rectified, K_rightcam_rectified, D_rightcam_rectified, image_shape)

        # For wide baseline disparity calculations
        num_disparities = int(image_shape[0]*0.15)
        block_size = 9
        self.wide_stereo_matcher = cv2.StereoSGBM_create(
            minDisparity = 1,
            numDisparities = num_disparities,
            blockSize = block_size,
            P1 = 8*3*block_size*block_size,
            P2 = 32*3*block_size*block_size,
            disp12MaxDiff = 0,
            uniquenessRatio = 6,
            speckleWindowSize = 300,
            speckleRange = 2,
            mode = cv2.STEREO_SGBM_MODE_SGBM
        )
    
    def get_wide_rectified_stereo_images(self):
        left_image = self.left_cam.get_left_image(should_rectify=False)
        right_image = self.right_cam.get_left_image(should_rectify=False)

        left_image_rectified = cv2.remap(left_image, self.map_wide_left_leftcam_x, self.map_wide_left_leftcam_y, cv2.INTER_LINEAR)
        right_image_rectified = cv2.remap(right_image, self.map_wide_left_rightcam_x, self.map_wide_left_rightcam_y, cv2.INTER_LINEAR)

        return left_image_rectified, right_image_rectified
    
    def get_wide_disparity(self):
        left_image, right_image = self.get_wide_rectified_stereo_images()
        disparity = self.left_cam.calc_disparity_(left_image, right_image, self.wide_stereo_matcher)
        return disparity
    
    def get_wide_pc(self):
        left_image, right_image = self.get_wide_rectified_stereo_images()
        disparity = self.get_wide_disparity()

        point_image = cv2.reprojectImageTo3D(disparity, self.Q_wide, handleMissingValues=False)

        valid_indeces = np.logical_and(np.all(np.isfinite(point_image), axis=2), point_image[:,:,2] < 150)
        pc_xyz = point_image[valid_indeces]
        pc_bgr = left_image[valid_indeces] / 255
        pc_rgb = np.fliplr(pc_bgr)

        return pc_xyz, pc_rgb
    
    def get_left_wide_parameters(self):
        K = self.left_P[:3,:3]
        D = np.zeros((1,5), dtype=np.float32)
        return K, D
    
    def project_into_left_wide_image(self, xyz):
        image_shape_xy = self.left_cam.image_shape
        K, D = self.get_left_wide_parameters()
        image_points_inside, image_points_xyz = self.left_cam.project_into_image_with_K_D(xyz, K, D, image_shape_xy)
        return image_points_inside, image_points_xyz
    
    # def get_wide_zed_rectified_images_with_disp0_suggestion(self):
    #     left0_zed = self.left_cam.get_left_image(should_rectify=True)
    #     right0_zed = self.right_cam.get_left_image(should_rectify=True)
    #     left0, right0 = self.rectifier_wide_from_rectified_short_zed.rectify(left0_zed, right0_zed)

    #     Q = self.Q_wide
    #     fB = Q[2,3] / Q[3,2]
    #     disp0_zed = self.left_cam.get_neural_disp()
    #     point_image = cv2.reprojectImageTo3D(disp0_zed, self.left_cam.Q, handleMissingValues=False)
    #     depth0_zed = point_image[:,:,2]
    #     depth0, _ = self.rectifier_wide_from_rectified_short_zed.rectify(depth0_zed, None)
    #     disp0_suggestion = fB / depth0
    #     valid_indeces = np.isfinite(disp0_suggestion)
    #     disp0_suggestion[~valid_indeces] = 0

    #     return left0, right0, disp0_short, disp0_suggestion
    
    # def get_wide_from_short_disp(self, disp_wide):


class ImageRectifier:
    def __init__(self, R, t, K_left, D_left, K_right, D_right, image_shape) -> None:
        self.R1, self.R2, self.left_P, self.right_P, self.Q, roi1, roi2 = cv2.stereoRectify(K_left, D_left, K_right, D_right, image_shape, R, t, alpha=0) # alpha = 0 is zoomed to only valid pixels
        self.map_left_x, self.map_left_y = cv2.initUndistortRectifyMap(K_left, D_left, self.R1, self.left_P, image_shape, cv2.CV_32FC1)
        self.map_right_x, self.map_right_y = cv2.initUndistortRectifyMap(K_right, D_right, self.R2, self.right_P, image_shape, cv2.CV_32FC1)
    
    def rectify(self, image_left, image_right=None):
        image_left_rectified = cv2.remap(image_left, self.map_left_x, self.map_left_y, cv2.INTER_LINEAR)
        if image_right is None:
            return image_left_rectified, None
        
        image_right_rectified = cv2.remap(image_right, self.map_right_x, self.map_right_y, cv2.INTER_LINEAR)
        return image_left_rectified, image_right_rectified

class CachedCalibratedStereoSVOCamera(CalibratedStereoSVOCamera):
    def __init__(self, left_svo_file_path, right_svo_file_path, nanos_to_add_to_left, R_wide, t_wide, cache_folder_path) -> None:
        super().__init__(left_svo_file_path, right_svo_file_path, nanos_to_add_to_left, R_wide, t_wide)
        self.cache_folder_path = cache_folder_path
    
    def get_short_disparity(self):
        timestamp = self.get_timestamp()
        filepath = f"{self.cache_folder_path}/short_{timestamp}.pkl"
        if os.path.isfile(filepath):
            with open(filepath, "rb") as file:
                disparity = pickle.load(file)
        else:
            left_image, right_image = self.get_short_rectified_stereo_images()
            disparity = self.calc_disparity_(left_image, right_image, self.short_stereo_matcher)
            with open(filepath, "wb+") as file:
                pickle.dump(disparity, file)

        # disp_viz = cv2.ximgproc.getDisparityVis(disparity*16)
        # disp_viz = cv2.normalize(disp_viz,  None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # cv2.imshow("Short disparity", disp_viz)
        # cv2.waitKey(10)

        return disparity
    
    def get_wide_disparity(self):
        timestamp = self.get_timestamp()
        filepath = f"{self.cache_folder_path}/wide_{timestamp}.pkl"
        if os.path.isfile(filepath):
            with open(filepath, "rb") as file:
                disparity = pickle.load(file)
        else:
            left_image, right_image = self.get_wide_rectified_stereo_images()
            disparity = self.calc_disparity_(left_image, right_image, self.wide_stereo_matcher)
            with open(filepath, "wb+") as file:
                pickle.dump(disparity, file)

        # disp_viz = cv2.ximgproc.getDisparityVis(disparity*16)
        # disp_viz = cv2.normalize(disp_viz,  None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # cv2.imshow("Wide disparity", disp_viz)
        # cv2.waitKey(10)

        return disparity
