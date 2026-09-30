import numpy as np
import pyzed.sl as sl


class SVOCamera:
    """Reads a ZED .svo recording using the ZED SDK's NEURAL_PLUS depth."""

    def __init__(self, svo_file_path) -> None:
        cam_input_type = sl.InputType()
        cam_input_type.set_from_svo_file(svo_file_path)
        init_params = sl.InitParameters(input_t=cam_input_type, svo_real_time_mode=False)
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
        init_params.depth_minimum_distance = 0
        init_params.depth_maximum_distance = 60
        self.cam = sl.Camera()
        err = self.cam.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED camera open failed with error: {err}")

        self.runtime = sl.RuntimeParameters()
        self.left_image = sl.Mat()
        self.disparity = sl.Mat()
        self.depth = sl.Mat()

        cam_resolution = self.cam.get_camera_information().camera_configuration.resolution
        self.image_shape = (cam_resolution.width, cam_resolution.height)
        baseline = self.cam.get_camera_information().camera_configuration.calibration_parameters.get_camera_baseline()
        self.R = np.eye(3)
        self.T = np.array([-baseline, 0, 0])

    @property
    def length(self):
        return self.cam.get_svo_number_of_frames()

    def set_svo_position(self, n_frames):
        assert n_frames < self.length
        self.cam.set_svo_position(n_frames)

    def grab(self):
        err = self.cam.grab(self.runtime)
        return err

    def get_left_image(self, should_rectify=False):
        if should_rectify:
            self.cam.retrieve_image(self.left_image, sl.VIEW.LEFT)
        else:
            self.cam.retrieve_image(self.left_image, sl.VIEW.LEFT_UNRECTIFIED)
        image_bgr = self.left_image.get_data()[:, :, :3]
        return np.ascontiguousarray(image_bgr)

    def get_timestamp(self):
        return self.cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).data_ns

    def close(self):
        self.cam.close()

    def get_left_parameters(self):
        """Intrinsics K and distortion D of the (rectified) left camera."""
        cp = self.cam.get_camera_information().camera_configuration.calibration_parameters.left_cam
        K = np.array([
            [cp.fx, 0, cp.cx],
            [0, cp.fy, cp.cy],
            [0, 0, 1]
        ])
        # ZED distortion is [k1, k2, p1, p2, k3], same order as OpenCV
        D = cp.disto
        D = np.array([D[0], D[1], D[2], D[3], D[4]])
        return K, D

    def get_neural_disp(self):
        """Disparity in pixels (positive), with invalid values set to 0."""
        self.cam.retrieve_measure(self.disparity, sl.MEASURE.DISPARITY)
        disp_negative_infs = self.disparity.get_data()
        disp_negative = disp_negative_infs.copy()
        disp_negative[~np.isfinite(disp_negative)] = 0
        disp = -disp_negative
        return disp

    def get_depth_image(self):
        """Depth in meters, with invalid values set to 0."""
        self.cam.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
        return np.nan_to_num(self.depth.get_data(), nan=0)

    def set_svo_position_timestamp(self, timestamp):
        """Seek to the frame closest to `timestamp` (ns, ZED clock)."""
        n_frames_end = self.length - 1
        n_frames = self.find_n_frames(timestamp, 0, n_frames_end)
        self.set_svo_position(n_frames)

    def find_n_frames(self, timestamp, n_frames_start, n_frames_end):
        # Binary search over frame indices
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
