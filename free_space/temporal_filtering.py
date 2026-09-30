from collections import deque

import numpy as np
from numba import njit, prange

from free_space import utils as ut


class TemporalFiltering:
    """Smooths the water mask over the last N frames and flags frames where the
    segmentation looks wrong.
    """

    def __init__(self, N, camera_height=-1.6):
        self.N = N
        self.past_frames = deque(maxlen=self.N)
        self.rolling_sum = None
        self.camera_height = camera_height  # meters, expected water plane height

    def add_frame(self, frame):
        self.past_frames.append(frame)

    def filter(self, water_mask):
        """A pixel is water if it is water now, or in at least 2/3 of the last N frames."""
        if self.rolling_sum is None:
            self.rolling_sum = np.zeros_like(water_mask, dtype=np.int32)

        if len(self.past_frames) < self.N:
            smoothed_water_mask = water_mask.copy()
        else:
            threshold = self.N * 2 // 3
            smoothed_water_mask = threshold_and_logical_or(
                water_mask, self.rolling_sum, threshold
            )

        if len(self.past_frames) == self.N:
            oldest_frame = self.past_frames[0]
            self.rolling_sum -= oldest_frame

        self.rolling_sum += water_mask
        self.add_frame(water_mask)

        return smoothed_water_mask

    def detect_segmentation_failure(self, water_mask, plane_model):
        """
        Returns True if the water mask covers too little/much of the image, changes
        too much from the latest frame, or the water plane is at the wrong height.

        Must be called after filter(), so past_frames[-1] is the current
        unfiltered mask.
        """
        water_ratio = np.mean(water_mask)
        if water_ratio < 0.1 or water_ratio > 0.9:
            print("Water ratio:", water_ratio)
            return True

        prev_mask = self.past_frames[-1] if self.past_frames else None

        iou = ut.calculate_iou(water_mask, prev_mask)

        if iou < 0.5:
            print("Mask iou:", iou)
            return True

        normal = plane_model[:3]
        d = plane_model[3]
        normal_length = np.linalg.norm(normal)
        height = d / normal_length

        if abs(height - self.camera_height) > 0.2:
            print("Plane height deviation:", abs(height - self.camera_height))
            return True

        return False


@njit(parallel=True)
def threshold_and_logical_or(water_mask, rolling_sum, threshold):
    H, W = water_mask.shape
    smoothed = np.empty((H, W), dtype=np.uint8)

    for i in prange(H):
        for j in range(W):
            if rolling_sum[i, j] >= threshold or water_mask[i, j] == 1:
                smoothed[i, j] = 1
            else:
                smoothed[i, j] = 0

    return smoothed
