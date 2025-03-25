import numpy as np
from shapely.geometry import Polygon
from collections import deque
import utilities as ut
import cv2
from scipy.interpolate import griddata, Rbf
from scipy import stats
from fastSAM import FastSAMSeg
import json

class Stixels:

    def __init__(self, num_stixels, img_shape, min_stixel_height=20):
        self.num_stixels = num_stixels
        self.img_shape = img_shape
        self.stixel_width = self.stixel_width = int(img_shape[1] // self.num_stixels)

    def get_free_space_boundary(self, water_mask):
        H, W = water_mask.shape

        search_height = H - 50
        reversed_mask = water_mask[:search_height, :][::-1, :]

        reversed_zero_mask = (reversed_mask == 0)
        found = reversed_zero_mask.any(axis=0)
        first_free_idx = reversed_zero_mask.argmax(axis=0)

        free_space_boundary = np.full(W, H, dtype=int)
        free_space_boundary[found] = (search_height - 1) - first_free_idx[found]

        #boundary_mask = np.zeros_like(water_mask, dtype=np.uint8)
        #cols = np.arange(W)
        #valid = free_space_boundary < H
        #boundary_mask[free_space_boundary[valid], cols[valid]] = 1

        free_space_boundary[0] = free_space_boundary[1]

        return free_space_boundary  #, boundary_mask


    def create_stixels(self, water_mask, disparity_img, depth_img, upper_contours):

        free_space_boundary = self.get_free_space_boundary(water_mask)
        segmentation_score_map = self.create_segmentation_score_map(disparity_img, free_space_boundary, upper_contours)
        
        

    def create_segmentation_score_map(self, disparity_img, free_space_boundary, upper_contours):

        normalized_disparity = cv2.normalize(disparity_img, None, 0, 1, cv2.NORM_MINMAX)
        blurred_image = cv2.GaussianBlur(normalized_disparity, (5, 5), 0)
        grad_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=5)
        grad_y = cv2.convertScaleAbs(grad_y)
        _, grad_y = cv2.threshold(grad_y, 0.3, 1, cv2.THRESH_BINARY)
        grad_y = ut.filter_mask_by_boundary(grad_y, free_space_boundary, offset=10)
        grad_y = ut.get_bottommost_line(grad_y)
        

        cv2.imshow("grad_y", grad_y.astype(np.uint8)*255)


    def get_optimal_height(self):
        pass

    def get_stixel_depths_from_lidar(self):
        pass

    def get_stixel_footprints(self):
        pass

    def overlay_stixels_on_image(self):
        pass

    def create_free_space_plygon(self):
        pass