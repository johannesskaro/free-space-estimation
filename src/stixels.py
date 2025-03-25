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

    def __init__(self, num_stixels):
        self.num_stixels = num_stixels

    def get_free_space_boundary(self, water_mask):
        pass


    def create_stixels(self):
        pass

    def create_segmentation_score_map(self):
        pass

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