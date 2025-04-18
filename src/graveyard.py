import numpy as np
import utilities as ut
from scipy.spatial.transform import Rotation as R

def transform_points_into_world(self, p_cam, pose):
        if p_cam.size == 0:
            return np.empty((0, 2))
        
        pos = np.array([pose[1], pose[0]])
        ori = pose[3:]
        heading = R.from_quat(ori).as_euler('xyz', degrees=False)[2] + np.pi
        Rot = ut.rotation_matrix(heading)

        t_world_to_cam = Rot.dot(self.t_body_to_cam) + pos
        p_world = (Rot.dot(p_cam.T)).T + t_world_to_cam

        return p_world
    
def transform_lines_into_world(self, lines, pose):

    pos = np.array([pose[1], pose[0]])
    ori = pose[3:]
    heading = R.from_quat(ori).as_euler('xyz', degrees=False)[2] + np.pi
    Rot = ut.rotation_matrix(heading)

    t_world_to_cam = Rot.dot(self.t_body_to_cam) + pos

    M = np.eye(3)
    M[:2, :2] = Rot
    M[:2, 2] = t_world_to_cam
    M_inv = np.linalg.inv(M)
    T_dual = M_inv.T
    lines_world = T_dual.dot(lines.T).T

    return lines_world

def associate_prev_stixels(self, dist_threshold=1, z_diff_threshold=2):
    N = self.num_stixels
    association_list = np.full(N, -1, dtype=int)
    pts = self.prev_stixel_footprints_curr_frame.copy()

    z_lidar = self.stixel_lidar_depths.copy()

    for n, ray in enumerate(self.projection_rays):

        #if self.dynamic_stixel_list[n] == True:
        #    continue

        if pts.size == 0:
            continue

        idx = n
        idx_minus = n-1 if n-1 >= 0 else n
        idx_plus = n+1 if n+1 < N else n

        d_n_minus_1 = ut.distance_from_point_to_line(pts[idx_minus], ray)
        d_n = ut.distance_from_point_to_line(pts[idx], ray)
        d_n_plus_1 = ut.distance_from_point_to_line(pts[idx_plus], ray)

        #print("d_n_minus_1: ", d_n_minus_1)
        #print("d_n: ", d_n)
        #print("d_n_plus_1: ", d_n_plus_1)
        #print(" ----- ")

        iterating = True

        if d_n <= d_n_minus_1 and d_n <= d_n_plus_1:
            if d_n < dist_threshold:
                #if self.prev_dynamic_stixel_list[idx] == False and self.prev_stixel_has_measurement[idx] == True:
                if self.prev_stixel_has_measurement[idx] == True:
                    if self.prev_dynamic_stixel_list[idx] == self.dynamic_stixel_list[n]:
                    #z_diff = np.abs(z_stereo_prev[idx] - z_stereo[n])
                    #z_diff = np.abs(pts[idx, 0] - z_lidar[n])
                    #if z_diff < z_diff_threshold:
                        association_list[n] = idx

        elif d_n_minus_1 < d_n and d_n_minus_1 < d_n_plus_1:
            i = 2
            d_n_minus_i_prev = d_n_minus_1
            while iterating:
                #print("minus", n)
                idx = n - i
                if idx < 0:
                    if d_n_minus_i_prev < dist_threshold:
                        #if self.prev_dynamic_stixel_list[0] == False and self.prev_stixel_has_measurement[0] == True:
                        if self.prev_stixel_has_measurement[0] == True:
                                if self.prev_dynamic_stixel_list[0] == self.dynamic_stixel_list[n]:
                            #z_diff = np.abs(z_stereo_prev[0] - z_stereo[n])
                            #z_diff = np.abs(pts[0, 0] - z_lidar[n])
                            #if z_diff < z_diff_threshold:            
                                    association_list[n] = 0
                    iterating = False
                    break

                d_n_minus_i = ut.distance_from_point_to_line(pts[idx], ray)
                if d_n_minus_i < d_n_minus_i_prev:
                    d_n_minus_i_prev = d_n_minus_i
                    i += 1
                    continue
                else:
                    if d_n_minus_i_prev < dist_threshold:
                        #if self.prev_dynamic_stixel_list[idx + 1] == False and self.prev_stixel_has_measurement[idx + 1] == True:
                        if self.prev_stixel_has_measurement[idx + 1] == True:
                                if self.prev_dynamic_stixel_list[idx + 1] == self.dynamic_stixel_list[n]:
                            #z_diff = np.abs(z_stereo_prev[idx + 1] - z_stereo[n])
                            #z_diff = np.abs(pts[idx + 1, 0] - z_lidar[n])
                            #if z_diff < z_diff_threshold:
                                    association_list[n] = idx + 1
                    iterating = False
                    break

        elif d_n_plus_1 < d_n and d_n_plus_1 < d_n_minus_1:
            i = 2
            d_n_plus_i_prev = d_n_plus_1
            while iterating:
                #print("plus", n)
                idx = n + i
                if idx >= N:
                    if d_n_plus_i_prev < dist_threshold:
                        #if self.prev_dynamic_stixel_list[N - 1] == False and self.prev_stixel_has_measurement[N - 1] == True:
                        if self.prev_stixel_has_measurement[N - 1] == True:
                                if self.prev_dynamic_stixel_list[N - 1] == self.dynamic_stixel_list[n]:
                            #z_diff = np.abs(z_stereo_prev[N - 1] - z_stereo[n])
                            #z_diff = np.abs(pts[N - 1, 0] - z_lidar[n])
                            #if z_diff < z_diff_threshold:
                                    association_list[n] = N - 1
                    iterating = False
                    break

                d_n_plus_i = ut.distance_from_point_to_line(pts[idx], ray)
                if d_n_plus_i < d_n_plus_i_prev:
                    d_n_plus_i_prev = d_n_plus_i
                    i += 1
                    continue
                else:
                    if d_n_plus_i_prev < dist_threshold:
                        #if self.prev_dynamic_stixel_list[idx - 1] == False and self.prev_stixel_has_measurement[idx - 1] == True:
                        if self.prev_stixel_has_measurement[idx - 1] == True:
                                if self.prev_dynamic_stixel_list[idx - 1] == self.dynamic_stixel_list[n]:
                            #z_diff = np.abs(z_stereo_prev[idx - 1] - z_stereo[n])
                            #z_diff = np.abs(pts[idx - 1, 0] - z_lidar[n])
                            #if z_diff < z_diff_threshold:
                                    association_list[n] = idx - 1
                    iterating = False
                    break
        else:
            print("Error associating previous Stixels for index", n)

    #print(association_list)

    self.association_list = association_list

def associate_prev_stixels_2(self, dist_threshold=0.01, z_diff_threshold=1):
    N = self.num_stixels
    association_list = np.full(N, -1, dtype=int)

    #relevant_prev_points_list = self.prev_stixel_has_measurement & (~self.prev_dynamic_stixel_list)
    relevant_prev_points_list = self.prev_stixel_has_measurement.copy()

    #pts = self.prev_stixel_footprints_curr_frame[relevant_prev_points_list]
    pts = self.prev_stixel_footprints_curr_frame.copy()
    indices = np.where(relevant_prev_points_list)[0]

    z_lidar = self.stixel_lidar_depths.copy()

    for n, ray in enumerate(self.projection_rays):

        if self.dynamic_stixel_list[n] == True:
            continue

        if pts.size == 0 or indices.size == 0:
            continue

        pos = np.abs(indices - n).argmin()
        pos_minus = np.clip(pos - 1, 0, len(indices) - 1)
        pos_plus = np.clip(pos + 1, 0, len(indices) - 1)

        idx = indices[pos]
        idx_minus = np.clip(indices[pos_minus], 0, N - 1)
        idx_plus = np.clip(indices[pos_plus], 0, N - 1)

        d_n_minus_1 = ut.distance_from_point_to_line(pts[idx_minus], ray)
        d_n = ut.distance_from_point_to_line(pts[idx], ray)
        d_n_plus_1 = ut.distance_from_point_to_line(pts[idx_plus], ray)

        #print("d_n_minus_1: ", d_n_minus_1)
        #print("d_n: ", d_n)
        #print("d_n_plus_1: ", d_n_plus_1)
        #print(" ----- ")

        iterating = True

        if d_n <= d_n_minus_1 and d_n <= d_n_plus_1:
            if d_n < dist_threshold:
                if self.prev_dynamic_stixel_list[idx] == False:
                    #z_diff = np.abs(pts[idx, 0] - z_lidar[n])
                    #if z_diff < z_diff_threshold or np.isnan(z_lidar[n]):
                    #if z_diff < z_diff_threshold:
                        association_list[n] = idx

        elif d_n_minus_1 < d_n and d_n_minus_1 < d_n_plus_1:
            i = 2
            d_n_minus_i_prev = d_n_minus_1
            while iterating:
                #print("minus", n)
                pos_minus = pos - i
                if pos_minus < 0:
                    idx = indices[0]
                    if d_n_minus_i_prev < dist_threshold:
                        if self.prev_dynamic_stixel_list[idx] == False:
                            #z_diff = np.abs(pts[idx, 0] - z_lidar[n])
                            #if z_diff < z_diff_threshold:            
                                association_list[n] = idx
                    iterating = False
                    break
                else:
                    idx = indices[pos_minus]

                d_n_minus_i = ut.distance_from_point_to_line(pts[idx], ray)
                if d_n_minus_i < d_n_minus_i_prev:
                    d_n_minus_i_prev = d_n_minus_i
                    i += 1
                    continue
                else:
                    if d_n_minus_i_prev < dist_threshold:
                        idx = indices[pos_minus + 1]
                        if self.prev_dynamic_stixel_list[idx] == False:
                            #z_diff = np.abs(pts[idx, 0] - z_lidar[n])
                            #if z_diff < z_diff_threshold:
                                association_list[n] = idx
                    iterating = False
                    break

        elif d_n_plus_1 < d_n and d_n_plus_1 < d_n_minus_1:
            i = 2
            d_n_plus_i_prev = d_n_plus_1
            while iterating:
                #print("plus", n)
                pos_plus = pos + i
                if pos_plus >= len(indices):
                    idx = indices[-1]
                    if d_n_plus_i_prev < dist_threshold:
                        if self.prev_dynamic_stixel_list[idx] == False:
                            #z_diff = np.abs(pts[idx, 0] - z_lidar[n])
                            #if z_diff < z_diff_threshold:
                                association_list[n] = idx
                    iterating = False
                    break
                else:
                    idx = indices[pos_plus]

                d_n_plus_i = ut.distance_from_point_to_line(pts[idx], ray)
                if d_n_plus_i < d_n_plus_i_prev:
                    d_n_plus_i_prev = d_n_plus_i
                    i += 1
                    continue
                else:
                    if d_n_plus_i_prev < dist_threshold:
                        idx = indices[pos_plus - 1]
                        if self.prev_dynamic_stixel_list[idx] == False:
                            #z_diff = np.abs(pts[idx, 0] - z_lidar[n])
                            #if z_diff < z_diff_threshold:
                                association_list[n] = idx
                    iterating = False
                    break
        else:
            print("Error associating previous Stixels for index", n)

    #print(association_list)

    self.association_list = association_list


def associate_prev_stixels_3(self, ang_thres_deg=0.25):
    ang_thres = np.deg2rad(ang_thres_deg)
    N = self.num_stixels
    association_list = np.full(N, -1, dtype=int)

    relevant_prev_points_list = self.prev_stixel_has_measurement.copy()
    pts = self.prev_stixel_footprints_curr_frame.copy()
    indices = np.where(relevant_prev_points_list)[0]

    z_lidar = self.stixel_lidar_depths.copy()

    for n, ray in enumerate(self.projection_rays):

        #if self.dynamic_stixel_list[n] == True:
            #   continue

        if pts.size == 0 or indices.size == 0:
            continue

        pos = np.abs(indices - n).argmin()
        pos_minus = np.clip(pos - 1, 0, len(indices) - 1)
        pos_plus = np.clip(pos + 1, 0, len(indices) - 1)

        idx = indices[pos]
        idx_minus = np.clip(indices[pos_minus], 0, N - 1)
        idx_plus = np.clip(indices[pos_plus], 0, N - 1)

        d_n_minus_1 = ut.angular_error(pts[idx_minus], ray)
        d_n = ut.angular_error(pts[idx], ray)
        d_n_plus_1 = ut.angular_error(pts[idx_plus], ray)

        #print("d_n_minus_1: ", d_n_minus_1)
        #print("d_n: ", d_n)
        #print("d_n_plus_1: ", d_n_plus_1)
        #print(" ----- ")

        iterating = True

        if d_n <= d_n_minus_1 and d_n <= d_n_plus_1:
            if d_n < ang_thres:
                #if self.prev_dynamic_stixel_list[idx] == self.dynamic_stixel_list[n]:
                #if self.prev_dynamic_stixel_list[idx] == False:
                        association_list[n] = idx

        elif d_n_minus_1 < d_n and d_n_minus_1 < d_n_plus_1:
            i = 2
            d_n_minus_i_prev = d_n_minus_1
            while iterating:
                #print("minus", n)
                pos_minus = pos - i
                if pos_minus < 0:
                    idx = indices[0]
                    if d_n_minus_i_prev < ang_thres:
                        #if self.prev_dynamic_stixel_list[idx] == self.dynamic_stixel_list[n]:
                        #if self.prev_dynamic_stixel_list[idx] == False:
                                association_list[n] = idx
                    iterating = False
                    break
                else:
                    idx = indices[pos_minus]

                d_n_minus_i = ut.angular_error(pts[idx], ray)
                if d_n_minus_i < d_n_minus_i_prev:
                    d_n_minus_i_prev = d_n_minus_i
                    i += 1
                    continue
                else:
                    if d_n_minus_i_prev < ang_thres:
                        idx = indices[pos_minus + 1]
                        #if self.prev_dynamic_stixel_list[idx] == self.dynamic_stixel_list[n]:
                        #if self.prev_dynamic_stixel_list[idx] == False:
                        association_list[n] = idx
                    iterating = False
                    break

        elif d_n_plus_1 < d_n and d_n_plus_1 < d_n_minus_1:
            i = 2
            d_n_plus_i_prev = d_n_plus_1
            while iterating:
                #print("plus", n)
                pos_plus = pos + i
                if pos_plus >= len(indices):
                    idx = indices[-1]
                    if d_n_plus_i_prev < ang_thres:
                        #if self.prev_dynamic_stixel_list[idx] == self.dynamic_stixel_list[n]:
                        #if self.prev_dynamic_stixel_list[idx] == False:
                                association_list[n] = idx
                    iterating = False
                    break
                else:
                    idx = indices[pos_plus]

                d_n_plus_i = ut.angular_error(pts[idx], ray)
                if d_n_plus_i < d_n_plus_i_prev:
                    d_n_plus_i_prev = d_n_plus_i
                    i += 1
                    continue
                else:
                    if d_n_plus_i_prev < ang_thres:
                        idx = indices[pos_plus - 1]
                        #if self.prev_dynamic_stixel_list[idx] == self.dynamic_stixel_list[n]:
                        #if self.prev_dynamic_stixel_list[idx] == False:
                        association_list[n] = idx
                    iterating = False
                    break
        else:
            print("Error associating previous Stixels for index", n)

    #print(association_list)

    self.association_list = association_list