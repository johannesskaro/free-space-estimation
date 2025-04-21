import numpy as np
import utilities as ut
from scipy.spatial.transform import Rotation as R
from collections import defaultdict

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

    def associate_prev_stixels_old(self, delta_heading, ang_thres_deg=0.5, z_diff_thres=1):
        ang_thres = np.deg2rad(ang_thres_deg)
        N = self.num_stixels
        association_heigth = np.full(N, -1, dtype=int)
        association_depth = np.full(N, -1, dtype=int)
        best_ray_err = np.full(N, np.inf, dtype=float)
        best_ray_idx = np.full(N, -1, dtype=int)

        cam_fov = 110
        RAY_SPACING = cam_fov / N
        DELTA_IDX = int(round(delta_heading / RAY_SPACING))

        relevant_prev_points_list = self.prev_stixel_has_measurement.copy()
        pts = self.prev_stixel_footprints_curr_frame.copy()
        indices = np.where(relevant_prev_points_list)[0]

        z_lidar = self.stixel_lidar_depths.copy()

        for n, ray in enumerate(self.projection_rays):

            if pts.size == 0 or indices.size == 0:
                continue

            idx_pred = n + DELTA_IDX
            pos = np.abs(indices - idx_pred).argmin()
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
                    if self.prev_dynamic_stixel_list[idx] == self.dynamic_stixel_list[n]:
                        best_ray_idx[n] = idx
                        best_ray_err[n] = d_n
                    if self.prev_dynamic_stixel_list[idx] == False or self.dynamic_stixel_list[n] == False:
                        z_diff = np.abs(pts[idx, 0] - z_lidar[n])
                        if z_diff < z_diff_thres:
                            association_heigth[n] = idx

            elif d_n_minus_1 < d_n and d_n_minus_1 < d_n_plus_1:
                i = 2
                d_n_minus_i_prev = d_n_minus_1
                while iterating:
                    #print("minus", n)
                    pos_minus = pos - i
                    if pos_minus < 0:
                        idx = indices[0]
                        if d_n_minus_i_prev < ang_thres:
                            if self.prev_dynamic_stixel_list[idx] == self.dynamic_stixel_list[n]:
                                best_ray_idx[n] = idx
                                best_ray_err[n] = d_n_minus_i_prev
                            if self.prev_dynamic_stixel_list[idx] == False or self.dynamic_stixel_list[n] == False:
                                z_diff = np.abs(pts[idx, 0] - z_lidar[n])
                                if z_diff < z_diff_thres:
                                    association_heigth[n] = idx
                                    
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
                            if self.prev_dynamic_stixel_list[idx] == self.dynamic_stixel_list[n]:
                                best_ray_idx[n] = idx
                                best_ray_err[n] = d_n_minus_i_prev
                            if self.prev_dynamic_stixel_list[idx] == False or self.dynamic_stixel_list[n] == False:
                                z_diff = np.abs(pts[idx, 0] - z_lidar[n])
                                if z_diff < z_diff_thres:
                                    association_heigth[n] = idx
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
                            if self.prev_dynamic_stixel_list[idx] == self.dynamic_stixel_list[n]:
                                best_ray_idx[n] = idx
                                best_ray_err[n] = d_n_plus_i_prev
                            if self.prev_dynamic_stixel_list[idx] == False or self.dynamic_stixel_list[n] == False:
                                z_diff = np.abs(pts[idx, 0] - z_lidar[n])
                                if z_diff < z_diff_thres:
                                    association_heigth[n] = idx
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
                            if self.prev_dynamic_stixel_list[idx] == self.dynamic_stixel_list[n]:
                                best_ray_idx[n] = idx
                                best_ray_err[n] = d_n_plus_i_prev
                            if self.prev_dynamic_stixel_list[idx] == False or self.dynamic_stixel_list[n] == False:
                                z_diff = np.abs(pts[idx, 0] - z_lidar[n])
                                if z_diff < z_diff_thres:
                                    association_heigth[n] = idx
                        iterating = False
                        break
            else:
                print("Error associating previous Stixels for index", n)

        claim = defaultdict(list)
        for k, idx in enumerate(best_ray_idx):
            if idx != -1:                         # ray has a tentative match
                claim[idx].append((k, best_ray_err[k]))

        for idx, lst in claim.items():
            # lst is [(ray, err), (ray, err), ...]  all wanting the same point idx
            k_best, _ = min(lst, key=lambda t: t[1])   # smallest angular error
            association_depth[k_best] = idx 

    
        self.association_depth = association_depth
        self.association_height = association_heigth



    def transform_prev_stixels_into_curr_frame_old(self, prev_pose, curr_pose):

        p_cam_prev = self.prev_stixel_footprints

        if p_cam_prev.size == 0:
            return np.empty((0, 2))

        #pos_prev = np.array([prev_pose[1], prev_pose[0]]) # Convert from NED to XZ-plane
        #pos_curr = np.array([curr_pose[1], curr_pose[0]]) # Convert from NED to XZ-plane
        pos_prev = np.array([prev_pose[0], prev_pose[1]]) 
        pos_curr = np.array([curr_pose[0], curr_pose[1]]) 
        ori_prev = prev_pose[3:]
        ori_curr = curr_pose[3:]

        # + np.pi is only for MA2, since it is driving backwards, remove for BlueBoat
        heading_prev = R.from_quat(ori_prev).as_euler('xyz', degrees=False)[2] + np.pi
        heading_curr = R.from_quat(ori_curr).as_euler('xyz', degrees=False)[2] + np.pi

        #print("heading_curr: ", heading_curr * 180/np.pi)
        #print("Direction: ", heading_curr * 180/np.pi - heading_prev * 180/np.pi)

        R_prev = ut.rotation_matrix(heading_prev)
        R_curr = ut.rotation_matrix(heading_curr)

        t_world_to_cam_prev = R_prev.dot(self.t_body_to_cam) + pos_prev
        t_world_to_cam_curr = R_curr.dot(self.t_body_to_cam) + pos_curr

        #p_cam_prev = np.asarray(prev_stixel_points)
        p_world_prev = (R_prev.dot(p_cam_prev.T)).T + t_world_to_cam_prev
        p_cam_curr = (R_curr.T.dot((p_world_prev - t_world_to_cam_curr).T)).T

        self.prev_stixel_footprints_curr_frame = p_cam_curr.copy()


    def refine_assocatied_depths(self, delta_heading):

        cam_fov = 110
        RAY_SPACING = cam_fov / self.num_stixels
        DELTA_IDX = int(round(delta_heading / RAY_SPACING))

        for n in range(self.num_stixels):

            z_lidar = self.stixel_lidar_depths[n]     

            if np.isnan(z_lidar) or np.isinf(z_lidar):
                has_lidar = False
            else:
                has_lidar = True

            if self.association_depth[n] != -1:
                has_prop = True
                idx = self.association_depth[n]

                z_lidar_prop = self.prev_stixel_lidar_depths[idx]

                if np.isnan(z_lidar_prop) or np.isinf(z_lidar_prop):
                    prop_has_lidar = False
                else:
                    prop_has_lidar = True

            else:
                has_prop = False
                
            if has_prop:
                #if has_lidar and (not prop_has_lidar):
                if has_lidar:
                    self.prop_set.discard(idx)

                if (not has_lidar) and prop_has_lidar:
                    self.prop_set.add(n)
                
                if (not has_lidar) and not prop_has_lidar:
                    if idx in self.prop_set:
                        idx_pred = idx + DELTA_IDX
                        if 0 <= idx_pred < self.num_stixels and n == idx_pred:
                            self.prop_set.discard(idx)
                            self.prop_set.add(n)
                        else:
                            self.association_depth[n] = -1

        #print(self.prop_set)