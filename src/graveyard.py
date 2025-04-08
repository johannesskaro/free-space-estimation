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