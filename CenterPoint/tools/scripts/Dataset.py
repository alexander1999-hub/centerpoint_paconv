#from waymo_open_dataset import dataset_pb2, label_pb2
#from simple_waymo_open_dataset_reader import WaymoDataFileReader
#from simple_waymo_open_dataset_reader import dataset_pb2, label_pb2
#from simple_waymo_open_dataset_reader import utils
#from scipy.spatial.transform import Rotation as R
import numpy as np 
import math
import pickle



def get_point_cloud_from_gt_frame(frame):
    laser_name = dataset_pb2.LaserName.TOP
    laser = utils.get(frame.lasers, laser_name)
    laser_calibration = utils.get(frame.context.laser_calibrations, laser_name)

    # Parse the top laser range image and get the associated projection.
    ri, camera_projection, range_image_pose = utils.parse_range_image_and_camera_projection(laser)

    # Convert the range image to a point cloud.
    pcl, pcl_attr = utils.project_to_pointcloud(frame, ri, camera_projection, range_image_pose, laser_calibration)

    return pcl

def box_center_to_corner_gt(box, parameter):
    # To return
    corner_boxes = np.zeros((8, 3))

    translation = box[0:3]
    #h, w, l = box[3], box[4], box[5]
    h = box[3] * (1 + parameter)
    w = box[4] * (1 + parameter)
    l = box[5] * (1 + parameter)
    rotation = box[6]
    #print(rotation)

    # Create a bounding box outline
    bounding_box = np.array([
        [-l/2, l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2],
        [-w/2, -w/2, -w/2, -w/2, w/2, w/2, w/2, w/2],
        [-h/2, -h/2, h/2, h/2, -h/2, -h/2, h/2, h/2]])

    # Standard 3x3 rotation matrix around the Z axis
    #rotation_matrix = np.array([
    #    [np.cos(rotation), -np.sin(rotation), 0.0],
    #    [np.sin(rotation), np.cos(rotation), 0.0],
    #   [0.0, 0.0, 1.0]])

    rotation_matrix = np.array([
        [-np.sin(rotation), -np.cos(rotation), 0.0],
        [np.cos(rotation), -np.sin(rotation), 0.0],
        [0.0, 0.0, 1.0]])

    phi = math.pi / 2
    x = -np.sin(rotation + math.pi / 2)
    y = np.cos(rotation + math.pi / 2) 
    z = 0

    horizontal_rotation = np.array([
        [np.cos(phi)+(1-np.cos(phi))*x*x, (1-np.cos(phi))*x*y-np.sin(phi)*z, (1-np.cos(phi))*x*z+np.sin(phi)*y], 
        [(1-np.cos(phi))*y*x + np.sin(phi)*z, np.cos(phi)+(1-np.cos(phi))*y*y, (1-np.cos(phi))*y*z-np.sin(phi)*x],
        [(1-np.cos(phi))*z*x - np.sin(phi)*y, (1-np.cos(phi))*z*y+np.sin(phi)*x, np.cos(phi)+(1-np.cos(phi))*z*z]
    ])

    # Repeat the [x, y, z] eight times
    eight_points = np.tile(translation, (8, 1))
    #print(eight_points)
    # Translate the rotated bounding box by the
    # original center position to obtain the final box
    corner_box = np.dot(horizontal_rotation, np.dot(
        rotation_matrix, bounding_box)) + eight_points.transpose()
    return corner_box.transpose()

def box_center_to_corner_pred(box, parameter):
    # To return
    corner_boxes = np.zeros((8, 3))

    translation = box[0:3]
    #h, w, l = box[3], box[4], box[5]
    h = box[3] * (1 + parameter)
    w = box[4] * (1 + parameter)
    l = box[5] * (1 + parameter)
    rotation = box[6]
    #print(rotation)

    # Create a bounding box outline
    bounding_box = np.array([
        [-l/2, l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2],
        [-w/2, -w/2, -w/2, -w/2, w/2, w/2, w/2, w/2],
        [-h/2, -h/2, h/2, h/2, -h/2, -h/2, h/2, h/2]])

    # Standard 3x3 rotation matrix around the Z axis
    rotation_matrix = np.array([
        [np.cos(rotation), np.sin(rotation), 0.0],
        [-np.sin(rotation), np.cos(rotation), 0.0],
        [0.0, 0.0, 1.0]])

    #rotation_matrix = np.array([
    #    [-np.sin(rotation), np.cos(rotation), 0.0],
    #    [-np.cos(rotation), -np.sin(rotation), 0.0],
    #    [0.0, 0.0, 1.0]])

    phi = math.pi / 2
    x = np.cos(rotation + math.pi / 2)
    y = -np.sin(rotation + math.pi / 2)
    z = 0
    #rotated_vec = rotation.apply(vec)

    horizontal_rotation = np.array([
        [np.cos(phi)+(1-np.cos(phi))*x*x, (1-np.cos(phi))*x*y-np.sin(phi)*z, (1-np.cos(phi))*x*z+np.sin(phi)*y], 
        [(1-np.cos(phi))*y*x + np.sin(phi)*z, np.cos(phi)+(1-np.cos(phi))*y*y, (1-np.cos(phi))*y*z-np.sin(phi)*x],
        [(1-np.cos(phi))*z*x - np.sin(phi)*y, (1-np.cos(phi))*z*y+np.sin(phi)*x, np.cos(phi)+(1-np.cos(phi))*z*z]
    ])

    # Repeat the [x, y, z] eight times
    eight_points = np.tile(translation, (8, 1))
    #print(eight_points)
    # Translate the rotated bounding box by the
    # original center position to obtain the final box
    corner_box = np.dot(horizontal_rotation, np.dot(rotation_matrix, bounding_box)) + eight_points.transpose()
    return corner_box.transpose()    

class GtLoader:
    def __init__(self):
        pass

    def load(self, path):
        self.__datafile = WaymoDataFileReader(path)
        self.__datafile_iter = iter(self.__datafile)

    def first_frame(self):
        return next(iter(self.__datafile))
        

    def next_frame(self):
        return next(self.__datafile_iter)

def get_boxes_from_waymo_frame(frame, parameter):
    #print(frame)
    boxes = 0
    gt_classes = list([])
    first_time = True
    for i in frame.laser_labels:
        box = [
            i.box.center_x,
            i.box.center_y,
            i.box.center_z,
            i.box.width,
            i.box.length,
            i.box.height,
            i.box.heading
        ]
        gt_classes.append(i.type)
        if first_time:
            boxes = np.array([box_center_to_corner_gt(box, parameter)])
            first_time = False
        else:
            boxes = np.concatenate((boxes, np.array([box_center_to_corner_gt(box, parameter)])), axis=0)
    return boxes, gt_classes



class PredLoader:
    def __init__(self):
        pass

    def load(self, path, num):
        with open(path, 'rb') as f:
            self.__datafilePred = pickle.load(f)
        self.__num = num
        self.__it = -1

    def first_frame(self):
        return self.__datafilePred['seq_' + str(self.__num) + '_frame_' + str(0) + '.pkl']
        

    def next_frame(self):
        self.__it += 1
        return self.__datafilePred['seq_' + str(self.__num) + '_frame_' + str(self.__it) + '.pkl']

def get_boxes_from_pred_frame(frame, parameter):
    boxes = 0
    first_time = True
    for i in frame['box3d_lidar']:
        if first_time:
            boxes = np.array([box_center_to_corner_pred(i, parameter)])
            first_time = False
        else:
            boxes = np.concatenate((boxes, np.array([box_center_to_corner_pred(i, parameter)])), axis=0)
    return boxes

