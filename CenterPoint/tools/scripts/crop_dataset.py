import numpy as np
import math
import cv2
import io
import sys

from simple_waymo_open_dataset_reader import WaymoDataFileReader
from simple_waymo_open_dataset_reader import dataset_pb2, label_pb2
from simple_waymo_open_dataset_reader import utils


def crop_points(point_cloud, boxes):

    #laser_name = dataset_pb2.LaserName.TOP
    #laser = utils.get(frame.lasers, laser_name)
    #laser_calibration = utils.get(frame.context.laser_calibrations, laser_name)

    waymo_labels = []
    waymo_boxes = []

    for box in boxes:
        waymo_label = label_pb2.Label()
        waymo_label.box.center_x = box[0]
        waymo_label.box.center_y = box[1]
        waymo_label.box.center_z = box[2]
        waymo_label.box.width = box[3]
        waymo_label.box.length = box[4]
        waymo_label.box.height = box[5]
        waymo_label.box.heading = box[6]
        waymo_boxes.append(waymo_label.box)

    vehicle_to_labels = [np.linalg.inv(utils.get_box_transformation_matrix(label.box)) for label in waymo_label.box]
    vehicle_to_labels = np.stack(vehicle_to_labels)

    pcl1 = np.concatenate((pcl,np.ones_like(pcl[:,0:1])),axis=1)

    proj_pcl = np.einsum('lij,bj->lbi', vehicle_to_labels, pcl1)

    mask = np.logical_and.reduce(np.logical_and(proj_pcl >= -1, proj_pcl <= 1),axis=2)

    result = []

    for i in range(len(proj_pcl)):

        for j in range(len(mask[i])):
            if mask[i][j]:
                    result.append([pcl[j][0], pcl[j][1], pcl[j][2]])

    return np.array(result)