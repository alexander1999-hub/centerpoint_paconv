import argparse
import copy
import json
import os
import sys
import random
import math
try:
    import apex
except:
    print("No APEX!")
import numpy as np
import torch
import yaml
from det3d import torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    batch_processor,
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)
from det3d.torchie.trainer import get_dist_info, load_checkpoint
from det3d.torchie.trainer.utils import all_gather, synchronize
from torch.nn.parallel import DistributedDataParallel
import pickle 
import time 
from scripts.Dataset import get_boxes_from_pred_frame
#from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import os
import open3d as o3d
from open3d import *
from util.util import cal_loss, IOStream, load_cfg_from_cfg_file, merge_cfg_from_list
from model.my_DGCNN_PAConv import PAConv
import torch.nn as nn

device_o3d = o3d.core.Device("CUDA:0")


def get_parser():
    cfg = load_cfg_from_cfg_file('tools/config/dgcnn_paconv_train.yaml')
    cfg['manual_seed'] = cfg.get('manual_seed', 0)
    cfg['workers'] = cfg.get('workers', 6)
    return cfg

def define_models():

    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

    args = get_parser()

    model_pedestrians =  PAConv(args).to(device, dtype=torch.float)
    model_vehicles =  PAConv(args).to(device, dtype=torch.float)
    
    model_vehicles.load_state_dict(torch.load("tools/checkpoints/dgcnn_paconv_train/vehicles_12_08.t7"))
    model_pedestrians.load_state_dict(torch.load("tools/checkpoints/dgcnn_paconv_train/pedestrians_12_08.t7"))
    model_vehicles = model_vehicles.eval()
    model_pedestrians = model_pedestrians.eval()

    return model_vehicles, model_pedestrians, device

def define_model_4cls():
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

    args = get_parser()

    model =  PAConv(args).to(device, dtype=torch.float)
    model.load_state_dict(torch.load("tools/checkpoints/dgcnn_paconv_train/best_model.t7"))
    model = model.eval()

    return model, device

def box_center_to_corner_gt(box, parameter):
    # To return
    corner_boxes = np.zeros((8, 3))

    translation = box[0:3]
    #h, w, l = box[3], box[4], box[5]
    h = box[3] * (1 + parameter)
    w = box[4] * (1 + parameter)
    l = box[5] * (1 + parameter)
    rotation = box[8] + math.pi / 2
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

def transform(num_points : int, pointcloud : np.array) : 
    if (pointcloud.shape[0] < num_points) : 
        while (pointcloud.shape[0] < num_points) : 
            new_point = (pointcloud[random.randint(0, pointcloud.shape[0]-1)] + pointcloud[random.randint(0, pointcloud.shape[0]-1)]) / 2 + (pointcloud[random.randint(0, pointcloud.shape[0]-1)] - pointcloud[random.randint(0, pointcloud.shape[0]-1)]) * random.randint(-5,5)/100
            pointcloud = np.concatenate((pointcloud, [new_point]), axis = 0)
    if (pointcloud.shape[0] > num_points) : 
            np.random.shuffle(pointcloud)
            pointcloud = pointcloud[0:num_points]
    return pointcloud

def divide_chunks(l, n):
    for i in range(0, len(l), n): 
        yield l[i:i + n]

def load_my_waymo_data(class_name, frame_dict, batch_size): 

    loaded_waymo_pointclouds = list([])

    false_list = list([])
    index_list = list([])
    for box_index in frame_dict.keys() : 
        if class_name == 'vehicle' : 
            if frame_dict[box_index][1] == 0:
                index_list.append(box_index)
                #print(frame_dict[box_index][0].shape[0])
                if frame_dict[box_index][0].shape[0] >= 2 :
                    loaded_waymo_pointclouds.append(transform(100, frame_dict[box_index][0]))
                else : 
                    loaded_waymo_pointclouds.append(np.zeros((100, 3)))
                    false_list.append(box_index)
        if class_name == 'pedestrian' : 
            if frame_dict[box_index][1] == 1:
                index_list.append(box_index)
                if frame_dict[box_index][0].shape[0] >= 2 :
                    loaded_waymo_pointclouds.append(transform(100, frame_dict[box_index][0]))
                else : 
                    loaded_waymo_pointclouds.append(np.zeros((100, 3)))
                    false_list.append(box_index)
    #print("load data", len(index_list), len(loaded_waymo_pointclouds))
    temp = list(zip(loaded_waymo_pointclouds, index_list))
    batches = divide_chunks(temp, batch_size), false_list

    return batches

def load_my_waymo_data_4cls(frame_dict, batch_size): 

    loaded_waymo_pointclouds = list([])

    false_list = list([])
    index_list = list([])
    for box_index in frame_dict.keys() : 
        if frame_dict[box_index][1] == 1 or frame_dict[box_index][1] == 0 :
            index_list.append(box_index)
            if frame_dict[box_index][0].shape[0] >= 2 :
                loaded_waymo_pointclouds.append(transform(50, frame_dict[box_index][0]))
            else : 
                loaded_waymo_pointclouds.append(np.zeros((50, 3)))
                false_list.append(box_index)
    #print("load data", len(index_list), len(loaded_waymo_pointclouds))
    temp = list(zip(loaded_waymo_pointclouds, index_list))
    batches = divide_chunks(temp, batch_size), false_list

    return batches

def PAConv_test(frame_dict, class_name, batch_size, model, device):

    #device = torch.device('cuda:0')
    #torch.cuda.set_device(device)
    m = nn.Softmax(dim=1)
    test_pred = []
    index_list = []
    data_list_zipped, false_list = load_my_waymo_data(class_name, frame_dict, batch_size)
    for batch in data_list_zipped:
        pointcloud,index = zip(*batch)
        index = list(index)
        index_list.extend(index)
        pointcloud = np.stack(list(pointcloud))

        data = torch.tensor(pointcloud).to(device, dtype=torch.float)
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        with torch.no_grad():
            logits = model(data)
            #print("/n logits: ", logits)
            
            #preds1 = m(logits).max(dim=1)[1]
            #print("softmax :", preds1)

        softmax = m(logits).to(device)
        b = torch.zeros(softmax.shape[0]).to(device)
        a_1 = (softmax > 0.85)[:,1].to(device)
        preds = a_1 + b
        #preds = logits.max(dim=1)[1].to(device)
        #print("n_preds :",preds)
        #print("n1preds :",preds1)
        test_pred.append(preds.detach().cpu().numpy())
    if len(test_pred) > 0 : 
        test_pred = np.concatenate(test_pred)
    elif len(test_pred) == 0 : 
        test_pred = np.array([])

    #print(class_name, len(test_pred), len(index_list), len(false_list))

    return test_pred, index_list, false_list

def PAConv_test_4cls(frame_dict, batch_size, model, device):

    #device = torch.device('cuda:0')
    #torch.cuda.set_device(device)
    #m = nn.Softmax(dim=1)
    test_pred = []
    index_list = []
    data_list_zipped, false_list = load_my_waymo_data_4cls(frame_dict, batch_size)
    for batch in data_list_zipped:
        pointcloud,index = zip(*batch)
        index = list(index)
        index_list.extend(index)
        pointcloud = np.stack(list(pointcloud))

        data = torch.tensor(pointcloud).to(device, dtype=torch.float)
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        with torch.no_grad():
            logits = model(data)
            #print("/n logits: ", logits)
            
            #preds1 = m(logits).max(dim=1)[1]
            #print("softmax :", preds1)

        #softmax = m(logits).to(device)
        #b = torch.zeros(softmax.shape[0]).to(device)
        #a_1 = (softmax > 0.85)[:,1].to(device)
        #preds = a_1 + b
        preds = logits.max(dim=1)[1].to(device)
        #print("n_preds :",preds)
        #print("n1preds :",preds1)
        test_pred.append(preds.detach().cpu().numpy())
    if len(test_pred) > 0 : 
        test_pred = np.concatenate(test_pred)
    elif len(test_pred) == 0 : 
        test_pred = np.array([])

    #print(class_name, len(test_pred), len(index_list), len(false_list))

    #print("Amount of clouds for classification ", len(test_pred))
    return test_pred, index_list, false_list

def my_box_center_to_corner_pred(box, parameter):
    # To return
    #corner_boxes = torch.zeros((8, 3)).to(device)

    translation = torch.unsqueeze(box[0:3], 1).to(device='cuda:0')
    #h, w, l = box[3], box[4], box[5]
    l = box[3] * (1 + parameter)
    w = box[4] * (1 + parameter)
    h = box[5] * (1 + parameter)
    rotation = -1 * box[6]
    z_axis = 1/2 * torch.cuda.FloatTensor([-h,-h,-h,-h,h,h,h,h])
    #print(rotation)

    bounding_box = torch.cuda.FloatTensor([
        [l/2, l/2, -l/2, -l/2],
        [w/2, -w/2, -w/2, w/2]])    
    
    rotation_matrix = torch.cuda.FloatTensor([
                [torch.cos(rotation), -torch.sin(rotation)],
                [torch.sin(rotation), torch.cos(rotation)]
        ])
    temp = torch.mm(rotation_matrix, bounding_box)
    temp1 = torch.cat((temp,temp),1)
    temp2 = torch.vstack((temp1, z_axis))
    temp3 = temp2 + translation
    return temp3.transpose(0,1)


def get_cropped_boxes_gpu(pred_boxes_tensor, pred_labels, points_tensor, scores, parameter, pcd_gpu, count) : 
    frame_dict = {}
    """
    list_for_vis = []
    list_for_vis.append(pcd_gpu)

    lines = [
                [0, 1],
                [0, 2],
                [1, 3],
                [2, 3],
                [4, 5],
                [4, 6],
                [5, 7],
                [6, 7],
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
                [0, 3],
                [1, 2],
                [4, 7],
                [5, 6]
            ]
    """

    for i in range(len(pred_boxes_tensor)) : 
        if scores[i] < 0.4 : 
            #box_coord = box_center_to_corner_pred_gpu(pred_boxes_tensor[i], parameter=0.15, device = device1)
            box_coord = my_box_center_to_corner_pred(pred_boxes_tensor[i], parameter=0.15).cpu().detach().numpy()
            points_test = o3d.utility.Vector3dVector(box_coord)
            oriented_bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(points_test)
            one_crop = pcd_gpu.crop(oriented_bounding_box)

            frame_dict[i] = (np.asarray(one_crop.points), pred_labels[i])
    """
            line_set_pred = o3d.geometry.LineSet(
                points_test,
                lines=o3d.utility.Vector2iVector(lines),
            )
            list_for_vis.append(line_set_pred)
            
            frame_dict[i] = (np.asarray(one_crop.points), pred_labels[i])
    if count == 1 :
        o3d.visualization.draw_geometries(list_for_vis)
    """                
    return frame_dict

def get_cropped_boxes(pred_boxes, pred_labels, pcd, scores) : 
    frame_dict = {}

    for i in range(len(pred_boxes)) : 
        if scores[i] < 0.4 : 
            points = o3d.utility.Vector3dVector(pred_boxes[i])
            oriented_bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(points)
            one_crop = pcd.crop(oriented_bounding_box)
            frame_dict[i] = (np.asarray(one_crop.points), pred_labels[i])
        
    
    return frame_dict

def check_classes(pred_boxes: np.array, pred_classes, gt_boxes: np.array, gt_classes) : 
        boxes_list_pred = list([])
        classes_list_pred = list([])

        boxes_list_gt = list([])
        classes_list_gt = list([])

        pred_poly = []
        gt_poly = []

        inter_gt = []

        for i in range(len(pred_boxes)):
                test_pcd = pred_boxes[i]
                test_pcd = np.delete(test_pcd,2,1)
                try :     
                        hull = ConvexHull(test_pcd)
                except :
                        test_pcd = np.array([
                                [0,0.001],
                                [0,0.002],
                                [0,0.003],
                                [0.001,0.003],
                                [0.002,0.003],
                                [0.003,0.003],
                                [0.002,0.001],
                                [0.002,0]         
                        ])
                        hull = ConvexHull(test_pcd)
                hull_indices = hull.vertices
                hull_pts = test_pcd[hull_indices, :]
                result = list(map(tuple, hull_pts))
                p1 = Polygon(result)
                pred_poly.append(p1)
        
        for j in range(len(gt_boxes)):
                test_pcd = gt_boxes[j]
                test_pcd = np.delete(test_pcd,2,1)
                hull = ConvexHull(test_pcd)
                hull_indices = hull.vertices
                hull_pts = test_pcd[hull_indices, :]
                result = list(map(tuple, hull_pts))
                p1 = Polygon(result)
                gt_poly.append(p1)
                                 
        for i in range(len(pred_boxes)) : 
                if (pred_classes[i] == 0) or (pred_classes[i] == 1) : 
                        intersect_count = 0
                        for j in range(len(gt_boxes)) : 
                                if pred_poly[i].intersects(gt_poly[j]) :
                                        if (pred_poly[i].intersection(gt_poly[j]).area / pred_poly[i].union(gt_poly[j]).area) > 0.5 :
                                                inter_gt.append(j)
                                                #print(i, j, iou3d(pred_boxes[i], gt_boxes[j]), pred_classes[i], gt_classes[j])
                                                intersect_count += 1
                                                if pred_classes[i] == 0 :
                                                        if gt_classes[j] == 1 : 
                                                                classes_list_pred.append(0)
                                                        else : 
                                                                classes_list_pred.append(1)
                                                else :
                                                        if gt_classes[j] == 2 : 
                                                                classes_list_pred.append(2)
                                                        else : 
                                                                classes_list_pred.append(3)
                                                break
                        if intersect_count == 0 : 
                                if pred_classes[i] == 0 : 
                                        classes_list_pred.append(1)
                                else :
                                        classes_list_pred.append(3)
                        boxes_list_pred.append(i)

        for i in range(len(gt_boxes)) :
                if i not in inter_gt : 
                        if gt_classes[i] == 1 :
                                boxes_list_gt.append(i)
                                classes_list_gt.append(0)
                        if gt_classes[i] == 2 :
                                boxes_list_gt.append(i)
                                classes_list_gt.append(2)

        return boxes_list_pred, classes_list_pred, boxes_list_gt, classes_list_gt

def recall(pred_boxes: np.array, pred_classes, gt_boxes: np.array, gt_classes) : 
        vehicles_gt = 0
        pedestr_gt = 0 
        cycl_gt = 0
        vehicles_pred = 0
        pedestr_pred = 0
        cycl_pred = 0
        vehicles_true = 0
        pedestr_true = 0
        cycl_true = 0

        pred_poly = []
        gt_poly = []

        for i in range(len(pred_boxes)):
                test_pcd = pred_boxes[i]
                test_pcd = np.delete(test_pcd,2,1)
                try :     
                        hull = ConvexHull(test_pcd)
                except :
                        test_pcd = np.array([
                                [0,0.001],
                                [0,0.002],
                                [0,0.003],
                                [0.001,0.003],
                                [0.002,0.003],
                                [0.003,0.003],
                                [0.002,0.001],
                                [0.002,0]         
                        ])
                        hull = ConvexHull(test_pcd)
                hull_indices = hull.vertices
                hull_pts = test_pcd[hull_indices, :]
                result = list(map(tuple, hull_pts))
                p1 = Polygon(result)
                pred_poly.append(p1)
        
        for j in range(len(gt_boxes)):
                test_pcd = gt_boxes[j]
                test_pcd = np.delete(test_pcd,2,1)
                hull = ConvexHull(test_pcd)
                hull_indices = hull.vertices
                hull_pts = test_pcd[hull_indices, :]
                result = list(map(tuple, hull_pts))
                p1 = Polygon(result)
                gt_poly.append(p1)
                                 
        for i in range(len(gt_boxes)) : 

                if gt_classes[i] == 1 : 
                    vehicles_gt += 1
                elif gt_classes[i] == 2 : 
                    pedestr_gt += 1
                elif gt_classes[i] == 4 : 
                    cycl_gt += 1

                for j in range(len(pred_boxes)) : 
                        if gt_poly[i].intersects(pred_poly[j]) :
                                if (pred_poly[j].intersection(gt_poly[i]).area / pred_poly[j].union(gt_poly[i]).area) > 0.5 :
                                        #inter_gt.append(j)
                                        #print(i, j, iou3d(pred_boxes[i], gt_boxes[j]), pred_classes[i], gt_classes[j])
                                        if gt_classes[i] == 1 :
                                                if pred_classes[j] == 0 : 
                                                        vehicles_true += 1
                                                        break
                                        elif gt_classes[i] == 2 :
                                                if pred_classes[j] == 1 : 
                                                        pedestr_true += 1
                                                        break
                                        elif gt_classes[i] == 4 : 
                                                if pred_classes[j] == 2 : 
                                                        cycl_true += 1
                                                        break
        try:
            recall_veh = vehicles_true / vehicles_gt
        except : 
            recall_veh = 0
            print("No vehicles in GT")
        try :
            recall_ped = pedestr_true / pedestr_gt
        except : 
            recall_ped = 0
            print("No pedestrians in GT")
        try :
            recall_cycl = cycl_true / cycl_gt
        except : 
            recall_cycl = 0
            print("No cyclists in GT")
        recall_3cls = (vehicles_true + pedestr_true + cycl_true) / (vehicles_gt + pedestr_gt + cycl_gt)
        recall_4cls = (vehicles_true + pedestr_true + cycl_true) / len(gt_classes)

        print("In GT there are ", vehicles_gt, " vehicles, ", pedestr_gt, " dedestrians, ", cycl_gt, " cyclists")
        print("Recall_vehicles = ", recall_veh, " recall_pedestrians = ", recall_ped, " recall_cyclists = ", recall_cycl)
        print("Recall for 3 classes = ", recall_3cls, "Recall for 4 classes = ", recall_4cls)

        return recall_4cls



def throw_false(class1_results, class1_idx, class1_false_list, class2_results, class2_idx, class2_false_list, box3d_lidar, scores, label_preds) : 
    pred_labels_copy = label_preds

    my_box3d_lidar = list([]) #torch.Tensor(np.array([]))
    my_scores = list([]) #torch.Tensor(np.array([]))
    my_label_preds = list([]) #torch.Tensor(np.array([]))

    for index in range(len(pred_labels_copy)) : 
        counter = 0
        if (index in class1_false_list) or (index in class2_false_list) :
            counter+=1
        if (index in class1_idx) and (class1_results[class1_idx.index(index)] % 2 == 1) : 
            counter+=1
        if (index in class2_idx) and (class2_results[class2_idx.index(index)] % 2 == 1) : 
            counter+=1
        if counter == 0 : 
            my_box3d_lidar.append(box3d_lidar[index])
            my_scores.append(scores[index])
            my_label_preds.append(label_preds[index])
    if len(my_box3d_lidar) > 0 :
        my_box3d_lidar = torch.stack(my_box3d_lidar) #torch.Tensor(np.array([]))
        my_scores = torch.stack(my_scores) #torch.Tensor(np.array([]))
        my_label_preds = torch.stack(my_label_preds) #torch.Tensor(np.array([]))
    elif len(my_box3d_lidar) == 0:
        my_box3d_lidar = torch.unsqueeze(box3d_lidar[0],0)
        my_scores = torch.unsqueeze(scores[0],0)
        my_label_preds = torch.unsqueeze(label_preds[0],0)
    return my_box3d_lidar, my_scores, my_label_preds

def throw_false_4cls(class_results, class_idx, class_false_list, box3d_lidar, scores, label_preds) : 
    pred_labels_copy = label_preds

    my_box3d_lidar = list([]) #torch.Tensor(np.array([]))
    my_scores = list([]) #torch.Tensor(np.array([]))
    my_label_preds = list([]) #torch.Tensor(np.array([]))

    for index in range(len(pred_labels_copy)) : 
        counter = 0
        if (index in class_false_list) :
            counter+=1
        if (index in class_idx) and (class_results[class_idx.index(index)] % 2 == 1 or class_results[class_idx.index(index)] % 2 == 3) : 
            counter+=1
        if counter == 0 : 
            my_box3d_lidar.append(box3d_lidar[index])
            my_scores.append(scores[index])
            my_label_preds.append(label_preds[index])
    if len(my_box3d_lidar) > 0 :
        my_box3d_lidar = torch.stack(my_box3d_lidar) #torch.Tensor(np.array([]))
        my_scores = torch.stack(my_scores) #torch.Tensor(np.array([]))
        my_label_preds = torch.stack(my_label_preds) #torch.Tensor(np.array([]))
    elif len(my_box3d_lidar) == 0:
        my_box3d_lidar = torch.unsqueeze(box3d_lidar[0],0)
        my_scores = torch.unsqueeze(scores[0],0)
        my_label_preds = torch.unsqueeze(label_preds[0],0)
    return my_box3d_lidar, my_scores, my_label_preds

def save_pred(pred, root):
    with open(os.path.join(root, "prediction.pkl"), "wb") as f:
        pickle.dump(pred, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work_dir", required=True, help="the dir to save logs and models")
    parser.add_argument(
        "--checkpoint", help="the dir to checkpoint which the model read from"
    )
    parser.add_argument(
        "--txt_result",
        type=bool,
        default=False,
        help="whether to save results to standard KITTI format of txt type",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--speed_test", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--testset", action="store_true")

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args

def main():

    #model_vehicles, model_pedestrians, device_PAConv = define_models()
    model_4cls, device_PAConv = define_model_4cls()
    my_time = list([])
    time_pcd = list([])
    time_crop = list([])
    time_cls1 = list([0,0])
    time_cls2 = list([])
    time_throw = list([])
    time_box = list([])

    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.local_rank = args.local_rank

    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

        cfg.gpus = torch.distributed.get_world_size()
    else:
        cfg.gpus = args.gpus

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info("Distributed testing: {}".format(distributed))
    logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    if args.testset:
        print("Use Test Set")
        dataset = build_dataset(cfg.data.test)
    else:
        print("Use Val Set")
        dataset = build_dataset(cfg.data.val)

    data_loader = build_dataloader(
        dataset,
        batch_size=cfg.data.samples_per_gpu if not args.speed_test else 1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

    # put model on gpus
    if distributed:
        model = apex.parallel.convert_syncbn_model(model)
        model = DistributedDataParallel(
            model.cuda(cfg.local_rank),
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank,
            # broadcast_buffers=False,
            find_unused_parameters=True,
        )
    else:
        # model = fuse_bn_recursively(model)
        model = model.cuda()

    model.eval()
    mode = "val"

    logger.info(f"work dir: {args.work_dir}")
    if cfg.local_rank == 0:
        prog_bar = torchie.ProgressBar(len(data_loader.dataset) // cfg.gpus)

    detections = {}
    cpu_device = torch.device("cpu")

    start = time.time()

    start = int(len(dataset) / 3)
    end = int(len(dataset) * 2 /3)

    time_start = 0 
    time_end = 0 

    counter = 0

    data_list = []

    for i, data_batch in enumerate(data_loader) : 
        #print(data_batch['metadata'][0]['token'])
        data_list.append(data_batch)

    #for i, data_batch in enumerate(data_loader):
    for i in range(len(data_list)) : 
        data_batch = next((x for x in data_list if x['metadata'][0]['token'] == 'seq_0_frame_'+str(i)+'.pkl'), None)
        #print(data_batch['metadata'][0]['token'], i)
        counter += 1

        if i == start:
            torch.cuda.synchronize()
            time_start = time.time()

        if i == end:
            torch.cuda.synchronize()
            time_end = time.time()

        with torch.no_grad():
            outputs = batch_processor(
                model, data_batch, train_mode=False, local_rank=args.local_rank,
            )

        for output in outputs:

            token = output["metadata"]["token"]
            for k, v in output.items():
                if k not in [
                    "metadata",
                ]:
                    output[k] = v.to(cpu_device)
            
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            ### my code starts here
            
            starter_pcd, ender_pcd = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter_pcd.record()
            pc_name = os.path.join('./data/Waymo/val/lidar', token)
            points = pickle.load(open(pc_name, 'rb'))['lidars']['points_xyz']
            #points = data_batch['points'][:,1:4]
            points_tensor = torch.cuda.FloatTensor(points)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            #pc_name = os.path.join('./data/Waymo/val/lidar', token)
            #points = pickle.load(open(pc_name, 'rb'))['lidars']['points_xyz']
            ender_pcd.record()
            torch.cuda.synchronize()
            time_pcd.append(starter_pcd.elapsed_time(ender_pcd))
            
            #anno_name = os.path.join('./data/Waymo/val/annos', token)
            #anno_objects = pickle.load(open(anno_name, 'rb'))['objects']
            #gt_boxes = list([])
            #gt_classes = list([])

            #for item in anno_objects : 
                #print(item['box'].shape)
             #   gt_boxes.append(box_center_to_corner_gt(item['box'], parameter=0.15))
             #   gt_classes.append(item['label'])
                #print(item['label'])
            
            #print(len(gt_boxes), len(gt_classes), gt_boxes[2].shape)

            """
            lines = [
                [0, 1],
                [0, 2],
                [1, 3],
                [2, 3],
                [4, 5],
                [4, 6],
                [5, 7],
                [6, 7],
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
                [0, 3],
                [1, 2],
                [4, 7],
                [5, 6]
            ]
            """
            """
            visual = o3d.visualization.Visualizer()
            visual.create_window()
            visual.get_render_option().line_width = 2
            visual.get_render_option().point_size = 1
            visual.add_geometry(pcd)
             
            
            colors_gt = [[1, 0, 0] for i in range(len(lines))]
            colors_pred= [[0, 1, 0] for i in range(len(lines))]
            """
            """
            for i in range(len(gt_boxes)) : 

                line_set_pred = o3d.geometry.LineSet(
                    points = o3d.utility.Vector3dVector(gt_boxes[i]),
                    lines=o3d.utility.Vector2iVector(lines),
                )
                line_set_pred.colors = o3d.utility.Vector3dVector(colors_gt)
                visual.add_geometry(line_set_pred)
            """
            
            
            pred_labels = output['label_preds']
            pred_scores = output['scores']
            pred_boxes_tensor = output['box3d_lidar']

            
            starter_box, ender_box = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter_box.record()
            boxes_dict = get_cropped_boxes_gpu(pred_boxes_tensor, pred_labels, points_tensor, pred_scores,parameter=0.15, pcd_gpu=pcd, count=counter)
            #pred_boxes = get_boxes_from_pred_frame({'box3d_lidar' : output['box3d_lidar']}, parameter=0.15)
            ender_box.record()
            torch.cuda.synchronize()
            time_box.append(starter_box.elapsed_time(ender_box))
            
            """
            for i in range(len(pred_boxes)) : 
                
                line_set_pred = o3d.geometry.LineSet(
                    points = o3d.utility.Vector3dVector(pred_boxes[i]),
                    lines=o3d.utility.Vector2iVector(lines),
                )
                if pred_scores[i] < 0.4 : 
                    visual.add_geometry(line_set_pred)
                else : 
                    line_set_pred.colors = o3d.utility.Vector3dVector(colors_pred)
                    visual.add_geometry(line_set_pred)
            """
            
            starter_crop, ender_crop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter_crop.record()
            #boxes_dict = get_cropped_boxes(output, points, pred_scores)
            #boxes_dict = get_cropped_boxes(pred_boxes, pred_labels, pcd, pred_scores)
            ender_crop.record()
            torch.cuda.synchronize()
            time_crop.append(starter_crop.elapsed_time(ender_crop))
                        

            #starter_cls1, ender_cls1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            #starter_cls1.record()
            #class1_results, class1_idx, class1_false_list = PAConv_test(boxes_dict, 'vehicle', 30, model_vehicles, device_PAConv)
            #ender_cls1.record()
            #torch.cuda.synchronize()
            #time_cls1.append(starter_cls1.elapsed_time(ender_cls1))
            #print(len(class1_results), len(class1_idx), len(class1_false_list))
            #print("class1_false_list", class1_results, class1_idx)

            starter_cls2, ender_cls2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter_cls2.record()
            class_results, class_idx, class_false_list = PAConv_test_4cls(boxes_dict, 30, model_4cls, device_PAConv)
            #class2_results, class2_idx, class2_false_list = PAConv_test(boxes_dict, 'pedestrian', 30, model_pedestrians, device_PAConv)
            ender_cls2.record()
            torch.cuda.synchronize()
            time_cls2.append(starter_cls2.elapsed_time(ender_cls2))
            #print(len(class2_results), len(class2_idx), len(class2_false_list))
            #print("class2_false_list", class2_results, class2_idx)

            #print(len(output['box3d_lidar']), " objects in frame")

            starter_throw, ender_throw = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter_throw.record()
            my_box3d_lidar, my_scores, my_label_preds = throw_false_4cls(class_results, class_idx, class_false_list, output['box3d_lidar'], output['scores'], output['label_preds'])
            #my_box3d_lidar, my_scores, my_label_preds = throw_false(class1_results, class1_idx, class1_false_list, class2_results, class2_idx, class2_false_list, output['box3d_lidar'], output['scores'], output['label_preds'])
            ender_throw.record()
            torch.cuda.synchronize()
            time_throw.append(starter_throw.elapsed_time(ender_throw))

            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            my_time.append(curr_time)
            #print(len(output['box3d_lidar']))

            output['box3d_lidar'] = my_box3d_lidar
            output['scores'] = my_scores
            output['label_preds'] = my_label_preds
            #print(len(output['box3d_lidar']))
        
            #visual.capture_screen_image('./data/screenshots/4_cls_nusc_' + token[0:-4] + '.jpg', do_render=True)
            #visual.destroy_window()
        
            ### my code ends here

            detections.update(
                {token: output,}
            )
            if args.local_rank == 0:
                prog_bar.update()

    synchronize()

    all_predictions = all_gather(detections)

    print("\n Total time per frame: ", (time_end -  time_start) / (end - start))
    print("\n Time for classification module per frame: ", np.mean(my_time))
    print("\n mean time pcd ", np.mean(time_pcd))
    print("\n mean time box ", np.mean(time_box))
    #print("\n mean time crop ", np.mean(time_crop))
    print("\n mean time cls1 ", np.mean(time_cls1))
    print("\n mean time cls2 ", np.mean(time_cls2))
    print("\n mean time throw ", np.mean(time_throw))

    if args.local_rank != 0:
        return

    predictions = {}
    for p in all_predictions:
        predictions.update(p)

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    save_pred(predictions, args.work_dir)

    result_dict, _ = dataset.evaluation(copy.deepcopy(predictions), output_dir=args.work_dir, testset=args.testset)

    if result_dict is not None:
        for k, v in result_dict["results"].items():
            print(f"Evaluation {k}: {v}")

    if args.txt_result:
        assert False, "No longer support kitti"

if __name__ == "__main__":
    main()
