import numpy as np
import pickle
from scripts.crop_dataset import crop_points
from scripts.Dataset import get_boxes_from_pred_frame, get_boxes_from_waymo_frame, GtLoader, get_point_cloud_from_gt_frame
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from shapely.geometry import Polygon
import os
import open3d as o3d
from open3d import *

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

def get_cropped_boxes(path_to_pred_file: str, path_to_gt_pkl_dir: str, path_to_gt_tfrecord_file: str) : 
        gt_frame_list = os.listdir(path_to_gt_pkl_dir)
        seq_number = gt_frame_list[5][4]
        detections_file = open(path_to_pred_file, 'rb')
        detections = pickle.load(detections_file)
        detections_file.close()

        gt_file = path_to_gt_tfrecord_file
        gt_object = GtLoader()
        gt_object.load(gt_file)

        cropped_pointclouds = list([])
        cropped_classes = list([])

        for frame_index in range(len(gt_frame_list)) : 
                frame_name = 'seq_'+ str(seq_number) +'_frame_'+ str(frame_index) +'.pkl'
                pred_frame = detections['seq_'+ str(seq_number) +'_frame_'+ str(frame_index) +'.pkl']
                pred_labels = pred_frame['classes']
                print(pred_frame.keys())
                pred_boxes = get_boxes_from_pred_frame({'boxes' : pred_frame['boxes']}, parameter=0.15).astype("float64") 
                pc_name = os.path.join(path_to_gt_pkl_dir, frame_name)
                points_xyz = pickle.load(open(pc_name, 'rb'))['lidars']['points_xyz']

                gt_frame = gt_object.next_frame()
                #gt_points = get_point_cloud_from_gt_frame(gt_frame)
                gt_boxes, gt_classes = get_boxes_from_waymo_frame(gt_frame, parameter=0.15)

                print("frame index ", frame_index)
                #print('pred_boxes: ', len(pred_boxes), ' pred_classes: ', len(pred_labels), ' gt_boxes ' , len(gt_boxes), ' gt_classes ', len(gt_classes))

                boxes_list_pred, classes_list_pred, boxes_list_gt, classes_list_gt = check_classes(pred_boxes, pred_labels, gt_boxes, gt_classes)
                #print(len(boxes_list_pred), len(classes_list_pred), len(boxes_list_gt), len(classes_list_gt))
                #print('pred_list: ', len(pred_list), 'classes_list: ', len(classes_list))

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_xyz)
                
                for i in range(len(boxes_list_pred)) : 
                        points = o3d.utility.Vector3dVector(pred_boxes[boxes_list_pred[i]])
                        oriented_bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(points)
                        one_crop = pcd.crop(oriented_bounding_box)
                        cropped_pointclouds.append(np.asarray(one_crop.points))
                
                cropped_classes.extend(classes_list_pred)

                for i in range(len(boxes_list_gt)) :
                        points = o3d.utility.Vector3dVector(gt_boxes[boxes_list_gt[i]])
                        oriented_bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(points)
                        one_crop = pcd.crop(oriented_bounding_box)
                        cropped_pointclouds.append(np.asarray(one_crop.points))
                cropped_classes.extend(classes_list_gt)

                print(len(cropped_pointclouds), len(cropped_classes))

        print("finally ", len(cropped_pointclouds), len(cropped_classes))
        with open('../Results12/points.data', 'wb') as f:
                pickle.dump(cropped_pointclouds, f)
        with open('../Results12/classes.data', 'wb') as f:
                pickle.dump(cropped_classes, f)

get_cropped_boxes('../Results12/detections.pkl', '../Results12/lidar', '../Results12/segment-14624061243736004421_1840_000_1860_000_with_camera_labels.tfrecord')
