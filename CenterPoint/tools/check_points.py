import numpy as np
import pickle
from scripts.crop_dataset import crop_points
from scripts.Dataset import get_boxes_from_pred_frame, get_boxes_from_waymo_frame, GtLoader, get_point_cloud_from_gt_frame
from scripts.bbox_iou import box3d_iou
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from shapely.geometry import Polygon
from scipy.spatial.transform import Rotation as R
import copy
import matplotlib.pyplot as plt
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


def text_3d(text, pos, direction=None, degree=0.0, density=10, font='/usr/share/fonts/truetype/freefont/FreeMono.ttf', font_size=100):
        """
        Generate a 3D text point cloud used for visualization.
        :param text: content of the text
        :param pos: 3D xyz position of the text upper left corner
        :param direction: 3D normalized direction of where the text faces
        :param degree: in plane rotation of text
        :param font: Name of the font - change it according to your system
        :param font_size: size of the font
        :return: o3d.geoemtry.PointCloud object
        """
        if direction is None:
                direction = (0., 0., 1.)

        from PIL import Image, ImageFont, ImageDraw
        from pyquaternion import Quaternion

        font_obj = ImageFont.truetype(font, font_size)
        font_dim = font_obj.getsize(text)

        img = Image.new('RGB', font_dim, color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
        img = np.asarray(img)
        img_mask = img[:, :, 0] < 128
        indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

        pcd = o3d.geometry.PointCloud()
        pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
        pcd.points = o3d.utility.Vector3dVector(indices / 100.0)

        raxis = np.cross([0.0, 0.0, 1.0], direction)
        if np.linalg.norm(raxis) < 1e-6:
                raxis = (0.0, 0.0, 1.0)
        trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
                Quaternion(axis=direction, degrees=degree)).transformation_matrix
        trans[0:3, 3] = np.asarray(pos)
        pcd.transform(trans)
        return pcd

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
                
                gt_frame = gt_object.next_frame()
                

                if frame_index == 150 :
                        pred_labels = pred_frame['classes']
                        pred_boxes = get_boxes_from_pred_frame({'boxes' : pred_frame['boxes']}, parameter=0.15).astype("float64") 

                        gt_points = get_point_cloud_from_gt_frame(gt_frame)
                        gt_boxes, gt_classes = get_boxes_from_waymo_frame(gt_frame, parameter=0.15)

                        pc_name = os.path.join(path_to_gt_pkl_dir, frame_name)
                        points_xyz = pickle.load(open(pc_name, 'rb'))['lidars']['points_xyz']

                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(points_xyz)
                        colors_pcd = [[0, 0, 1] for i in range(len(points_xyz))]
                        pcd.colors = o3d.utility.Vector3dVector(colors_pcd)
                        #pcd_10 = text_3d('Test-10mm', pos=(pred_boxes[5][0]+pred_boxes[5][1]+pred_boxes[5][2]+pred_boxes[5][3])/4, font_size=100, density=10)

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
                        colors_gt = [[1, 0, 0] for i in range(len(lines))]
                        colors_pred = [[0, 1, 0] for i in range(len(lines))]
                        list_for_vis = []
                        #list_for_vis.append(pcd)
                        #colors_10_pcd = [[1, 0, 0] for i in range(len(pcd_10.points))]
                        #pcd_10.colors = o3d.utility.Vector3dVector(colors_10_pcd)
                        #list_for_vis.append(pcd_10)

                        
                        for i in range(len(pred_boxes)) :
                                line_set_gt = o3d.geometry.LineSet(
                                        points=o3d.utility.Vector3dVector(pred_boxes[i]),
                                        lines=o3d.utility.Vector2iVector(lines),
                                )
                                line_set_gt.colors = o3d.utility.Vector3dVector(colors_gt)
                                #list_for_vis.append(line_set_gt)     
                        #o3d.visualization.draw_geometries(list_for_vis)
                        

                        boxes_list_pred, classes_list_pred, boxes_list_gt, classes_list_gt = check_classes(pred_boxes, pred_labels, gt_boxes, gt_classes)

                        """
                        for i in range(len(boxes_list_pred)) :
                                if classes_list_pred[i] == 1 :
                                        line_set_gt = o3d.geometry.LineSet(
                                                points=o3d.utility.Vector3dVector(pred_boxes[boxes_list_pred[i]]),
                                                lines=o3d.utility.Vector2iVector(lines),
                                        )
                                        line_set_gt.colors = o3d.utility.Vector3dVector(colors_pred)
                                        list_for_vis.append(line_set_gt)

                                        
                                        points = o3d.utility.Vector3dVector(pred_boxes[boxes_list_pred[i]])
                                        oriented_bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(points)
                                        one_crop = pcd.crop(oriented_bounding_box)
                                        list_for_vis.append(one_crop)
                                        #cropped_pointclouds.append(np.asarray(one_crop.points))     
                                        #o3d.visualization.draw_geometries(list_for_vis)

                        for i in range(len(boxes_list_gt)) :
                                if classes_list_gt[i] == 0 :
                                        line_set_gt = o3d.geometry.LineSet(
                                                points=o3d.utility.Vector3dVector(gt_boxes[boxes_list_gt[i]]),
                                                lines=o3d.utility.Vector2iVector(lines),
                                        )
                                        line_set_gt.colors = o3d.utility.Vector3dVector(colors_gt)
                                        list_for_vis.append(line_set_gt)

                                        points = o3d.utility.Vector3dVector(gt_boxes[boxes_list_gt[i]])
                                        oriented_bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(points)
                                        one_crop = pcd.crop(oriented_bounding_box) 
                                        list_for_vis.append(one_crop)    
                        o3d.visualization.draw_geometries(list_for_vis)
                        print(len(boxes_list_pred), len(classes_list_pred), len(boxes_list_gt), len(classes_list_gt))

                        

                        """
                        for i in range(len(gt_boxes)) :
                                line_set_gt = o3d.geometry.LineSet(
                                        points=o3d.utility.Vector3dVector(gt_boxes[i]),
                                        lines=o3d.utility.Vector2iVector(lines),
                                )
                                line_set_gt.colors = o3d.utility.Vector3dVector(colors_gt)
                                list_for_vis.append(line_set_gt)     

                        for i in range(len(pred_boxes)) :
                                line_set_gt = o3d.geometry.LineSet(
                                        points=o3d.utility.Vector3dVector(pred_boxes[i]),
                                        lines=o3d.utility.Vector2iVector(lines),
                                )
                                line_set_gt.colors = o3d.utility.Vector3dVector(colors_pred)
                                list_for_vis.append(line_set_gt)     
                        o3d.visualization.draw_geometries(list_for_vis)
                        

                        """

                        pred_poly = []
                        gt_poly = []

                        inter_pred = []
                        inter_gt = []

                        for i in range(len(pred_boxes)):
                                test_pcd = pred_boxes[i]
                                test_pcd = np.delete(test_pcd,2,1)
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
                        
                        for i in range(len(pred_poly)) :
                                inter_count = 0
                                for j in range(len(gt_poly)) : 
                                        if pred_poly[i].overlaps(gt_poly[j]) :
                                                if pred_poly[i].intersection(gt_poly[j]).area / pred_poly[i].union(gt_poly[j]).area > 0.3 :
                        """
                                                        
                                        
                        
        #list_for_vis.append(line_set_gt)
                        #o3d.visualization.draw_geometries(list_for_vis)
                        #one_crop = vol.crop_point_cloud(pcd)
                                #print(one_crop)
                        #print(len(cropped_pointclouds), len(classes_list))
                                                
                                        
                        #test_pcd = pred_boxes[90]
                        #test_pcd = np.delete(test_pcd,2,1)
                                        
                        #hull = ConvexHull(test_pcd)
                        #convex_hull_plot_2d(hull)

                        #fig = plt.figure()
                        #ax = fig.add_subplot(1,1,1)

                        #hull_indices = hull.vertices

                        # These are the actual points.
                        #hull_pts = test_pcd[hull_indices, :]
                        #result = list(map(tuple, hull_pts))
                        #p1 = Polygon(result)
                        #x,y = p1.exterior.xy
                        #plt.plot(x,y)
                        #plt.show()
                        """
                        for i in range(len(pred_boxes)) :
                                test_pcl = pred_boxes[i]
                                test_pcl[:,2] = 0 
                                line_set_gt = o3d.geometry.LineSet(
                                        points=o3d.utility.Vector3dVector(test_pcl),
                                        lines=o3d.utility.Vector2iVector(lines),
                                )
                                line_set_gt.colors = o3d.utility.Vector3dVector(colors_gt)
                                list_for_vis.append(line_set_gt)     
                        o3d.visualization.draw_geometries(list_for_vis)   """


get_cropped_boxes('../Results12/detections_voxelnet.pkl', '../Results12/lidar/', '../Results12/segment-14624061243736004421_1840_000_1860_000_with_camera_labels.tfrecord')