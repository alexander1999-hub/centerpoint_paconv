# modified from the single_inference.py by @muzi2045
#from spconv.utils import VoxelGenerator as VoxelGenerator
from det3d.core.input.voxel_generator import VoxelGenerator as VoxelGenerator
from det3d.datasets.pipelines.loading import read_single_waymo
from det3d.datasets.pipelines.loading import get_obj
from det3d.torchie.trainer import load_checkpoint
from det3d.models import build_detector
from det3d.torchie import Config
from tqdm import tqdm 
import numpy as np
import pickle 
import argparse
import torch
import time 
import os 
import random

from scripts.Dataset import get_boxes_from_pred_frame, get_boxes_from_waymo_frame, GtLoader, get_point_cloud_from_gt_frame
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import os
import open3d as o3d
from open3d import *
from util.util import cal_loss, IOStream, load_cfg_from_cfg_file, merge_cfg_from_list

from model.my_DGCNN_PAConv import PAConv

def get_parser():
    cfg = load_cfg_from_cfg_file('config/dgcnn_paconv_train.yaml')
    cfg['manual_seed'] = cfg.get('manual_seed', 0)
    cfg['workers'] = cfg.get('workers', 6)
    return cfg

voxel_generator = None 
model = None 
device = None 

def define_models():

    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

    args = get_parser()

    model_pedestrians =  PAConv(args).to(device, dtype=torch.float)
    model_vehicles =  PAConv(args).to(device, dtype=torch.float)
    
    model_vehicles.load_state_dict(torch.load("checkpoints/dgcnn_paconv_train/vehicles_12_08.t7"))
    model_pedestrians.load_state_dict(torch.load("checkpoints/dgcnn_paconv_train/pedestrians_12_08.t7"))
    model_vehicles = model_vehicles.eval()
    model_pedestrians = model_pedestrians.eval()

    return model_vehicles, model_pedestrians


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
    for box_index in range(len(frame_dict)) : 
        if class_name == 'vehicle' : 
            if frame_dict[box_index][1] == 0:
                index_list.append(box_index)
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

def load_my_waymo_data_4_class(frame_dict, batch_size): 

    loaded_waymo_pointclouds = list([])

    false_list = list([])
    index_list = list([])
    for box_index in range(len(frame_dict)) : 
        if frame_dict[box_index][1] == 0 or frame_dict[box_index][1] == 1:
            index_list.append(box_index)
            if frame_dict[box_index][0].shape[0] >= 2 :
                loaded_waymo_pointclouds.append(transform(100, frame_dict[box_index][0]))
            else : 
                loaded_waymo_pointclouds.append(np.zeros((100, 3)))
                false_list.append(box_index)
        """
        if class_name == 'pedestrian' : 
            if frame_dict[box_index][1] == 1:
                index_list.append(box_index)
                if frame_dict[box_index][0].shape[0] >= 2 :
                    loaded_waymo_pointclouds.append(transform(100, frame_dict[box_index][0]))
                else : 
                    loaded_waymo_pointclouds.append(np.zeros((100, 3)))
                    false_list.append(box_index)
        """
    #print("load data", len(index_list), len(loaded_waymo_pointclouds))
    temp = list(zip(loaded_waymo_pointclouds, index_list))
    batches = divide_chunks(temp, batch_size), false_list


    return batches

def check_classes(pred_boxes: np.array, pred_classes, gt_boxes: np.array, gt_classes) : 
        boxes_list_pred = list([])
        classes_list_pred = list([])

        pred_poly = []
        gt_poly = []

        print("pred classes ", len(pred_classes),  pred_classes)

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
                intersect_count = 0
                for j in range(len(gt_boxes)) : 
                        if pred_poly[i].intersects(gt_poly[j]) :
                                if (pred_poly[i].intersection(gt_poly[j]).area / pred_poly[i].union(gt_poly[j]).area) > 0.5 :
                                        intersect_count += 1
                                        if pred_classes[i] == 0 :
                                                if gt_classes[j] == 1 : 
                                                        classes_list_pred.append(0)
                                                else : 
                                                        classes_list_pred.append(1)
                                        elif pred_classes[i] == 1 :
                                                if gt_classes[j] == 2 : 
                                                        classes_list_pred.append(2)
                                                else : 
                                                        classes_list_pred.append(3)
                                        else : 
                                            classes_list_pred.append(4)
                                        break
                if intersect_count == 0 : 
                        if pred_classes[i] == 0 : 
                                classes_list_pred.append(1)
                        elif pred_classes[i] == 1 :
                                classes_list_pred.append(3)
                        else : 
                                classes_list_pred.append(4)
                boxes_list_pred.append(i)

        return boxes_list_pred, classes_list_pred

def PAConv_test(frame_dict, class_name, batch_size, model):

    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

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

        preds = logits.max(dim=1)[1]
        test_pred.append(preds.detach().cpu().numpy())

    test_pred = np.concatenate(test_pred)

    #print(class_name, len(test_pred), len(index_list), len(false_list))

    return test_pred, index_list, false_list

def get_cropped_boxes(detections, frame_points) : 
        cropped_pointclouds = list([])

        pred_frame = detections
        pred_labels = pred_frame['classes']
        print("pred labels ", len(pred_labels))
        pred_boxes = get_boxes_from_pred_frame({'boxes' : pred_frame['boxes']}, parameter=0).astype("float64") 

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(frame_points)
         
        for i in range(len(pred_boxes)) : 
            points = o3d.utility.Vector3dVector(pred_boxes[i])
            oriented_bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(points)
            one_crop = pcd.crop(oriented_bounding_box)
            cropped_pointclouds.append(np.asarray(one_crop.points))

        frame_dict = {}

        for i in range(len(pred_boxes)) : 
            frame_dict[i] = (cropped_pointclouds[i], pred_labels[i])
        
        return frame_dict, pred_boxes, pred_labels


def load_cloud_from_deecamp_file(pc_f):
        #logging.info('loading cloud from: {}'.format(pc_f))
        num_features = 4
        cloud = np.fromfile(pc_f, dtype=np.float32, count=-1).reshape([-1, num_features])
        # last dimension should be the timestamp.
        cloud = np.hstack((cloud, np.zeros([cloud.shape[0], 1])))
        return cloud

def initialize_model(args):
    global model, voxel_generator  
    cfg = Config.fromfile(args.config)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, map_location="cpu")
    # print(model)
    if args.fp16:
        print("cast model to fp16")
        model = model.half()

    model = model.cuda()
    model.eval()

    global device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    range = cfg.voxel_generator.range
    voxel_size = cfg.voxel_generator.voxel_size
    max_points_in_voxel = cfg.voxel_generator.max_points_in_voxel
    max_voxel_num = cfg.voxel_generator.max_voxel_num[1]
    voxel_generator = VoxelGenerator(
        voxel_size=voxel_size,
        point_cloud_range=range,
        max_num_points=max_points_in_voxel,
        max_voxels=max_voxel_num
    )
    return model 

def voxelization(points, voxel_generator):
    voxel_output = voxel_generator.generate(points)  
    voxels, coords, num_points = \
        voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']

    return voxels, coords, num_points  

def _process_inputs(points, fp16):
    voxels, coords, num_points = voxel_generator.generate(points)
    #voxels, coords, num_points = voxelization(points, voxel_generator)
    num_voxels = np.array([voxels.shape[0]], dtype=np.int32)
    grid_size = voxel_generator.grid_size
    coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values = 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    voxels = torch.tensor(voxels, dtype=torch.float32, device=device)
    coords = torch.tensor(coords, dtype=torch.int32, device=device)
    num_points = torch.tensor(num_points, dtype=torch.int32, device=device)
    num_voxels = torch.tensor(num_voxels, dtype=torch.int32, device=device)

    if fp16:
        voxels = voxels.half()

    inputs = dict(
            voxels = voxels,
            num_points = num_points,
            num_voxels = num_voxels,
            coordinates = coords,
            shape = [grid_size]
        )

    return inputs 

def run_model(points, fp16=False):
    with torch.no_grad():
        data_dict = _process_inputs(points, fp16)
        outputs = model(data_dict, return_loss=False)[0]

    return {'boxes': outputs['box3d_lidar'].cpu().numpy(),
        'scores': outputs['scores'].cpu().numpy(),
        'classes': outputs['label_preds'].cpu().numpy()}

def process_example(points, fp16=False):
    output = run_model(points, fp16)

    assert len(output) == 3
    assert set(output.keys()) == set(('boxes', 'scores', 'classes'))
    num_objs = output['boxes'].shape[0]
    assert output['scores'].shape[0] == num_objs
    assert output['classes'].shape[0] == num_objs

    return output    

def throw_false(class1_results, class1_idx, class1_false_list, class2_results, class2_idx, class2_false_list, pred_labels) : 
    pred_labels_copy = pred_labels
    for index in range(len(pred_labels)) : 
        if (index in class1_false_list) or (index in class2_false_list) :
            #print("success")
            pred_labels_copy[index] = 5 
        if (index in class1_idx) and (class1_results[class1_idx.index(index)] % 2 == 1) : 
            pred_labels_copy[index] = 5
            #print("success1")
        if (index in class2_idx) and (class2_results[class2_idx.index(index)] % 2 == 1) : 
            pred_labels_copy[index] = 5
            #print("success2")
    return pred_labels_copy

def true_amount(classes) : 

    true_count = 0
    false_count = 0

    for item in classes :
        if (item != 4) and (item != 5) : 
            if (item % 2) == 0 : 
                true_count += 1
            elif (item % 2) == 1 : 
                false_count += 1
    return true_count, false_count, true_count / (true_count + false_count)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CenterPoint")
    parser.add_argument("config", help="path to config file")
    parser.add_argument(
        "--checkpoint", help="the path to checkpoint which the model read from", default=None, type=str
    )
    parser.add_argument('--input_data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--threshold', default=0.5)
    parser.add_argument('--visual', action='store_true', default=False)
    parser.add_argument("--online", action='store_true', default=False)
    parser.add_argument('--num_frame', default=-1, type=int)
    args = parser.parse_args()

    print("Please prepare your point cloud in waymo format and save it as a pickle dict with points key into the {}".format(args.input_data_dir))
    print("One point cloud should be saved in one pickle file.")
    print("Download and save the pretrained model at {}".format(args.checkpoint))

    # Run any user-specified initialization code for their submission.
    model = initialize_model(args)

    latencies = []
    visual_dicts = []
    pred_dicts = {}
    counter = 0 

    gt_frame_list = os.listdir(args.input_data_dir)
    seq_number = gt_frame_list[5][4]

    batch_size = 25
    path_to_gt_tfrecord_file = '../Results12/segment-14624061243736004421_1840_000_1860_000_with_camera_labels.tfrecord'

    gt_file = path_to_gt_tfrecord_file
    gt_object = GtLoader()
    gt_object.load(gt_file)

    for frame_index in tqdm(range(len(gt_frame_list))):

        frame_name = 'seq_'+ str(seq_number) +'_frame_'+ str(frame_index) +'.pkl'
        if counter == args.num_frame:
            break
        else:
            counter += 1 

        pc_name = os.path.join(args.input_data_dir, frame_name)
        #print(pc_name)
        #points = load_cloud_from_deecamp_file(pc_name)
        points = np.concatenate((pickle.load(open(pc_name, 'rb'))['lidars']['points_xyz'], pickle.load(open(pc_name, 'rb'))['lidars']['points_feature']), axis=1)
        #points_xyz = pickle.load(open(pc_name, 'rb'))['lidars']['points_xyz']

        #gt_frame = gt_object.next_frame()
        #gt_boxes, gt_classes = get_boxes_from_waymo_frame(gt_frame, parameter=0.15)

        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_xyz)

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
        list_for_vis1 = []
        list_for_vis.append(points_xyz)
        list_for_vis1.append(points_xyz)

        for i in range(len(gt_boxes)) :
                line_set_gt = o3d.geometry.LineSet(
                        points=o3d.utility.Vector3dVector(gt_boxes[i]),
                        lines=o3d.utility.Vector2iVector(lines),
                )
                line_set_gt.colors = o3d.utility.Vector3dVector(colors_gt)
                list_for_vis.append(gt_boxes[i])
                list_for_vis1.append(gt_boxes[i])

        model_vehicles, model_pedestrians = define_models()

        #starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        #starter.record()
        """
        detections = process_example(points, args.fp16)
        #ender.record()
        """
        boxes_dict, pred_boxes, pred_labels = get_cropped_boxes(detections, points_xyz)

        class_results, class_idx, class_false_list = PAConv_test(boxes_dict, batch_size, model_vehicles)
        print(len(class_results), len(class_idx), len(class_false_list))
        print("class_false_list ", class_results, class_idx)

        class2_results, class2_idx, class2_false_list = PAConv_test(boxes_dict, 'pedestrian', batch_size, model_pedestrians)
        print(len(class2_results), len(class2_idx), len(class2_false_list))
        print("class2_false_list", class2_results, class2_idx)

        boxes_list_pred, classes_list_pred = check_classes(pred_boxes, pred_labels, gt_boxes, gt_classes)

        print("classes list pred ", len(classes_list_pred), classes_list_pred)
        print("old classes true amount ", true_amount(classes_list_pred))
        for i in range(len(pred_boxes)) :
                if classes_list_pred[i] != 4 and classes_list_pred[i] != 5 :
                        line_set_pred = o3d.geometry.LineSet(
                                points=o3d.utility.Vector3dVector(pred_boxes[i]),
                                lines=o3d.utility.Vector2iVector(lines),
                        )
                        line_set_pred.colors = o3d.utility.Vector3dVector(colors_pred)
                        list_for_vis.append(pred_boxes[i]) 

        new_classes_list_pred = throw_false(class1_results, class1_idx, class1_false_list, class2_results, class2_idx, class2_false_list, classes_list_pred)
        print("new_classes_list_pred", new_classes_list_pred)
        print("new classes ture amount ", true_amount(new_classes_list_pred))
        for i in range(len(pred_boxes)) :
                if new_classes_list_pred[i] != 4 and new_classes_list_pred[i] != 5 :
                        line_set_pred = o3d.geometry.LineSet(
                                points=o3d.utility.Vector3dVector(pred_boxes[i]),
                                lines=o3d.utility.Vector2iVector(lines),
                        )
                        line_set_pred.colors = o3d.utility.Vector3dVector(colors_pred)
                        list_for_vis1.append(pred_boxes[i]) 
        
        
        with open(os.path.join(args.output_dir, 'detections2_test.pkl'), 'wb') as f:
            pickle.dump(list_for_vis, f)

        with open(os.path.join(args.output_dir, 'detections3_test.pkl'), 'wb') as f:
            pickle.dump(list_for_vis1, f)
        """

        #torch.cuda.synchronize()
        #curr_time = starter.elapsed_time(ender)
        #print("Elapsed time: ", curr_time)

        #print("YESS")

        if args.visual and args.online:
            pcd = o3d.geometry.PointCloud()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])

            visual = [pcd]
            num_dets = detections['scores'].shape[0]
            visual += plot_boxes(detections, args.threshold)

            o3d.visualization.draw_geometries(visual)
        elif args.visual:
            visual_dicts.append({'points': points, 'detections': detections})

        pred_dicts.update({frame_name: detections})
        #print(type(pred_dicts))

    if args.visual:
        with open(os.path.join(args.output_dir, 'visualization.pkl'), 'wb') as f:
            pickle.dump(visual_dicts, f)

    with open(os.path.join(args.output_dir, 'detections_voxelnet.pkl'), 'wb') as f:
        pickle.dump(pred_dicts, f)
