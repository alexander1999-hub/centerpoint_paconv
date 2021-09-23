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
#import open3d as o3d
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

voxel_generator = None 
model = None 
device = None 

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
    #loaded_waymo_classes = list([])

    false_list = list([])
    index_list = list([])
    for box_index in range(len(frame_dict)) : 
        if class_name == 'vehicle' : 
            if frame_dict[box_index][1] == 0:
                index_list.append(box_index)
                #loaded_waymo_classes.append(frame_dict[box_index][1])
                if frame_dict[box_index][0].shape[0] >= 2 :
                    loaded_waymo_pointclouds.append(transform(100, frame_dict[box_index][0]))
                else : 
                    loaded_waymo_pointclouds.append(np.zeros((100, 3)))
                    false_list.append(box_index)
        if class_name == 'pedestrian' : 
            if frame_dict[box_index][1] == 1:
                index_list.append(box_index)
                #loaded_waymo_classes.append(frame_dict[box_index][1])
                if frame_dict[box_index][0].shape[0] >= 2 :
                    loaded_waymo_pointclouds.append(transform(100, frame_dict[box_index][0]))
                else : 
                    loaded_waymo_pointclouds.append(np.zeros((100, 3)))
                    false_list.append(box_index)
    #zipped_data = list(zip(loaded_waymo_pointclouds, loaded_waymo_classes))
    temp = list(zip(loaded_waymo_pointclouds, index_list))
    batches = divide_chunks(temp, batch_size), false_list


    return batches

def get_parser():
    #parser = argparse.ArgumentParser(description='3D Object Classification')
    #parser.add_argument('--config', type=str, default='config/dgcnn_paconv.yaml', help='config file')
    #parser.add_argument('opts', help='see config/dgcnn_paconv.yaml for all options', default=None, nargs=argparse.REMAINDER)
    #args = parser.parse_args()
    #assert args.config is not None
    cfg = load_cfg_from_cfg_file('config/dgcnn_paconv_train.yaml')
    #if args.opts is not None:
    #    cfg = merge_cfg_from_list(cfg, args.opts)

    cfg['manual_seed'] = cfg.get('manual_seed', 0)
    cfg['workers'] = cfg.get('workers', 6)
    return cfg

def PAConv_test(frame_dict, class_name, batch_size):
    #test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points, pt_norm=False),
                             #batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    #device = torch.device("cuda" if args.cuda else "cpu")
    device = torch.device('cuda:1')
    torch.cuda.set_device(device)

    args = get_parser()

    # Try to load models:
    from model.my_DGCNN_PAConv import PAConv
    model = PAConv(args).to(device, dtype=torch.float)
    if class_name == 'vehicle' :
        model.load_state_dict(torch.load("checkpoints/dgcnn_paconv_train/vehicles_best_model_29_07.t7"))
    if class_name == 'pedestrian' :
        model.load_state_dict(torch.load("checkpoints/dgcnn_paconv_train/predestrians_best_model_29_07_21..t7"))
    model = model.eval()
    test_pred = []
    timings = list([])
    index_list = []
    data_list_zipped, false_list = load_my_waymo_data(class_name, frame_dict, batch_size)
    #starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for batch in data_list_zipped:
        pointcloud,index = zip(*batch)
        index = list(index)
        index_list.append(index)
        pointcloud = np.stack(list(batch))
        #label = np.stack(list(label))
        #pointcloud = np.array(waymo_data_test[figure_index:figure_index+custom_batch_size])
        #label = np.array(waymo_labels_test[figure_index:figure_index+custom_batch_size])[:np.newaxis]
        #figure_index+=custom_batch_size
    #for data, label in test_loader:
        #data, label = torch.tensor(pointcloud).to(device, dtype=torch.float), torch.tensor(label).to(device, dtype=torch.long)
        data = torch.tensor(pointcloud).to(device, dtype=torch.float)
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        with torch.no_grad():
            #starter.record()
            logits = model(data)
            #ender.record()
            #torch.cuda.synchronize()
            #curr_time = starter.elapsed_time(ender)
            #timings.append(curr_time)
        preds = logits.max(dim=1)[1]
        #test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    #print("Batch size ", str(batch_size), ", time for one batch ", np.mean(timings))
    #test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    #test_acc = metrics.accuracy_score(test_true, test_pred)
    #print(precision_recall_fscore_support(test_true, test_pred, average=None, labels=[0, 1]))
    #avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    #outstr = 'Test :: test acc: %.6f, test avg acc: %.6f' % (test_acc, avg_per_class_acc)
    #io.cprint(outstr)
    return test_pred, index_list, false_list

def get_cropped_boxes(detections, frame_points) : 
        cropped_pointclouds = list([])

        pred_frame = detections
        pred_labels = pred_frame['classes']
        pred_boxes = get_boxes_from_pred_frame({'boxes' : pred_frame['boxes']}, parameter=0).astype("float64") 
        #gt_frame = gt_object.next_frame()
        #gt_points = get_point_cloud_from_gt_frame(gt_frame)
        #gt_boxes, gt_classes = get_boxes_from_waymo_frame(gt_frame, parameter=0)

        #print("frame index ", frame_index)
        #print('pred_boxes: ', len(pred_boxes), ' pred_classes: ', len(pred_labels), ' gt_boxes ' , len(gt_boxes), ' gt_classes ', len(gt_classes))

        #boxes_list_pred, classes_list_pred = check_classes(pred_boxes, pred_labels, gt_boxes, gt_classes)
        #print(len(pred_boxes), len(pred_labels), len(boxes_list_pred), len(classes_list_pred))
        print(len(pred_boxes))

        pcd = o3d.geometry.PointCloud()
        #pcd.points = o3d.utility.Vector3dVector(gt_points)
        pcd.points = o3d.utility.Vector3dVector(frame_points)
        
        #for i in range(len(boxes_list_pred)) : 
        #    points = o3d.utility.Vector3dVector(pred_boxes[boxes_list_pred[i]])
        #    oriented_bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(points)
        #    one_crop = pcd.crop(oriented_bounding_box)
        #    cropped_pointclouds.append(np.asarray(one_crop.points))
    
        for i in range(len(pred_boxes)) : 
            points = o3d.utility.Vector3dVector(pred_boxes[i])
            oriented_bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(points)
            one_crop = pcd.crop(oriented_bounding_box)
            cropped_pointclouds.append(np.asarray(one_crop.points))
        
        #cropped_classes.extend(classes_list_pred)

        print(len(cropped_pointclouds))

        frame_dict = {}

        for i in range(len(pred_boxes)) : 
            frame_dict[i] = (cropped_pointclouds[i], pred_labels[i])
        
        return frame_dict


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
    #print(gt_frame_list)
    seq_number = gt_frame_list[5][4]

    #class_name1 = 'pedestrian'
    #class_name2 = 'vehicle'
    #batch_size = 25
    #path_to_gt_tfrecord_file = '../dataset_waymo/TFRecord/segment-17791493328130181905_1480_000_1500_000_with_camera_labels.tfrecord'

    #gt_file = path_to_gt_tfrecord_file
    #gt_object = GtLoader()
    #gt_object.load(gt_file)

    for frame_index in tqdm(range(len(gt_frame_list))):
        frame_name = 'seq_'+ str(seq_number) +'_frame_'+ str(frame_index) +'.pkl'
        if counter == args.num_frame:
            break
        else:
            counter += 1 

        pc_name = os.path.join(args.input_data_dir, frame_name)
        print(pc_name)
        #points = load_cloud_from_deecamp_file(pc_name)
        points = np.concatenate((pickle.load(open(pc_name, 'rb'))['lidars']['points_xyz'], pickle.load(open(pc_name, 'rb'))['lidars']['points_feature']), axis=1)
        points_xyz = pickle.load(open(pc_name, 'rb'))['lidars']['points_xyz']

        detections = process_example(points, args.fp16)
        #boxes_dict = get_cropped_boxes(detections, points_xyz)
        #class1_results, class1_idx, class1_false_list = PAConv_test(boxes_dict, class_name1, batch_size)
        #class2_results, class2_idx, class2_false_list = PAConv_test(boxes_dict, class_name2, batch_size)

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

    with open(os.path.join(args.output_dir, 'detections.pkl'), 'wb') as f:
        pickle.dump(pred_dicts, f)
