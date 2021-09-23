#!/usr/bin/python3
import rospy
import ros_numpy
import numpy as np
import torch
import time

from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from ros_marker import make_marker
from lidar_objects_msgs.msg import Object, ObjectArray
from dynamic_reconfigure.server import Server
from centerpoint_ros_node.cfg import ThresholdsConfig
from tf.transformations import quaternion_from_euler
import tf2_ros
from tf2_geometry_msgs import do_transform_pose
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

from det3d.models import build_detector
from det3d.torchie import Config
# from det3d.core.input.voxel_generator import VoxelGenerator
from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
from collections import defaultdict, deque
from geometry_msgs.msg import PoseStamped, PointStamped, Point
from scipy.spatial import distance

from tf.transformations import euler_from_quaternion
from Predictive_Tracker.PredictiveTracker import PredictiveTracker

FP16 = True

def get_xyz_points(msg, remove_nans=True, dtype=np.float):
    cloud_array = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]

    points = np.zeros(cloud_array.shape + (len(msg.fields),), dtype=dtype)
    points[..., 0] = cloud_array['x']
    points[..., 1] = cloud_array['y']
    points[..., 2] = cloud_array['z']
    return points

def remove_close(points, radius, radius_y=0):
    x_filt = np.abs(points[:, 0]) < radius
    y_filt = np.abs(points[:, 1]) < radius if radius_y == 0 else radius_y
    not_close = np.where(np.logical_not(np.logical_and(x_filt, y_filt)))
    points = points[not_close]
    return points

def get_annotations_indices(types, thresh, label_preds, scores):
            indexs = []
            annotation_indices = []
            for i in range(label_preds.shape[0]):
                if label_preds[i] == types:
                    indexs.append(i)
            for index in indexs:
                if scores[index] >= thresh:
                    annotation_indices.append(index)
            return annotation_indices

class ObjectTFConverter:
    def __init__(self):
        self.fixed_frame = rospy.get_param("~fixed_frame", "local_map")
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)

    def _get_transform(self, frame_id, stamp):
        try:
            transform = self.buffer.lookup_transform(self.fixed_frame, frame_id, stamp, rospy.Duration(0.12))
            return transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException) as e:
            rospy.logerr("ObjectTFConverter: %s", e)
            return None

    def transform_pointcloud(self, pointcloud):
        transform = self._get_transform(pointcloud.header.frame_id, pointcloud.header.stamp)
        tf_pointcloud = do_transform_cloud(pointcloud, transform)
        return tf_pointcloud

    def transform_pose(self, pose):
        transform = self._get_transform(pose.header.frame_id, pose.header.stamp)
        if transform:
            return do_transform_pose(pose, transform)
        else:
            return None

class TrajectoryFactory:
    def __init__(self, cfg):
        self.dict = defaultdict(deque)
        #self.age_dict = defaultdict()
        self.max_jump_distance = rospy.get_param("~max_jump_distance", 2)
        self.max_timeout = rospy.get_param("~max_object_timeout_sec", 5)
        self.max_stored_points = rospy.get_param("~max_stored_points", 100)
        self.history_size = rospy.get_param("~trajectory_size_sec", 10)
        self.period = rospy.get_param("~trajectory_period_sec", 1)
        # self.prediction_time_delta = rospy.get_param("~prediction_time_delta", 3.)
        if len(cfg.tasks) == 1:
            self.types_of_interest = [0, 1, 2]
        else:    
            self.types_of_interest = [0, 1, 7, 8]  # TODO unhardcode? May be different size for different types?
        self.previous_objects = ObjectArray()
        self.old_obj = Object()

    def append_frame(self, objects):
        for obj in objects.objects:
            if obj.label in self.types_of_interest:
                self.append(obj, objects.header.stamp.to_sec(), objects)
                obj.trajectory = self.get(obj.id, objects.header.frame_id)
        self.previous_objects = objects
        return objects

    def append(self, obj, stamp, objects):
        new_pt = np.array([
            obj.pose.position.x,
            obj.pose.position.y,
            obj.pose.position.z,
            stamp
        ], dtype=np.float64)
        if len(self.dict[obj.id]) != 0:
            dist = np.linalg.norm(new_pt[:-1] - self.dict[obj.id][-1][:-1])
            if dist > self.max_jump_distance:    
                #self.age_dict[obj.id] +=1
                self.dict.pop(obj.id)
                #self.age_dict.pop(obj.id)

        self.dict[obj.id].append(new_pt)
        #self.age_dict[obj.id] = 0
        if len(self.dict[obj.id]) > self.max_stored_points:
            self.dict[obj.id].popleft()

    def append_hidden_obj(self, objects, idd):
        for o in self.previous_objects.objects:
            if self.old_obj == idd:
                self.old_obj = o
                break
        new_obj = Object()
        new_obj.pose.orientation = self.old_obj.pose.orientation
        new_obj.pose.position.x = self.old_obj.pose.position.x + self.old_obj.velocity.linear.x  
        new_obj.pose.position.y = self.old_obj.pose.position.y + self.old_obj.velocity.linear.y
        new_obj.pose.position.z = self.old_obj.pose.position.z
        new_obj.size = self.old_obj.size
        new_obj.score = 0
        new_obj.id = self.old_obj.id
        new_obj.label = self.old_obj.label
        new_obj.velocity = self.old_obj.velocity
        objects.objects.append(new_obj)

    def _pt_from_xyzt_(self, xyzt, frame_id):
        pt = PointStamped()
        pt.header.frame_id = frame_id
        pt.header.stamp = rospy.Time.from_sec(xyzt[3])
        pt.point = Point(xyzt[0], xyzt[1], xyzt[2])
        return pt

    def get(self, track, frame_id):
        points = []
        points.append(self._pt_from_xyzt_(self.dict[track][0], frame_id))
        prev_t = self.dict[track][0][3]
        for xyzt in list(self.dict[track])[1: -1]:
            if xyzt[3] - prev_t >= self.period:
                points.append(self._pt_from_xyzt_(xyzt, frame_id))
                prev_t = xyzt[3]
        points.append(self._pt_from_xyzt_(self.dict[track][-1], frame_id))
        return points

    def clear_by_timeout(self, stamp):
        for k, v in list(self.dict.items()):
            if (stamp - (v[-1])[3]) > self.max_timeout:
                self.dict.pop(k)

class ProcessorROS:
    def __init__(self):
        # CenterPoint stuff
        self.points = None
        self.inputs = None

        # model loading
        config_path = rospy.get_param('~config_path')
        model_path = rospy.get_param('~model_path')
        self.cfg = Config.fromfile(config_path)

        inference_params = rospy.get_param('~inference_params')
        self.cfg.test_cfg['pc_range'] = inference_params['model']['reader']['pc_range'][:2]
        self.cfg.test_cfg['voxel_size'] = inference_params['model']['reader']['voxel_size'][:2]
        self.cfg.test_cfg['post_center_limit_range'] = inference_params['post_center_limit_range']
        self.cfg.model['reader']['pc_range'] = inference_params['model']['reader']['pc_range']
        self.cfg.model['reader']['voxel_size'] = inference_params['model']['reader']['voxel_size']
        self.range = inference_params['model']['reader']['pc_range']
        self.voxel_size = inference_params['model']['reader']['voxel_size']
       
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = build_detector(self.cfg.model, train_cfg=None, test_cfg=self.cfg.test_cfg)
        self.net.load_state_dict(torch.load(model_path)["state_dict"])
        if FP16:
            self.net = self.net.half()
        self.net = self.net.to(self.device).eval()

        self.max_points_in_voxel = self.cfg.voxel_generator.max_points_in_voxel
        self.max_voxel_num = self.cfg.voxel_generator.max_voxel_num[1]
        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.range,
            max_num_points=self.max_points_in_voxel,
            max_voxels=self.max_voxel_num,
        )
        orient_thresh = rospy.get_param("~orient_jump_threshold", np.pi/12)
        self.pred_tracker = PredictiveTracker(dt=0.1, orient=orient_thresh)  
        rospy.loginfo("Model & Config loaded!")

        # Object transformation from lidar's coordinates to local frame
        self.object_transformer = ObjectTFConverter()

        # Trajectory saving
        self.trajectories = TrajectoryFactory(self.cfg)

        # ROS stuff
        self.sub = rospy.Subscriber("pointcloud", PointCloud2, self.lidar_callback,
            queue_size=10, buff_size=2 ** 20 * rospy.get_param("~input_buffer_size", 2**15))
        self.pub_arr_bbox = rospy.Publisher("objects", ObjectArray, queue_size=10)
        self.frame_skip_diff = rospy.get_param("~frame_skip_diff")
        self.c_arr = None
        self.pub_marker = rospy.Publisher("marker_objects", MarkerArray, queue_size=10)
        self.visualization_on = rospy.get_param("~publish_markers", True)
        self.color_by = rospy.get_param("~color_by", "id")

        # Thresholds for filtering pointcloud
        self.f_thresholds = (rospy.get_param("~filter_x_range", 1),
            rospy.get_param("~filter_y_range", 0))

        # pedestrian filtering by range
        self.pedestrian_detection_range = rospy.get_param("~filter_pedestrian_range", 15)

        # Dynamic reconfigure
        self.thresholds = None
        srv = Server(ThresholdsConfig, self.param_callback)

        if len(self.cfg.tasks) == 1:
            self.tracking_names = [
                "VEHICLE",      #1
                "PEDESTRIAN",   #2
                "CYCLIST"       #3
            ]
        else:
            self.tracking_names = [
                "car",              # 0
                "truck",            # 1
                "construction",     # 2
                "bus",              # 3
                "trailer",          # 4
                "barrier",          # 5
                "motorcycle",       # 6
                "bicycle",          # 7
                "pedestrain",       # 8
                "traffic"           # 9
            ]

        rospy.loginfo_once("Initialization completed")

    def remove_low_score(self, image_anno):
        img_filtered_annotations = {}
        label_preds_ = image_anno["label_preds"].detach().cpu().numpy()
        scores_ = image_anno["scores"].detach().cpu().numpy()

        if len(self.cfg.tasks) == 1:    # check if using waymo config or nuscenes config
            VEHICLE_indices = get_annotations_indices(0, self.thresholds['VEHICLE'], label_preds_, scores_)
            PEDESTRIAN_indices = get_annotations_indices(1, self.thresholds['PEDESTRIAN'], label_preds_, scores_)
            CYCLIST_indices = get_annotations_indices(2, self.thresholds['CYCLIST'], label_preds_, scores_)

            for key in image_anno.keys():
                if key == 'metadata':
                    continue
                img_filtered_annotations[key] = (
                    image_anno[key][VEHICLE_indices +
                                    PEDESTRIAN_indices +
                                    CYCLIST_indices
                                    ])
        else:
            car_indices = get_annotations_indices(0, self.thresholds['car'], label_preds_, scores_)
            truck_indices = get_annotations_indices(1, self.thresholds['truck'], label_preds_, scores_)
            construction_vehicle_indices = get_annotations_indices(2, self.thresholds['construction'], label_preds_, scores_)
            bus_indices = get_annotations_indices(3, self.thresholds['bus'], label_preds_, scores_)
            trailer_indices = get_annotations_indices(4, self.thresholds['trailer'], label_preds_, scores_)
            barrier_indices = get_annotations_indices(5, self.thresholds['barrier'], label_preds_, scores_)
            motorcycle_indices = get_annotations_indices(6, self.thresholds['motorcycle'], label_preds_, scores_)
            bicycle_indices = get_annotations_indices(7, self.thresholds['bicycle'], label_preds_, scores_)
            pedestrain_indices = get_annotations_indices(8, self.thresholds['pedestrian'], label_preds_, scores_)
            traffic_cone_indices = get_annotations_indices(9, self.thresholds['traffic'], label_preds_, scores_)

            for key in image_anno.keys():
                if key == 'metadata':
                    continue
                img_filtered_annotations[key] = (
                    image_anno[key][car_indices +
                                    pedestrain_indices +
                                    bicycle_indices +
                                    bus_indices +
                                    construction_vehicle_indices +
                                    traffic_cone_indices +
                                    trailer_indices +
                                    barrier_indices +
                                    truck_indices
                                    ])

        return img_filtered_annotations

    def run_detection_tracking(self, points, header):
        t_t = time.time()
        self.points = points.reshape([-1, points.shape[1]])
        self.points = self.points[:, :5]
        self.points[:, 4] = 0  # timestamp value
        #self.points = remove_close(self.points, *self.f_thresholds)
        voxel_output = self.voxel_generator.generate(self.points)
        voxels, coords, num_points = voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
 
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)
        grid_size = self.voxel_generator.grid_size
        coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)

        voxels = torch.tensor(voxels, dtype=torch.float32, device=self.device)
        coords = torch.tensor(coords, dtype=torch.int32, device=self.device)
        num_points = torch.tensor(num_points, dtype=torch.int32, device=self.device)
        num_voxels = torch.tensor(num_voxels, dtype=torch.int32, device=self.device)

        if FP16:
            voxels = voxels.half()

        self.inputs = dict(
            voxels=voxels,
            num_points=num_points,
            num_voxels=num_voxels,
            coordinates=coords,
            shape=[grid_size]
        )
        torch.cuda.synchronize()
        t = time.time()

        with torch.no_grad():
            outputs = self.net(self.inputs, return_loss=False)[0]

        torch.cuda.synchronize()
        time_lag = time.time() - t
        # rospy.logdebug(f"network predict time cost: {time_lag}")
        outputs = self.remove_low_score(outputs)

        box3d = outputs["box3d_lidar"].detach().cpu().numpy()
        scores = outputs["scores"].detach().cpu().numpy()
        types = outputs["label_preds"].detach().cpu().numpy()
        # rospy.logdebug(f"predict boxes: {box3d.shape}")
        # Tracking section:
        box3d[:, -1] = -box3d[:, -1] - np.pi / 2
        preds = []
        pred = {}
        dets = []  # h, w, l, x, y, z, theta
        # rospy.logdebug(f"Number of detections: {box3d.shape[0]}")
        for i in range(box3d.shape[0]):
            dis = distance.euclidean(box3d[i, :3].tolist(), 0)
            if (len(self.cfg.tasks) == 1) and (int(types[i].tolist()) == 1) and (dis >= self.pedestrian_detection_range):
                continue
            elif (int(types[i].tolist()) == 8) and (dis >= self.pedestrian_detection_range):
                continue
            quat = quaternion_from_euler(0, 0, box3d[i, -1])
            velocity = box3d[i, 6:8].tolist()
            velocity.append(0.0)
            pred = {
                "sample_token": i,
                "translation": box3d[i, :3].tolist(),
                "size": box3d[i, 3:6].tolist(),
                "rotation": [quat[3], quat[0], quat[1], quat[2]],
                "velocity": velocity,
                "detection_name": int(types[i].tolist()),
                "detection_score": scores[i].tolist(),
            }
            preds.append(pred)

            # transforming pose of object to fixed coordinates
            pose = PoseStamped()
            pose.header = header
            pose.pose.position.x = box3d[i, 0]
            pose.pose.position.y = box3d[i, 1]
            pose.pose.position.z = box3d[i, 2]
            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]
            fixed_pose = self.object_transformer.transform_pose(pose)
            if fixed_pose is None:
                continue
            _, _, yaw = euler_from_quaternion([
                fixed_pose.pose.orientation.x,
                fixed_pose.pose.orientation.y,
                fixed_pose.pose.orientation.z,
                fixed_pose.pose.orientation.w
                ])
            det = [fixed_pose.pose.position.x, fixed_pose.pose.position.y, fixed_pose.pose.position.z, box3d[i, 3], box3d[i, 4], box3d[i, 5], yaw, types[i], i]
            dets.append(det)
        dets = np.array(dets)

        arr_bbox = ObjectArray()
        if len(dets) > 0 :
            tracks, velocities = self.pred_tracker.update(dets)  # hungarian + kalman with filter speed

            # rospy.logdebug(f"total cost time: {time.time() - t_t}")
            # rospy.logdebug(f"Types of detections: {types}")
            # rospy.logdebug(f"The scores of detections: {scores}")
            # rospy.logdebug(f"Tracking IDs of tracked detections: {tracks.keys}")

            # Filling objects array
            arr_bbox.header.frame_id = self.object_transformer.fixed_frame
            arr_bbox.header.stamp = header.stamp
            if len(tracks) != 0:
                for i, t in tracks.items():
                    bbox = Object()
                    q = quaternion_from_euler(0, 0, float(t[6]))
                    bbox.pose.orientation.x = q[0]
                    bbox.pose.orientation.y = q[1]
                    bbox.pose.orientation.z = q[2]
                    bbox.pose.orientation.w = q[3]
                    bbox.pose.position.x = float(t[0])
                    bbox.pose.position.y = float(t[1])
                    bbox.pose.position.z = float(t[2])
                    bbox.size.x = float(t[4])
                    bbox.size.y = float(t[3])
                    bbox.size.z = float(t[5])
                    try:
                        bbox.score = scores[int(t[8])]
                    except IndexError:  # for unmatched and unassigned objects
                        bbox.score = -1
                    bbox.id = i
                    bbox.label = int(t[7])
                    bbox.velocity.linear.x = float(velocities[i][0])
                    bbox.velocity.linear.y = float(velocities[i][1])
                    bbox.velocity.linear.z = float(velocities[i][2])
                    arr_bbox.objects.append(bbox)

        return arr_bbox

    def lidar_callback(self, msg):
        t_t = time.time()
        if self.frame_skip_diff and (rospy.Time.now().to_sec() - 
                msg.header.stamp.to_sec()) > self.frame_skip_diff:
            return
        np_p = get_xyz_points(msg, True)
        # Clear old ids of trajectories
        self.trajectories.clear_by_timeout(msg.header.stamp.to_sec())

        arr_bbox = self.run_detection_tracking(np_p, msg.header)
        if len(arr_bbox.objects) != 0:
            arr_bbox = self.trajectories.append_frame(arr_bbox)
            self.pub_arr_bbox.publish(arr_bbox)
            if self.visualization_on:
                self.publish_markers(arr_bbox)
            arr_bbox.objects = []
        rospy.logdebug(f"total callback time: {time.time() - t_t}")

    def publish_markers(self, objects):
        if self.c_arr is None:
            # clear old markers
            self.c_arr = MarkerArray()
            c_m = Marker()
            c_m.header = objects.header
            c_m.ns = "objects"
            c_m.id = 0
            c_m.action = Marker.DELETEALL
            c_m.lifetime = rospy.Duration()
            self.c_arr.markers.append(c_m)
            c_m.ns = "ids"
            self.c_arr.markers.append(c_m)
            c_m.ns = "velocities"
            self.c_arr.markers.append(c_m)
            c_m.ns = "velocities_text"
            self.c_arr.markers.append(c_m)
            c_m.ns = "trajectories"
            self.c_arr.markers.append(c_m)
        self.pub_marker.publish(self.c_arr)

        m_arr = MarkerArray()
        for obj in objects.objects:
            m_list = make_marker(obj, objects.header,tracking_names=self.tracking_names, color=self.color_by)
            m_arr.markers.extend(m_list)
        self.pub_marker.publish(m_arr)

    def param_callback(self, config, level):
        self.thresholds = {k: v for k, v in config.items() if k != 'groups'}
        rospy.loginfo("Current thresholds: \n%s", self.thresholds)
        return config

def main():
    rospy.init_node('centerpoint_ros_node')

    # CenterPoint
    proc = ProcessorROS()

    rospy.spin()

if __name__ == "__main__":
    main()
