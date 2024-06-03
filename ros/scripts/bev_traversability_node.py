#!/usr/bin/env python3
import cv2
import copy
import torch
import bisect
import numpy as np
from threading import Lock
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

import rospy
import message_filters
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage, Image, CameraInfo

import os
import sys
import rospkg

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('bev_navigation')
sys.path.append(os.path.join(pkg_path, '..'))

from traversability_research.cam_traversability.scripts.train.read_configs import get_cfg
from traversability.train.trainer import TrainingModule

class TRAVERSABILITY_MAP_NODE:
    def __init__(self):
        # Define config file
        config_file     = rospy.get_param('~config_file', 'temporal_model.yaml')
        # Define trained model
        self.model_path = rospy.get_param('~model_path')
        # Define ROS topics
        image_topic     = rospy.get_param('~image_topic', '/terrasentia/zed2/zed_node/left/image_rect_color/compressed')
        camera_topic    = rospy.get_param('~camera_topic', '/terrasentia/zed2/zed_node/left/camera_info')
        depth_topic     = rospy.get_param('~depth_topic', '/terrasentia/zed2/zed_node/depth/depth_registered')
        pose_topic      = rospy.get_param('~odom_topic', '/terrasentia/ekf')
        cam_pos_x       = rospy.get_param('~cam_pos_x', 0.17)
        cam_pos_y       = rospy.get_param('~cam_pos_y', 0.06)
        cam_pos_z       = rospy.get_param('~cam_pos_z', 0.37)
        self.publish_image = rospy.get_param('~publish_image', True)
        # Define inference frequency
        mapping_rate    = rospy.get_param('~mapping_rate', 10.0)
        self.data_lock = Lock()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print('BEV TRAVERSABILITY NODE using device:', self.device)

        # Load configs
        config_path = os.path.join(pkg_path, '../configs/' + config_file)
        config = get_cfg(config_path)
        # Camera to base_link transformation
        self.center_cam2base = np.array([
            [ 0.0, 0.0, 1.0, cam_pos_x],
            [-1.0, 0.0, 0.0, cam_pos_y],
            [ 0.0,-1.0, 0.0, cam_pos_z],
            [ 0.0, 0.0, 0.0, 1.0 ]])
        
        self.dtype = torch.float32
        self.input_size = tuple(config.MODEL.INPUT_SIZE)
        self.grid_bounds = config.MODEL.GRID_BOUNDS
        self.time_length = config.MODEL.TIME_LENGTH
        self.dt = 0.5

        # Create network object
        self.model = TrainingModule(config)

        # Load trained weights
        print(self.model_path)

        # Load trained weights
        trained_dict = torch.load(self.model_path, map_location='cpu')['state_dict']
        self.model.load_state_dict(trained_dict) #, strict=False)
        self.model.to(self.device)
        self.model.eval()

        # Initialize some data
        self.position = [0.0, 0.0, 0.0]
        self.quaternion = [0, 0, 0, 1]

        # Publishers
        if self.publish_image:
            self.mu_pub = rospy.Publisher('bev_traversability_mu', Image, queue_size=1)
            self.nu_pub = rospy.Publisher('bev_traversability_nu', Image, queue_size=1)

        # Subscribers
        rospy.Subscriber(pose_topic, Odometry, self.pose_callback)
        # Image filter subscriber
        rgb_sub     = message_filters.Subscriber(image_topic, CompressedImage, queue_size=1, buff_size=2**24)
        depth_sub   = message_filters.Subscriber(depth_topic, Image, queue_size=1, buff_size=2**24)
        cam_sub     = message_filters.Subscriber(camera_topic, CameraInfo, queue_size=1)

        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, cam_sub], 1, 0.2)
        ts.registerCallback(self.center_cam_cb)

        # Create data buffer
        self.buffer = {
            'left': {
                'stamp': [],
                'rgb': [],
                'depth': [],
                'intrinsics': [],
                'extrinsics': [],
                'position': [],
                'quaternion': []},
            'center': {
                'stamp': [],
                'rgb': [],
                'depth': [],
                'intrinsics': [],
                'extrinsics': [],
                'position': [],
                'quaternion': []},
            'right': {
                'stamp': [],
                'rgb': [],
                'depth': [],
                'intrinsics': [],
                'extrinsics': [],
                'position': [],
                'quaternion': []},
            'pcloud': None}

        # Set loop rate
        rate = rospy.Rate(mapping_rate)
        # Main loop
        while not rospy.is_shutdown():
            if len(self.buffer['center']['stamp']) < 1 or len(self.buffer['center']['extrinsics']) < 1:
                print('The node did not receive images yet...')

            else:
                pcloud, color_img, intrinsics, extrinsics = self.prepare_data()
                
                # And run our network model
                with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        trav_map, _, _ = self.model.model(color_img, pcloud, intrinsics, extrinsics)
                        trav_map = 1 - F.max_pool2d(1-trav_map, kernel_size=3, stride=1)

                self.publish_map(trav_map[0].detach().cpu().numpy())

            rate.sleep()

    def center_cam_cb(self, rgb_msg, depth_msg, cam_msg):
        # Get RGB image
        np_arr = np.frombuffer(rgb_msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) 
        color_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Get Depth image
        np_arr = np.frombuffer(depth_msg.data, np.float32)
        depth_img = copy.copy(np_arr.reshape(depth_msg.height, depth_msg.width))
        # Get Intrinsics
        intrinsics = np.array(cam_msg.K).reshape(3,3)
        # Get pose
        position = copy.deepcopy(self.position)
        quaternion = copy.deepcopy(self.quaternion)
        # Resize images if necessary
        if (self.input_size[0] != color_img.shape[1]) and (self.input_size[1] != color_img.shape[0]):
            intrinsics[:1] *= (self.input_size[0]/color_img.shape[1])
            intrinsics[1:2] *= (self.input_size[1]/color_img.shape[0])
            color_img = cv2.resize(color_img, self.input_size, interpolation=cv2.INTER_AREA)
            depth_img = cv2.resize(depth_img, self.input_size, interpolation=cv2.INTER_NEAREST)
        # Add information to the buffer
        with self.data_lock:
            self.buffer['center']['stamp'].append(rgb_msg.header.stamp.to_sec())
            self.buffer['center']['rgb'].append(color_img)
            self.buffer['center']['depth'].append(depth_img)
            self.buffer['center']['intrinsics'].append(intrinsics)
            self.buffer['center']['position'].append(position)
            self.buffer['center']['quaternion'].append(quaternion)
            # Remove data from the buffer if it's full
            if self.buffer['center']['stamp'][0] < self.buffer['center']['stamp'][-1] - (self.time_length-1)*self.dt:
                self.buffer['center']['stamp'].pop(0)
                self.buffer['center']['rgb'].pop(0)
                self.buffer['center']['depth'].pop(0)
                self.buffer['center']['intrinsics'].pop(0)
                self.buffer['center']['position'].pop(0)
                self.buffer['center']['quaternion'].pop(0)
            # And copy the list of positions and quaternions
            position = np.asarray(self.buffer['center']['position']).copy()
            quaternion = np.asarray(self.buffer['center']['quaternion']).copy()
        
        # Get rotations from quaternions
        rotation = R.from_quat(quaternion)
        # Get euler angles from rotations
        euler_angle = rotation.as_euler('zyx')
        # Get extrinsics disregarding heading angle
        heading_rot = R.from_euler('zyx', [euler_angle[-1,0], 0, 0])
        # Transform position in relation to the first timestamp
        position = (heading_rot.inv().as_matrix() @ (position.T - position[-1,None].T)).T
        # Get extrinsics disregarding heading angle
        extrinsics_list = []
        for i in range(len(self.buffer['center']['stamp'])):
            base_rot = R.from_euler('zyx', [0, euler_angle[i,1], euler_angle[i,2]])
            base_trans = np.eye(4)
            base_trans[:3,:3] = base_rot.as_matrix()
            odom_trans = np.eye(4)
            odom_rot = R.from_euler('zyx', [euler_angle[i,0]-euler_angle[-1,0], 0, 0])
            odom_trans[:3,:3] = odom_rot.as_matrix()
            odom_trans[:2,3] = position[i,:2]
            extrinsics = odom_trans @ base_trans @ self.center_cam2base
            extrinsics_list.append(extrinsics)

        with self.data_lock:
            self.buffer['center']['extrinsics'] = extrinsics_list

    def pose_callback(self, pose_msg):
        self.position = [
            pose_msg.pose.pose.position.x, 
            pose_msg.pose.pose.position.y,
            pose_msg.pose.pose.position.z]
        self.quaternion = [
            pose_msg.pose.pose.orientation.x, 
            pose_msg.pose.pose.orientation.y,
            pose_msg.pose.pose.orientation.z,
            pose_msg.pose.pose.orientation.w]

    def publish_map(self, map_image):
        mu_map = map_image[0]
        nu_map = map_image[1]

        if self.publish_image:
            im = 255*mu_map
            im = im.astype('uint8')
            # Populate image message
            msg = Image()
            msg.header.stamp = rospy.Time.now()
            msg.height = im.shape[0]
            msg.width = im.shape[1]
            msg.encoding = "mono8"
            msg.is_bigendian = False
            msg.step = im.shape[1]
            msg.data = np.array(im).tobytes()
            # Publish mu map image
            self.mu_pub.publish(msg)

            # Publish nu map image
            im = 255*nu_map
            im = im.astype('uint8')
            msg.data = np.array(im).tobytes()
            self.nu_pub.publish(msg)

    def prepare_data(self):
        # Lock data for data sharing
        with self.data_lock:
            self.buffer['center']['stamp']
            self.buffer['center']['rgb']
            self.buffer['center']['intrinsics']
            self.buffer['center']['extrinsics']
            timestamps_list = copy.deepcopy(self.buffer['center']['stamp'])
            color_img_list = copy.deepcopy(self.buffer['center']['rgb'])
            depth_img_list = copy.deepcopy(self.buffer['center']['depth'])
            intrinsics_list = copy.deepcopy(self.buffer['center']['intrinsics'])
            extrinsics_list = copy.deepcopy(self.buffer['center']['extrinsics'])

        timestamps = np.asarray(timestamps_list)
        color_img = np.stack(color_img_list)
        depth_img = np.stack(depth_img_list)
        intrinsics = np.stack(intrinsics_list)
        extrinsics = np.stack(extrinsics_list)
        # Sync timestamps
        ref_stamps = timestamps_list[-1] - np.flip(np.arange(self.time_length)) * self.dt
        sync = [bisect.bisect_left(timestamps, i) for i in ref_stamps]
        color_img = color_img[sync]
        depth_img = depth_img[sync]
        intrinsics = intrinsics[sync]
        extrinsics = extrinsics[sync]
        
        # Normalize color image
        image_tensor = torch.from_numpy(color_img/255.0)
        image_tensor = image_tensor.permute(0,3,1,2)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.type(self.dtype).to(self.device)
        # Get point cloud from depth image
        voxels = self.depth2voxels(depth_img, intrinsics, extrinsics)
        voxels = torch.tensor(voxels).unsqueeze(0)
        voxels = voxels.type(self.dtype).to(self.device)
        # And transform everything to tensor with appropriate shapes
        intrinsics = torch.tensor(intrinsics).view([1,-1,3,3]).type(self.dtype).to(self.device)
        extrinsics = torch.tensor(extrinsics).view([1,-1,4,4]).type(self.dtype).to(self.device)

        return voxels, image_tensor, intrinsics, extrinsics
    
    # def depth2voxels(self, depth_img, intrinsics, extrinsics):
    #     # separate rotation and translation from extrinsics matrix
    #     rotation, translation = extrinsics[:3, :3], extrinsics[:3, 3]
    #     # Create points from depth in the image space
    #     xs = np.arange(0, depth_img.shape[1]) #, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
    #     ys = np.arange(0, depth_img.shape[0]) #, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
    #     xs, ys = np.meshgrid(xs, ys)
    #     # Camera to ego reference frame
    #     points = np.stack((
    #         xs.flatten() * depth_img.flatten(),
    #         ys.flatten() * depth_img.flatten(),
    #         depth_img.flatten()), -1)
    #     points = points[np.isfinite(points[:,2])]
    #     combined_transformation = rotation @ np.linalg.inv(intrinsics)
    #     points = (combined_transformation @ points.T).T
    #     points += translation

    #     dx = np.asarray([row[2] for row in [self.grid_bounds['xbound'], self.grid_bounds['ybound'], self.grid_bounds['zbound']]])
    #     cx = np.asarray([np.round(row[1]/row[2] - 0.5) for row in [self.grid_bounds['xbound'], self.grid_bounds['ybound'], self.grid_bounds['zbound']]]).astype(int)
    #     nx = np.asarray([(row[1] - row[0]) / row[2] for row in [self.grid_bounds['xbound'], self.grid_bounds['ybound'], self.grid_bounds['zbound']]]).astype(int)

    #     # Create gridmap X x Y x Z
    #     voxels = np.zeros((nx[0], nx[1], nx[2]))
    #     idx_lidar = np.round(np.array([cx]) - points/dx - 0.5).astype(int)
    #     idx_lidar = idx_lidar[(idx_lidar[:,0] >= 0) * (idx_lidar[:,0] < nx[0]) * (idx_lidar[:,1] >= 0) * (idx_lidar[:,1] < nx[1]) * (idx_lidar[:,2] >= 0) * (idx_lidar[:,2] < nx[2])]
    #     voxels[idx_lidar[:,0], idx_lidar[:,1], idx_lidar[:,2]] = 1

    #     # Transform it to Z x X x Y
    #     voxels = voxels.transpose((2,0,1))
    #     return voxels
    
    def depth2voxels(self, depth_image_list, cam_intrinsics_list, cam_extrinsics_list):
        temporal_grid = []
        T = depth_image_list.shape[0]
        for t in range(T):
            depth_image = depth_image_list[t]
            cam_intrinsics = cam_intrinsics_list[t]
            cam_extrinsics = cam_extrinsics_list[t]
            # separate rotation and translation from extrinsics matrix
            rotation, translation = cam_extrinsics[:3, :3], cam_extrinsics[:3, 3]
            # Create points from depth in the image space
            xs = np.arange(0, depth_image.shape[1]) #, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
            ys = np.arange(0, depth_image.shape[0]) #, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
            xs, ys = np.meshgrid(xs, ys)
            # Camera to ego reference frame
            points = np.stack((
                xs.flatten() * depth_image.flatten() * 1e-3,
                ys.flatten() * depth_image.flatten() * 1e-3,
                depth_image.flatten() * 1e-3), -1)
            points = points[np.isfinite(points[:,2])]
            points = points[points[:,2] > 0]
            combined_transformation = rotation @ np.linalg.inv(cam_intrinsics)
            points = (combined_transformation @ points.T).T
            points += translation

            # pcloud = np.concatenate((pcloud, points))

            dx = np.asarray([row[2] for row in [self.grid_bounds['xbound'], self.grid_bounds['ybound'], self.grid_bounds['zbound']]])
            cx = np.asarray([np.round(row[1]/row[2] - 0.5) for row in [self.grid_bounds['xbound'], self.grid_bounds['ybound'], self.grid_bounds['zbound']]]).astype(int)
            nx = np.asarray([(row[1] - row[0]) / row[2] for row in [self.grid_bounds['xbound'], self.grid_bounds['ybound'], self.grid_bounds['zbound']]]).astype(int)

            # Create gridmap X x Y x Z
            grid = np.zeros((nx[0], nx[1], nx[2]))
            idx_lidar = np.round(np.array([cx]) - points/dx - 0.5).astype(int)
            idx_lidar = idx_lidar[(idx_lidar[:,0] >= 0) * (idx_lidar[:,0] < nx[0]) * (idx_lidar[:,1] >= 0) * (idx_lidar[:,1] < nx[1]) * (idx_lidar[:,2] >= 0) * (idx_lidar[:,2] < nx[2])]
            grid[idx_lidar[:,0], idx_lidar[:,1], idx_lidar[:,2]] = 1

            # Transform it to Z x X x Y
            grid = grid.transpose((2,0,1))
            temporal_grid.append(grid)

        return np.stack(temporal_grid)

if __name__ == '__main__':
    rospy.init_node('bev_trav_predictor')
    TRAVERSABILITY_MAP_NODE()