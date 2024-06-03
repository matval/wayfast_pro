#!/usr/bin/env python3
import cv2
import copy
import numpy as np
from threading import Lock
from scipy.spatial.transform import Rotation as R

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Image

import os
import sys
import rospkg

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('bev_navigation')
sys.path.append(os.path.join(pkg_path, '..'))

from traversability.train.read_configs import get_cfg
from traversability.train.trainer import TrainingModule

class TRAVERSABILITY_MAP_NODE:
    def __init__(self):
        # Define config file
        config_file = rospy.get_param('~config_file', 'temporal_model.yaml')
        # Define ROS topics
        lidar_topic = rospy.get_param('~lidar_topic', '/terrasentia/scan')
        pose_topic  = rospy.get_param('~odom_topic', '/terrasentia/ekf')
        lidar_pos_x = rospy.get_param('~lidar_pos_x', 0.17)
        lidar_pos_y = rospy.get_param('~lidar_pos_y', 0.0)
        lidar_pos_z = rospy.get_param('~lidar_pos_z', 0.34)
        self.publish_image = rospy.get_param('~publish_image', True)
        # Define inference frequency
        mapping_rate    = rospy.get_param('~mapping_rate', 10.0)
        self.data_lock = Lock()

        # Load configs
        config_path = os.path.join(pkg_path, '../configs/' + config_file)
        config = get_cfg(config_path)
        # Camera to base_link transformation
        self.lidar2base = np.array([
            [ 1.0, 0.0, 0.0, lidar_pos_x],
            [ 0.0, 1.0, 0.0, lidar_pos_y],
            [ 0.0, 0.0, 1.0, lidar_pos_z],
            [ 0.0, 0.0, 0.0, 1.0 ]])
        
        self.grid_bounds = config.MODEL.GRID_BOUNDS
        self.dx = np.asarray([row[2] for row in [self.grid_bounds['xbound'], self.grid_bounds['ybound'], self.grid_bounds['zbound']]])
        self.cx = np.asarray([np.round(row[1]/row[2] - 0.5) for row in [self.grid_bounds['xbound'], self.grid_bounds['ybound'], self.grid_bounds['zbound']]]).astype(int)
        self.nx = np.asarray([(row[1] - row[0]) / row[2] for row in [self.grid_bounds['xbound'], self.grid_bounds['ybound'], self.grid_bounds['zbound']]]).astype(int)

        # Initialize some data
        self.scan = []
        self.extrinsics = copy.deepcopy(self.lidar2base)

        # Publishers
        if self.publish_image:
            self.mu_pub = rospy.Publisher('bev_traversability_mu', Image, queue_size=1)
            self.nu_pub = rospy.Publisher('bev_traversability_nu', Image, queue_size=1)

        # Subscribers
        rospy.Subscriber(pose_topic, Odometry, self.pose_callback)
        rospy.Subscriber(lidar_topic, LaserScan, self.lidar_callback, queue_size=1)

        # Set loop rate
        rate = rospy.Rate(mapping_rate)
        # Main loop
        while not rospy.is_shutdown():
            lidar_scan = copy.deepcopy(self.scan)
            lidar_extrinsics = copy.deepcopy(self.extrinsics)

            if len(lidar_scan) > 0:
                trav_map = self.lidar2map(lidar_scan, lidar_extrinsics)

                trav_map = np.stack([trav_map, trav_map], axis=0)

                self.publish_map(trav_map)

            rate.sleep()

    def lidar_callback(self, scan_msg):
        self.scan = scan_msg.ranges

    def pose_callback(self, pose_msg):
        quaternion = [
            pose_msg.pose.pose.orientation.x, 
            pose_msg.pose.pose.orientation.y,
            pose_msg.pose.pose.orientation.z,
            pose_msg.pose.pose.orientation.w]

        # Get rotations from quaternions
        rotation = R.from_quat(quaternion)
        # Get euler angles from rotations
        euler_angle = rotation.as_euler('zyx')
        # Get extrinsics disregarding heading angle
        base_rot = R.from_euler('zyx', [0, euler_angle[1], euler_angle[2]])
        base_trans = np.eye(4)
        base_trans[:3,:3] = base_rot.as_matrix()
        self.extrinsics = base_trans @ self.lidar2base

    def publish_map(self, map_image):
        mu_map = map_image[0]
        nu_map = 0.4*map_image[1]

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
    
    def lidar2map(self, lidar_scan, lidar_extrinsics):
        # Transform LiDAR to pointcloud
        lidar_scan = np.asarray(lidar_scan)
        angles = np.linspace(-2.35619449, 2.35619449, len(lidar_scan))
        filt = lidar_scan > 0.2
        lidar_scan = lidar_scan[filt]
        angles = angles[filt]
        lidar_pcloud = np.ones([4, len(lidar_scan)])
        lidar_pcloud[0] = lidar_scan*np.cos(angles)
        lidar_pcloud[1] = lidar_scan*np.sin(angles)
        lidar_pcloud[2] = 0.0
        # Transform point cloud by lidar_extrinsics
        lidar_pcloud = lidar_extrinsics @ lidar_pcloud
        # Filter values beyond bounderies
        filt = (lidar_pcloud[0] >= self.grid_bounds['xbound'][0]) * (lidar_pcloud[0] <= self.grid_bounds['xbound'][1]) * \
            (lidar_pcloud[1] >= self.grid_bounds['ybound'][0]) * (lidar_pcloud[1] <= self.grid_bounds['ybound'][1]) * \
            (lidar_pcloud[2] >= self.grid_bounds['zbound'][0]) * (lidar_pcloud[2] <= self.grid_bounds['zbound'][1])
        lidar_pcloud = lidar_pcloud[:, filt]
        # Create gridmap
        grid = np.zeros((self.nx[0], self.nx[1]))
        idx_u = np.round(np.array([self.cx[0]]).T - lidar_pcloud[0]/self.dx[0] - 0.5).astype(int)
        idx_v = np.round(np.array([self.cx[1]]).T - lidar_pcloud[1]/self.dx[1] - 0.5).astype(int)
        # idx_lidar = idx_lidar[:, (idx_lidar[0] >= 0) * (idx_lidar[0] < configs.map_size[0]) * (idx_lidar[1] >= 0) * (idx_lidar[1] < configs.map_size[1])]
        grid[idx_u, idx_v] = 1

        kernel = np.array([
            [0,1,1,1,0],
            [1,1,1,1,1],
            [1,1,1,1,1],
            [1,1,1,1,1],
            [0,1,1,1,0]], dtype=np.uint8)
        lidar_map = cv2.dilate(grid, kernel, iterations=1)
        lidar_map = 1-lidar_map

        return lidar_map

if __name__ == '__main__':
    rospy.init_node('lidar_trav_predictor')
    TRAVERSABILITY_MAP_NODE()