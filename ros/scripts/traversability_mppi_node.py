#!/usr/bin/env python3
import copy
import torch
import numpy as np

import rospy
import message_filters
from fpn_msgs.msg import MPCInput
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped

import os
import sys
import rospkg

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('bev_navigation')
sys.path.append(os.path.join(pkg_path, '..'))

from traversability_research.cam_traversability.scripts.train.read_configs import get_cfg
from traversability.mppi.mppi import MPPI
from traversability.mppi.utils import visualize_rollouts

class TRAVERSABILITY_MPPI_NODE:
    def __init__(self):
        # Define config file
        config_file     = rospy.get_param('~config_file', 'temporal_model.yaml')
        # Define ROS topics
        cmd_topic       = rospy.get_param('~cmd_topic', '/unitree/cmd')
        reference_topic = rospy.get_param('~reference_topic', '/unitree/path2')
        mu_topic = rospy.get_param('~mu_topic', '/unitree/bev_traversability_mu')
        nu_topic = rospy.get_param('~nu_topic', '/unitree/bev_traversability_nu')
        self.invert_ang_vel = rospy.get_param('~invert_ang_vel', False)
        self.verbose = rospy.get_param('~verbose', False)

        self.publish_rollouts = True
        
        self.dtype = torch.float
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print('MPPI NODE using device:', self.device)

        # Load configs
        config_path = os.path.join(pkg_path, '../configs/' + config_file)
        config = get_cfg(config_path)

        print('CONFIGS:\n', config)

        self.N = config.CONTROL.horizon
        self.grid_bounds = config.MODEL.GRID_BOUNDS
        real_dt = config.CONTROL.real_dt
        self.run_step = False

        # Create network object
        self.mppi = MPPI(config)
        self.mppi.eval()

        # Create empty reference trajectory
        self.mpc_reference = {
            'x': [],
            'y': [],
            'theta': [],
            'speed': np.array([]),
            'omega': np.array([])}

        # Subscribers
        rospy.Subscriber(reference_topic, MPCInput, self.reference_callback, queue_size=1)
        # Image filter subscriber
        mu_sub = message_filters.Subscriber(mu_topic, Image, queue_size=1, buff_size=2**24)
        nu_sub = message_filters.Subscriber(nu_topic, Image, queue_size=1, buff_size=2**24)
        ts = message_filters.ApproximateTimeSynchronizer([mu_sub, nu_sub], 1, 0.2)
        ts.registerCallback(self.traversability_callback)

        # Publishers
        self.pub_cmd = rospy.Publisher(cmd_topic, TwistStamped, queue_size=1)
        if self.publish_rollouts:
            self.pub_rollouts = rospy.Publisher('mppi_rollouts', Image, queue_size=1)
        # pub_output = rospy.Publisher("mpc_node/output", MPCOutput, queue_size=1)

        # Create traversability map that is fully traversable
        map_size = (
            int((self.grid_bounds['xbound'][1] - self.grid_bounds['xbound'][0])/self.grid_bounds['xbound'][2]),
            int((self.grid_bounds['ybound'][1] - self.grid_bounds['ybound'][0])/self.grid_bounds['ybound'][2]))
        self.trav_map  = np.ones((2, map_size[0], map_size[1]))

        # Set loop rate
        rate = rospy.Rate(50)
        # Main loop
        while not rospy.is_shutdown():
            self.run_navigation()
            rate.sleep()

    def run_navigation(self):
        # Run only when a new reference path arrives
        if not self.run_step:
            return

        mpc_reference = copy.deepcopy(self.mpc_reference)
        trav_map = copy.deepcopy(self.trav_map)

        if len(mpc_reference['x']) == 0:
            if self.verbose:
                rospy.loginfo("TRAV-MPPI empty wps")
            
            mpc_cmd = TwistStamped()
            mpc_cmd.header.stamp = rospy.Time.now()
            self.pub_cmd.publish(mpc_cmd)
        
        else:
            if self.verbose:
                rospy.loginfo("TRAV-MPPI wps are not empty")

            # Move data to GPU
            reference = torch.tensor([
                mpc_reference['x'],
                mpc_reference['y'],
                mpc_reference['theta'],
                mpc_reference['v'],
                mpc_reference['omega']], dtype=self.dtype, device=self.device).t()

            trav_map = torch.tensor(trav_map, dtype=self.dtype, device=self.device).unsqueeze(0)
            # And run MPPI
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    a_hat, trajs, pred_states, probs, trav_costs = self.mppi(reference.unsqueeze(0), trav_map)

            prediction = trajs.cpu().detach().numpy()
            rollouts = pred_states[0].cpu().detach().numpy()
            actions = a_hat[0].cpu().detach().numpy()
            probs = probs[0].cpu().detach().numpy()
            trav_costs = trav_costs[0].cpu().detach().numpy()

            print('actions:', actions)
            print('actions.shape:', actions.shape)

            # Publish command
            mpc_cmd = TwistStamped()
            mpc_cmd.header.stamp = rospy.Time.now()
            mpc_cmd.twist.linear.x = actions[0,0]
            if self.invert_ang_vel:
                mpc_cmd.twist.angular.z = -actions[0,1]
            else:
                mpc_cmd.twist.angular.z = actions[0,1]
            # publish commands
            self.pub_cmd.publish(mpc_cmd)

            if self.publish_rollouts:
                rollouts_im = visualize_rollouts(rollouts, trav_costs, self.grid_bounds)
                rollouts_msg = Image()
                rollouts_msg.header.stamp = rospy.Time.now()
                rollouts_msg.height = rollouts_im.shape[0]
                rollouts_msg.width = rollouts_im.shape[1]
                rollouts_msg.encoding = "rgb8"
                rollouts_msg.is_bigendian = False
                rollouts_msg.step = rollouts_im.shape[1] * 3
                rollouts_msg.data = np.array(rollouts_im).tobytes()
                self.pub_rollouts.publish(rollouts_msg)

            # # Publish MPPI predicted output
            # mppi_out = MPCOutput()
            # for i in range(paths.shape[0]):
            #     mppi_out.x.append(paths[i,0])
            #     mppi_out.y.append(paths[i,1])
            #     #mppi_out.mu.append(costs[path_idx])
            # # Publish outputs

            # print('mppi_out.x', mppi_out.x)
            # print('mppi_out.y', mppi_out.y)
            # self.pub_output.publish(mppi_out)

    def traversability_callback(self, mu_msg, nu_msg):
        mu_im = np.frombuffer(mu_msg.data, dtype=np.uint8).reshape(mu_msg.height, mu_msg.width)
        nu_im = np.frombuffer(nu_msg.data, dtype=np.uint8).reshape(nu_msg.height, nu_msg.width)
        im = np.stack([mu_im, nu_im], 0)
        self.trav_map = im/255.0

    def reference_callback(self, reference_msg):
        # First, let's prevent the MPC to run while we are preparing the reference
        self.run_step = False
        # The input waypoints are interpolated to have the same distance between points as the distance step used in the MPC
        if len(reference_msg.poses) == 0:
            if self.verbose:
                print("Received empty path")
            
            self.mpc_reference['x'].clear()
            self.mpc_reference['y'].clear()
            self.mpc_reference['theta'].clear()
            self.mpc_reference['v'] = np.array([])
            self.mpc_reference['omega'] = np.array([])

            # Now MPC can run
            self.run_step = True
    
        elif len(reference_msg.poses) < 4:
            if self.verbose:
                print("Path arrived with less than 4 points...")
                print("Using reference as a single point tracking.")

            # Then we follow the points as the only value without regression
            self.mpc_reference['x'].clear()
            self.mpc_reference['y'].clear()
            self.mpc_reference['theta'].clear()
            self.mpc_reference['v'] = np.ones(self.N) * reference_msg.reference_speed
            self.mpc_reference['omega'] = np.zeros(self.N)
            
            for t in range(self.N):
                x_ref = reference_msg.poses[0].pose.position.x
                y_ref = reference_msg.poses[0].pose.position.y
                heading_ref = np.arctan2(y_ref, x_ref)

                # Check if speed is negative and invert the heading to face forward
                if self.mpc_reference['v'][t] < 0:
                    # Normalize angle to [-pi,pi)
                    heading_ref = (heading_ref + np.pi) % (2*np.pi) - np.pi

                self.mpc_reference['x'].append(x_ref)
                self.mpc_reference['y'].append(y_ref)
                self.mpc_reference['theta'].append(heading_ref)

            # Now MPC can run
            self.run_step = True
        
        else:
            if self.verbose:
                print("Received path size:", len(reference_msg.poses))
                print("This type of reference is not implemented yet!!")

if __name__ == '__main__':
    rospy.init_node('traversability_mppi')
    TRAVERSABILITY_MPPI_NODE()