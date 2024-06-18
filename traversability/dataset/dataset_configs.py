import numpy as np

class UnitreeConfig:
    """
    This script contains the configurations for dataset creation.
    It defines the output files, topics for data extraction, dataset properties,
    and lists of rosbags to be extracted for training and validation.
    """

    def __init__(self):
        self.robot_name = "unitree_go1"

        # Define topics for data extraction
        self.camera_topics = [
            "/unitree/front_stereo/left_camera",
            "/unitree/left_stereo/left_camera",
            "/unitree/right_stereo/right_camera"
        ]

        self.gnss_topic     = "/unitree/fix"
        self.odom_topic     = "/unitree/dlio/odom_node/odom"
        self.pcloud_topic   = "/unitree/velodyne_points"
        self.command_topic  = "/unitree/motion_command"
        self.battery_topic  = "/unitree/battery_state"

        # Observations sampling time [s]
        self.observ_dt = 0.5
        self.state_dt = 0.1

        # Define list of rosbags to be extracted for training
        self.train_bags_list = [
            'unitree_1969-12-31-18-07-33_2.bag',
            'unitree_2024-04-19-16-09-02.bag',
            'unitree_2024-04-19-16-13-32.bag',
            'unitree_2024-05-08-15-10-00.bag',
            'unitree_2024-05-08-15-15-19.bag',
        ]

        # Define list of rosbags to be extracted for validation
        self.valid_bags_list = [
            'unitree_1969-12-31-18-05-25.bag',
            'unitree_2024-04-19-16-29-37.bag',
            'unitree_2024-05-08-15-23-34.bag',
        ]

class TSZedConfig:
    def __init__(self):
        # Define robot's name
        self.robot_name = "terrasentia_zed"

        # Define topics for data extraction
        self.camera_topics = [
            "/terrasentia/zed2/zed_node/left"
        ]

        # Define topics for data extraction
        self.odom_topic     = "/terrasentia/ekf"
        self.command_topic  = "/terrasentia/motion_command"

        # Define sensors transformations
        self.camera_offset = [0.17, 0, 0.37]

        self.cam2base = np.array([
            [ 0.0, 0.0, 1.0, self.camera_offset[0]], 
            [-1.0, 0.0, 0.0, self.camera_offset[1]],
            [ 0.0,-1.0, 0.0, self.camera_offset[2]],
            [ 0.0, 0.0, 0.0, 1.0 ]])

        # Observations sampling time [s]
        self.observ_dt = 0.5
        self.state_dt = 0.1

        # Define list of rosbags to be extracted
        self.train_bags_list = [
            'ts_2022_04_10_14h27m23s.bag',
            'ts_2022_04_14_16h30m09s.bag',
            'ts_2022_04_14_16h40m55s.bag',
            'ts_2022_04_21_16h07m26s.bag',
            'ts_2022_04_21_16h10m35s.bag',
            'ts_2022_04_21_16h30m09s.bag',
            'ts_2022_04_21_16h41m11s.bag',
            'ts_2022_04_21_16h53m33s.bag',
            'ts_2022_07_05_17h59m28s.bag',
            'ts_2022_07_06_14h52m48s.bag',
            'ts_2022_07_11_17h40m00s.bag',
            'ts_2022_07_12_16h05m40s.bag',
            'ts_2022_07_12_16h42m04s.bag',
            'ts_2023_04_10_20h05m59s.bag',
            'ts_2023_04_10_20h12m46s.bag',
            'ts_2023_04_11_17h53m56s_filtered.bag',
            'ts_2023_04_11_17h51m24s.bag',
            'ts_2023_04_11_18h00m20s.bag',
            'ts_2023_06_25_20h16m34s.bag',
            'ts_2023_06_25_20h32m20s.bag',
            'ts_2023_06_25_21h37m18s.bag',
            'ts_2023_06_28_20h12m27s.bag',
            'ts_2023_06_28_20h12m37s.bag',
            'ts_2023_06_28_20h29m17s.bag',
            'ts_2023_06_28_20h47m38s.bag',
            'ts_2023_06_28_20h12m04s.bag',
            'ts_2023_06_28_20h28m27s.bag',
            # # Bags from Shreya's dataset
            'ts_2023_04_11_19h57m53s.bag',
            'ts_2023_06_25_23h24m19s.bag'
        ]

        self.valid_bags_list = [
            'ts_2022_04_10_14h16m59s.bag',
            'ts_2023_04_11_17h47m24s.bag',
            'ts_2023_04_11_18h02m48s_filtered.bag',
            'ts_2023_06_25_21h55m53s.bag',
            'ts_2023_06_28_20h29m58s.bag',
            'ts_2023_06_28_20h48m37s.bag',
            'ts_2023_06_28_21h04m14s.bag',
            # # Bags from Shreya's dataset
            'ts_2023_04_11_17h51m03s.bag',
            'ts_2023_04_11_17h58m32s.bag',
            'ts_2023_04_11_18h17m40s_filtered.bag',
            'ts_2023_04_11_20h02m40s.bag',
            # # Ignore the following bags bc of error in the tf tree
            # # 'ts_2023_06_25_23h56m10s_filtered.bag'
        ]