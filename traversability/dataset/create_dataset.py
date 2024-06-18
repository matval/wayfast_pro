import os
import csv

from dataset_configs import UnitreeConfig
from traversability.terrasentia_utils.rosbag_data_extractor import DataExtractor
from dataset_functions import save_observations_from_rosbag, read_states_from_rosbag, create_directories
import argparse

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Create dataset from rosbag files.')

    # Add input and output directory arguments
    parser.add_argument('--input_dir', type=str, help='Path to the directory containing the rosbag files.')
    parser.add_argument('--output_dir', type=str, help='Path to the directory where the dataset will be saved.')

    # Parse the arguments
    args = parser.parse_args()

    # Load config parameters
    configs = UnitreeConfig()

    print('training bags list:\n', configs.train_bags_list)
    print('validation bags list:\n', configs.valid_bags_list)

    # Set the input and output directories
    source_dir = args.input_dir
    train_data_dir = os.path.join(args.output_dir, 'train/' + configs.robot_name)
    valid_data_dir = os.path.join(args.output_dir, 'valid/' + configs.robot_name)

    # Create directories in case they don't exist
    if not os.path.exists(train_data_dir):
        os.makedirs(train_data_dir)
    if not os.path.exists(valid_data_dir):
        os.makedirs(valid_data_dir)

    for data_dir, bags_list in zip([train_data_dir, valid_data_dir], [configs.train_bags_list, configs.valid_bags_list]):
        # Open the csv file first
        with open(os.path.join(data_dir, 'collections.csv'), 'w') as csv_dataset:
            writer = csv.DictWriter(csv_dataset, fieldnames=['rosbags'])
            writer.writeheader()

            # Then iterate over the rosbags
            for rosbag_file in bags_list:
                bag_name = rosbag_file.split('/')[-1].split('.')[-2]
                working_dir = os.path.join(data_dir, bag_name)
                
                # Write new bag csv file in main csv
                writer.writerow({'rosbags': bag_name})
                # Get rosbag path
                rosbag_path = os.path.join(args.input_dir, rosbag_file)
                # Get data from normal rosbag
                data_obj = DataExtractor(configs, rosbag_path)
                # Create the directories to organize the data
                create_directories(working_dir, configs)
                # Save the observations to a json file
                save_observations_from_rosbag(
                    configs, data_obj, working_dir
                )

                # Call function to read the states
                data_dict = read_states_from_rosbag(configs, data_obj, vel_in_body_frame=False)

                # Open the csv specific to this rosbag to save the states
                with open(os.path.join(working_dir, 'states.csv'), 'w') as csv_rosbag:
                    fieldnames = ['timestamp', 'lat_lon', 'position', 'orientation',
                                    'linear_velocity', 'angular_velocity',
                                    'command', 'traversability', 'voltage', 'current']
                    dataset_writer = csv.DictWriter(csv_rosbag, fieldnames=fieldnames)
                    dataset_writer.writeheader()

                    # Iterate over the states and save them to the csv file
                    for i in range(data_dict['timestamp'].shape[0]):
                        dataset_writer.writerow({
                            'timestamp': data_dict['timestamp'][i],
                            'lat_lon': data_dict['lat_lon'][i].tolist(),
                            'position': data_dict['position'][i].tolist(),
                            'orientation': data_dict['orientation'][i].tolist(),
                            'linear_velocity': data_dict['linear_velocity'][i].tolist(),
                            'angular_velocity': data_dict['angular_velocity'][i].tolist(),
                            'command': data_dict['command'][i].tolist(),
                            'traversability': [], # Traversability will be added later through optimization
                            'voltage': data_dict['voltage'][i],
                            'current': data_dict['current'][i]
                        })

    print('Done creating dataset!')