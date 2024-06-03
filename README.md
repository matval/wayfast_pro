# Energy-based Traversability Prediction with Uncertainty Estimation

A traversability prediction method with Class Activation Maps for sparse training.

![outline](images/WayFASTER.png)

## Introduction
The code and trained models of:

**WayFASTER: a Self-Supervised Traversability Prediction for Increased, [Mateus V. Gasparino](https://scholar.google.com/citations?user=UbtCA90AAAAJ&hl=en), [Arun N. Sivakumar](https://scholar.google.com/citations?user=peIOOn8AAAAJ&hl=en) and [Girish Chowdhary](https://scholar.google.com/citations?user=pf2zAXkAAAAJ&hl=en), ICRA 2024** [[Paper]]()

<p align="justify">
    We presented WayFASTER, a novel method for self-supervised traversability estimation that uses sequential information to predict a map that improves the traversability map visibility. For such, we use a neural network model that takes a sequence of RGB and depth images as input, and uses the camera’s intrinsic and extrinsic parameters to project the information to a 3D space and predict a 2D traversability map.
</p>

## Citation
If you find the code useful, please consider citing our paper using the following BibTeX entry.
```bibtex
@article{gasparino2024wayfaster,
  title={WayFASTER: a Self-Supervised Traversability Prediction for Increased Navigation Awareness},
  author={Gasparino, Mateus Valverde and Sivakumar, Arun Narenthiran and Chowdhary, Girish},
  journal={arXiv preprint arXiv:2402.00683},
  year={2024}
}
```

## System requirements
- Linux (Tested on Ubuntu 20.04)
- Python3 (Tested using Python 3.8) 
- PyTorch (Tested using Pytorch 1.13.1) 
- CUDA (Tested using CUDA 12.0)

## Repository overview

```
repository_root/
│
├── nav_pkg/
│   ├── nav_pkg/
│   │   ├── __init__.py    # Makes nav_pkg a Python package
│   │   ├── network.py     # Example: PyTorch model definitions
│   │   ├── train.py       # Training scripts
│   │   └── utils.py       # Utility functions
│   ├── setup.py           # Setup script for installing nav_pkg
│   └── requirements.txt   # Python dependencies
│
└── ros_nav_pkg/
    ├── CMakeLists.txt     # ROS package metadata and build config
    ├── package.xml        # ROS package manifest
    ├── scripts/           # Scripts that use nav_pkg
    │   └── run_navigation.py
    └── src/               # Source files for ROS nodes
```

## Creating the dataset

a. Run roscore
```shell
roscore
```

b. Run script
```shell
python3 traversability/dataset/create_dataset_unitree.py --input_dir=/home/mateus/unitree_rosbags --output_dir=../dataset/
```

## Installation

a. Create a python virtual environment and activate it.
```shell
python3 -m virtualenv env
source env/bin/activate
```
b. Install Pytorch.
```shell
pip3 install torch==2.2.2+cu118 torchvision==0.17.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```
c. Install Pybind11 package.
```shell
pip3 install pybind11
```
d. Install the `energy_traversability` package.
```shell
cd traversability
python3 setup.py develop
```

## WayFASTER dataset
- Download the WayFASTER dataset from [here](https://uofi.app.box.com/s/orehra8yt1xlh9mvv3yx9xe2776phtvx) and extract it to the `dataset` folder, outside of the `wayfaster` folder.
- In the `src/utils/train_config.py` file, update the `DATASET.TRAIN_DATA` and the `DATASET.VALID_DATA` parameter to the path of the dataset folder.

## Code execution
### Configuration parameters and training
The configuration parameters of the model such as the learning rate, batch size, and dataloader options are stored in the `src/utils` folder.
If you intend to modify the model parameters, please do so here. Also, the training and evaluation scripts are stored in the same folder.

### Model and data handling
The network model is stored in the `src/models` folder. The dataset analisys and handling are stored in the `scripts` folder.

To train the model, execute the following command. 
```shell
bash train_wayfaster.sh 
```

## Experimental results

![outline](images/waypoints.png)

## License
This code is released under the [MIT](https://opensource.org/license/mit) for academic usage.
