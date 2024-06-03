import sys
import yaml
import copy
from easydict import EasyDict

_C = EasyDict()

_C.TAG = 'default'

_C.TRAINING = EasyDict()
_C.TRAINING.LR = 3e-4
_C.TRAINING.USE_LEARNING_RATE_DECAY = True
_C.TRAINING.ENTROPY_WEIGHT = 0.5
_C.TRAINING.WEIGHT_DECAY = 1e-7
_C.TRAINING.RUN_FINETUNING = True
_C.TRAINING.EPOCHS = 50
_C.TRAINING.BATCH_SIZE = 16
_C.TRAINING.WORKERS = 8
_C.TRAINING.PRECISION = "bf16-mixed"
_C.TRAINING.DT = 0.1        # time step in seconds
_C.TRAINING.HORIZON = 500   # horizon in number of points
_C.TRAINING.NUM_ROLLOUTS = 10
_C.TRAINING.GAMMA = 0.99
_C.TRAINING.VIS_INTERVAL = 50
_C.TRAINING.VERBOSE = True
_C.TRAINING.OUTPUT_PATH = ""
_C.TRAINING.LOAD_NETWORK = None

_C.POINT_PILLAR = EasyDict()
_C.POINT_PILLAR.MAX_POINTS_PER_PILLAR = 100
_C.POINT_PILLAR.MAX_PILLARS = 12000
_C.POINT_PILLAR.NUM_FEATURES = 7
_C.POINT_PILLAR.NUM_CHANNELS = 64
_C.POINT_PILLAR.DOWNSAMPLE = 2

# encoder model parameters
_C.ENCODER = EasyDict()
_C.ENCODER.ENCODER = 'resnet50'
_C.ENCODER.DOWNSAMPLE = 8
_C.ENCODER.LATENT_DIM = 64
_C.ENCODER.PREDICT_DEPTH = True
_C.ENCODER.FUSE_PCLOUD = True
_C.ENCODER.INPUT_SIZE = (320, 180)
_C.ENCODER.GRID_BOUNDS = {
    'xbound': [-2.0, 8.0, 0.1],
    'ybound': [-5.0, 5.0, 0.1],
    'zbound': [-2.0, 2.0, 0.1],
    'dbound': [ 0.3, 8.0, 0.2]}

# flow model parameters
_C.FLOW = EasyDict()
# The dimension of the model's latent space
_C.FLOW.LATENT_DIM = 16
# The type of normalizing flow to use
_C.FLOW.FLOW_TYPE = "radial" # "maf"
# The certainty budget to allocate in the latent space
_C.FLOW.NUM_LAYERS = 16
# The certainty budget to allocate in the latent space
_C.FLOW.CERTAINTY_BUDGET = "normal"

_C.DATASET = EasyDict()
_C.DATASET.TRAIN_DATA = []
_C.DATASET.VALID_DATA = []
_C.DATASET.CSV_FILE = 'collections.csv'

_C.AUGMENTATIONS = EasyDict()
_C.AUGMENTATIONS.PCLOUD_DROPOUT = 0.3     # probability to drop the pointcloud input

# Set randomization seed
_C.SEED = 42

def merge_cfgs(base_cfg, new_cfg):
    config = copy.deepcopy(base_cfg)
    for key, val in new_cfg.items():
        if key in config:
            if type(config[key]) is EasyDict:
                config[key] = merge_cfgs(config[key], val)
            else:
                config[key] = val
        else:
            sys.exit("key {} doesn't exist in the default configs".format(key))

    return config

def get_cfg(cfg_file):
    cfg = copy.deepcopy(_C)

    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)

    cfg = merge_cfgs(cfg, new_config)

    return cfg