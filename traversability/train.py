import os
import torch
import logging
import argparse
import pytorch_lightning as pl
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# Custom packages
from train.dataloader import Dataset
from train.read_configs import get_cfg
from train.trainer import TrainingModule

def main(configs):
    print('configs:\n', configs)

    logger = logging.getLogger(__name__)
    logging.getLogger("energy_traversability").setLevel(logging.INFO)

    # Fix randomness
    pl.seed_everything(configs.SEED, workers=True)
    logger.info("Using seed %s.", os.getenv("PL_GLOBAL_SEED"))

    print('Pytorch version:', torch.__version__)

    # # Check if Tensor Cores are available
    # if torch.cuda.get_device_capability(0)[0] >= 7 and torch.__version__ >= '2.0.0':
    #     # CUDA device with compute capability 7.0 or higher supports Tensor Cores
    #     torch.set_float32_matmul_precision('medium')
    #     print("Tensor Cores are available. Matmul precision set to 'medium'.")
    # else:
    #     print("Tensor Cores are not available.")

    # Initialize logger
    wandb_logger = WandbLogger(log_model="all")

    # Add parameters to the logger
    wandb_logger.experiment.config.update(
        {
            "seed": os.getenv("PL_GLOBAL_SEED"),
            "epochs": configs.TRAINING.EPOCHS,
            "learning_rate": configs.TRAINING.LR,
            "entropy_weight": configs.TRAINING.ENTROPY_WEIGHT,
            "use_learning_rate_decay": configs.TRAINING.USE_LEARNING_RATE_DECAY,
            "run_finetuning": configs.TRAINING.RUN_FINETUNING,
            "encoder_latent_dim": configs.ENCODER.LATENT_DIM,
            "flow_latent_dim": configs.FLOW.LATENT_DIM,
            "flow_type": configs.FLOW.FLOW_TYPE,
            "flow_layers": configs.FLOW.NUM_LAYERS,
            "certainty_budget": configs.FLOW.CERTAINTY_BUDGET,
        }
    )

    # Image transform
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor()])

    train_dataset = Dataset(configs, configs.DATASET.TRAIN_DATA, train_transform)
    valid_dataset = Dataset(configs, configs.DATASET.VALID_DATA)

    train_loader = DataLoader(
        train_dataset,
        batch_size  = configs.TRAINING.BATCH_SIZE,
        shuffle     = True,
        num_workers = configs.TRAINING.WORKERS,
        pin_memory  = True,
        collate_fn  = train_dataset.collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size  = configs.TRAINING.BATCH_SIZE,
        shuffle     = False,
        num_workers = configs.TRAINING.WORKERS,
        pin_memory  = True,
        collate_fn  = valid_dataset.collate_fn,
    )

    model = TrainingModule(configs)

    # use to load a previously trained network
    if configs.TRAINING.LOAD_NETWORK is not None:
        print('Loading saved network from {}'.format(configs.MODEL.LOAD_NETWORK))
        pretrained_dict = torch.load(configs.TRAINING.LOAD_NETWORK, map_location='cpu')['state_dict']
        model.load_state_dict(pretrained_dict)
        # del checkpoint

    checkpoint_callback = ModelCheckpoint(
        filename            = 'checkpoint_{epoch}_{valid/loss:.4f}',
        monitor             = 'valid/loss',
        mode                = "min")
    
    lr_callback = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        devices                 = -1,
        num_nodes               = 1,
        gradient_clip_val       = 5,
        sync_batchnorm          = True,
        enable_model_summary    = True,
        accelerator             = 'gpu',
        # profiler                = 'pytorch', #'simple',
        logger                  = wandb_logger,
        default_root_dir        = 'checkpoints',
        max_epochs              = configs.TRAINING.EPOCHS,
        precision               = configs.TRAINING.PRECISION,
        strategy                = DDPStrategy(find_unused_parameters=True),
        callbacks               = [checkpoint_callback, lr_callback]
    )

    trainer.fit(model, train_loader, valid_loader)

if __name__ == "__main__":
    # Get arguments
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify path config file')

    args = parser.parse_args()

    # Load default configs and merge with args
    configs = get_cfg(args.cfg_file)

    main(configs)
