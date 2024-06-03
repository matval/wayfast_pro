import wandb
import torch
import numpy as np
import torchvision
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from models.traversability_net import EnergyNet
from models.dynamic_net import DynamicNet

class TrainingModule(pl.LightningModule):
    """
    A training module for a neural network model, built on PyTorch Lightning.

    This module initializes the model with given configurations, prepares the model for training and inference, 
    and sets up necessary parameters and hyperparameters.

    Attributes:
        eps (float): A small number added to prevent division by zero or log(0).
        gamma (torch.Tensor): Discount factors for future rewards.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay factor for L2 regularization.
        train_depth (bool): Flag indicating if depth information is used for training.
        predict_depth (bool): Flag indicating if the model predicts depth information.
        model (EnergyNet): The neural network model for processing inputs.
        grid_bounds (dict): The boundary definitions for the output map.
        map_size (tuple): The size of the output map, derived from grid_bounds.
        map_origin (tuple): The origin point of the output map.
        map_resolution (tuple): The resolution of the output map.
        vis_interval (int): Interval at which visualization is updated during training.
    """
    def __init__(self, configs):
        """
        Initializes the TrainingModule with given configurations.

        Args:
            configs: Configuration object containing model, training, and optimizer settings.
        """
        super().__init__()
        # Save hyperparamters to hparams.yaml
        self.save_hyperparameters()
        
        # Initialize a small epsilon value to avoid numerical issues
        self.eps = 1e-6
        self.dt = configs.TRAINING.DT

        # Prepare discount factors (gamma) for future and past rewards computation
        # indices = torch.arange(-configs.TRAINING.HORIZON//2, configs.TRAINING.HORIZON//2)
        indices = torch.arange(configs.TRAINING.HORIZON//configs.TRAINING.NUM_ROLLOUTS)
        gamma = torch.tensor([configs.TRAINING.GAMMA**abs(i) for i in indices])
        self.gamma = nn.Parameter(gamma, requires_grad=False)

        # Extract learning rate and weight decay from configs
        self.learning_rate = configs.TRAINING.LR
        self.weight_decay = configs.TRAINING.WEIGHT_DECAY
        
        # Determine if ecoder network predicts depth
        self.predict_depth = configs.ENCODER.PREDICT_DEPTH

        # Determine the pooling sizes for the progressive resolution
        self.pooling_size_list = [4, 10, 25, 50, 100, 200]
        
        # Initialize the energy traversability model with specified configurations
        self.model = EnergyNet(
            configs.ENCODER.GRID_BOUNDS,
            configs.ENCODER.INPUT_SIZE,
            downsample          = configs.ENCODER.DOWNSAMPLE,
            image_latent_dim    = configs.ENCODER.LATENT_DIM,
            predict_depth       = configs.ENCODER.PREDICT_DEPTH,
            fuse_pcloud         = configs.ENCODER.FUSE_PCLOUD,
            encoder             = configs.ENCODER.ENCODER,
            max_points_per_pillar = configs.POINT_PILLAR.MAX_POINTS_PER_PILLAR,
            max_pillars         = configs.POINT_PILLAR.MAX_PILLARS,
            flow_latent_dim     = configs.FLOW.LATENT_DIM,
            flow_num_layers     = configs.FLOW.NUM_LAYERS,
        )

        self.dynamic_model = DynamicNet(
            configs.ENCODER.GRID_BOUNDS,
        )

        # Define grid boundaries, size, origin, and resolution for mapping.
        self.grid_bounds = configs.ENCODER.GRID_BOUNDS
        self.map_size = (
            int((self.grid_bounds['xbound'][1] - self.grid_bounds['xbound'][0])/self.grid_bounds['xbound'][2]),
            int((self.grid_bounds['ybound'][1] - self.grid_bounds['ybound'][0])/self.grid_bounds['ybound'][2]))

        self.map_origin = (
            int((self.grid_bounds['xbound'][1])/self.grid_bounds['xbound'][2]),
            int((self.grid_bounds['ybound'][1])/self.grid_bounds['ybound'][2]))

        self.map_resolution = (
            self.grid_bounds['xbound'][2],
            self.grid_bounds['ybound'][2])

        # Setup the interval for updating visualizations during training
        self.vis_interval = configs.TRAINING.VIS_INTERVAL

        # self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step, including forward pass and loss computation.

        This method processes a batch of data, computes the loss for traversability and depth prediction,
        and logs the losses. Optionally, it visualizes the training results at specified intervals.

        Args:
            batch (Tuple[Tensor, ...]): The data for the current batch. Expected to contain:
                - color_img (Tensor): The RGB images.
                - pcloud (Tensor): The point clouds.
                - inv_intrinsics (Tensor): The inverse camera intrinsics.
                - extrinsics (Tensor): The camera extrinsics.
                - path (Tensor): The paths.
                - target_trav (Tensor): The target traversability maps.
                - trav_weights (Tensor): The weights for the traversability loss.
                - depth_target (Tensor): The target depth images.
                - depth_mask (Tensor): The mask for valid depth values.
                - label_img (Tensor): The label images (if used for additional tasks).
                - label_mask (Tensor): The mask for valid label values.
            batch_idx (int): The index of the current batch.

        Returns:
            Tensor: The total loss computed for this training step.

        The method logs training losses and optionally visualizes the predictions if the current batch index
        is a multiple of the visualization interval (`self.vis_interval`).
        """
        # Unpack the batch data
        pointcloud = batch['pointcloud']
        images      = batch['images']
        # pillars     = batch['pillars']
        # voxels      = batch['voxels']
        intrinsics  = batch['intrinsics']
        extrinsics  = batch['extrinsics']
        states      = batch['states']
        commands    = batch['commands']
        power       = batch['power']
        traversability = batch['traversability']
        mask = batch['mask']

        # opt_map, opt_dyn = self.optimizers()
        # opt_map.zero_grad()
        # opt_dyn.zero_grad()

        # Forward pass through the model
        pred_map, debug = self.model(pointcloud, images, intrinsics, extrinsics)

        # Calculate the progressive map size based on the current epoch
        current_stage = min(self.current_epoch // len(self.pooling_size_list), len(self.pooling_size_list)-1)
        # Resize pred_map to the current progressive size
        current_size = self.pooling_size_list[current_stage]
        pred_map = F.adaptive_avg_pool2d(pred_map, current_size)

        init_state = states[:,:,:1]
        # pred_states, pred_power = self.dynamic_model(init_state, commands, pred_map, self.dt)
        pred_states = self.dynamic_model(init_state, commands, pred_map, self.dt)

        self.log("train/m", self.dynamic_model.get_m(), sync_dist=True)
        self.log("train/Ixx", self.dynamic_model.get_I()[0,0], sync_dist=True)
        self.log("train/Ixy", self.dynamic_model.get_I()[0,1], sync_dist=True)
        self.log("train/Ixz", self.dynamic_model.get_I()[0,2], sync_dist=True)
        self.log("train/Iyy", self.dynamic_model.get_I()[1,1], sync_dist=True)
        self.log("train/Iyz", self.dynamic_model.get_I()[1,2], sync_dist=True)
        self.log("train/Izz", self.dynamic_model.get_I()[2,2], sync_dist=True)
        self.log("train/tau_v", self.dynamic_model.get_tau_v(), sync_dist=True)
        self.log("train/tau_omega", self.dynamic_model.get_tau_omega(), sync_dist=True)
        self.log("train/p_idle", self.dynamic_model.P_idle.data, sync_dist=True)
        self.log("train/eta", self.dynamic_model.get_eta(), sync_dist=True)

        # Calculate loss
        states_loss = self.states_criterion(pred_states, states, mask)
        # power_loss = self.power_criterion(pred_power, power[...,1:], mask[...,1:])
        # Calculate total loss
        loss = states_loss #+ 1e-5*power_loss

        # self.manual_backward(loss)
        # opt_map.step()
        # opt_dyn.step()

        # Visualization at specified intervals
        if (batch_idx % self.vis_interval) == 0:
            self.visualize(images, pointcloud, pred_map, debug, pred_states, states, commands, prefix='train')
    
        # Logging losses to TensorBoard
        self.log("train/loss", loss, sync_dist=True, batch_size=images.size(0))
        self.log("train/states_loss", states_loss, sync_dist=True, batch_size=images.size(0))
        # self.log("train/power_loss", power_loss, sync_dist=True, batch_size=images.size(0))
        # self.log("train/depth_loss", depth_loss, sync_dist=True, batch_size=images.size(0))
        self.log("train/power", power[0,0,0], sync_dist=True, batch_size=images.size(0))
        # self.log("train/pred_power", pred_power[0,0,0], sync_dist=True, batch_size=images.size(0))

        return loss

    def validation_step(self, batch, batch_idx):
        # Unpack the batch data
        pointcloud  = batch['pointcloud']
        images      = batch['images']
        # pillars     = batch['pillars']
        # voxels      = batch['voxels']
        intrinsics  = batch['intrinsics']
        extrinsics  = batch['extrinsics']
        states      = batch['states']
        commands    = batch['commands']
        power       = batch['power']
        traversability = batch['traversability']
        mask = batch['mask']

        # Forward pass through the model
        pred_map, debug = self.model(pointcloud, images, intrinsics, extrinsics)
        init_state = states[:,:,0:1]
        # pred_states, pred_power = self.dynamic_model(init_state, commands, pred_map, self.dt)
        pred_states = self.dynamic_model(init_state, commands, pred_map, self.dt)

        # Calculate loss
        states_loss = self.states_criterion(pred_states, states, mask)
        # power_loss = self.power_criterion(pred_power, power[...,1:], mask[...,1:])
        # Calculate the depth entropy loss
        # depth_loss = torch.mean(
        #     torch.sum(
        #         F.softmax(debug['depth_logits'], dim=2) * F.log_softmax(debug['depth_logits'] + self.eps, dim=2)
        #     , dim=2)
        # )
        # Calculate total loss
        loss = states_loss #+ 1e-5*power_loss #+ 100*depth_loss

        # Visualize results
        if (batch_idx % self.vis_interval) == 0:
            self.visualize(images, pointcloud, pred_map, debug, pred_states, states, commands, prefix='valid')

        # Logging to TensorBoard by default
        self.log("valid/loss", loss, sync_dist=True, batch_size=images.size(0))
        self.log("valid/states_loss", states_loss, sync_dist=True, batch_size=images.size(0))
        # self.log("valid/power_loss", power_loss, sync_dist=True, batch_size=images.size(0))
        # self.log("valid/depth_loss", depth_loss, sync_dist=True, batch_size=images.size(0))
        self.log("valid/power", power[0,0,0], sync_dist=True, batch_size=images.size(0))
        # self.log("valid/pred_power", pred_power[0,0,0], sync_dist=True, batch_size=images.size(0))

        return loss

    def configure_optimizers(self):
        # optimizer_map = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        # optimizer_dyn = torch.optim.AdamW(self.dynamic_model.parameters(), lr=1e-6, weight_decay=self.weight_decay)
        params = list(self.model.parameters()) + list(self.dynamic_model.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        # return optimizer_map, optimizer_dyn

        # Learning rate scheduler
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-5)
        
        return [optimizer], [] #[scheduler]

    def states_criterion(self, prediction, target, mask):
        B,K,T,D = prediction.shape
        error = target - prediction

        # Use cosine distance for orientation
        error[...,2:5] = 1 - torch.cos(target[...,2:5] - prediction[...,2:5])

        # Use gamma for scaling
        gamma = self.gamma.view(1,1,-1,1).expand(B,K,T,D)

        # Calculate the squared error
        sqr_error = gamma * error**2

        # Penalize the orientation and velocity errors less
        sqr_error[...,2:] = sqr_error[...,2:] * 0.1
        
        return torch.mean(sqr_error[mask])
    
    def power_criterion(self, prediction, target, mask):
        B,K,T = prediction.shape
        error = target - prediction

        # Use gamma for scaling
        gamma = self.gamma[1:].view(1,1,-1).expand(B,K,T)

        # Calculate the squared error
        sqr_error = gamma * error**2

        return torch.mean(sqr_error[mask]**2)

    def visualize(self, images, pointcloud, pred_map, debug, pred_states, states, commands, prefix='train'):
        # Visualize the camera inputs
        img = []
        for i in range(images.shape[1]):
            img.append(images[0][i])
        self.logger.log_image(
            key = prefix + "_images",
            images = img)

        # Visualize the point cloud
        # points = pointcloud[0]
        # points[:,3] = (points[:,3] - points[:,3].min())/(points[:,3].max() - points[:,3].min()) * 13 + 1
        # self.logger.experiment.log(
        #     {prefix + "_point_cloud": wandb.Object3D(points.cpu().numpy(), type='lidar')}
        # )

        # Visualize the predicted slopes map
        grid = torchvision.utils.make_grid(pred_map[0,:2].unsqueeze(1), nrow=2, normalize=False, pad_value=1)
        self.logger.log_image(
            key = prefix + "_pred_slopes",
            images = [grid]
        )

        # Visualize the predicted traversability map
        grid = torchvision.utils.make_grid(pred_map[0,2:].unsqueeze(1), nrow=8, normalize=False, pad_value=1)
        self.logger.log_image(
            key = prefix + "_pred_map",
            images = [grid]
        )
        
        # Visualize the predicted depth map
        n_d = (self.grid_bounds['dbound'][1] - self.grid_bounds['dbound'][0])/self.grid_bounds['dbound'][2]
        depth_img = []
        pred_depth = debug['depth_logits'][0]
        for i in range(pred_depth.shape[0]):
            depth_image = torch.argmax(pred_depth[i], dim=0, keepdim=True)/(n_d-1)
            depth_img.append(depth_image)

        self.logger.log_image(
            key = prefix + "_pred_depth",
            images = depth_img)

        # Visualize the debug information
        cam_projections = debug['cam_projections'][0]
        cam_projections = cam_projections.mean(0)
        cam_projections = (cam_projections - cam_projections.min())/(cam_projections.max() - cam_projections.min())
        self.logger.log_image(
            key = prefix + "_cam_projections",
            images = [cam_projections]
        )

        cam_weights = debug['cam_weights'][0]
        cam_weights = cam_weights.mean(0)
        cam_weights = (cam_weights - cam_weights.min())/(cam_weights.max() - cam_weights.min())
        self.logger.log_image(
            key = prefix + "_cam_weights",
            images = [cam_weights]
        )

        pillars = debug['pillars'][0]
        pillars = pillars.mean(0)
        pillars = (pillars - pillars.min())/(pillars.max() - pillars.min())
        self.logger.log_image(
            key = prefix + "_pillars",
            images = [pillars]
        )

        # Set the limits for x and y axes if known
        x_min, x_max = self.grid_bounds['xbound'][0], self.grid_bounds['xbound'][1]
        y_min, y_max = self.grid_bounds['ybound'][0], self.grid_bounds['ybound'][1]

        # Visualize the predicted states
        real_x = states[0,...,0].flatten().cpu().numpy()
        real_y = states[0,...,1].flatten().cpu().numpy()
        pred_x = pred_states[0,...,0].detach().flatten().cpu().numpy()
        pred_y = pred_states[0,...,1].detach().flatten().cpu().numpy()

        # Create a figure and attach a canvas
        fig, ax = plt.subplots()
        canvas = FigureCanvas(fig)

        # Plot real and predicted paths
        ax.scatter(real_x, real_y, c='blue', s=2, label='Real Path', alpha=0.6, edgecolors='none')
        ax.scatter(pred_x, pred_y, c='red', s=2, label='Predicted Path', alpha=0.6, edgecolors='none')

        # Setting axis limits
        ax.set_xlim(1.2*x_min, 1.2*x_max)
        ax.set_ylim(1.2*y_min, 1.2*y_max)

        box = Rectangle((x_min, y_min), width=abs(x_max - x_min), height=abs(y_max - y_min), 
                linewidth=1, edgecolor='green', facecolor='none')
        ax.add_patch(box)

        # Adding labels and title
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Real vs Predicted Paths')
        ax.legend()

        # Draw the canvas and convert to numpy array
        canvas.draw()
        s, (width, height) = canvas.print_to_buffer()
        # Create an array of the right shape and type to hold the data
        image = np.frombuffer(s, dtype=np.uint8).reshape((height, width, 4))

        # Log the image to wandb
        self.logger.experiment.log({
            prefix + "_paths": wandb.Image(image, caption="Real vs Predicted Paths")
        })

        # Clean up the figure to free memory
        fig.clear()
        plt.close(fig)

        # Visualize the commands, velocities, and predicted velocities for each rollout
        fig, axs = plt.subplots(commands.shape[1], 1, figsize=(8, 6*commands.shape[1]))
        canvas = FigureCanvas(fig)
        for i in range(commands.shape[1]):
            ax = axs[i]
            ax.plot(commands[0,i,:,0].detach().cpu().numpy(), label='Commands')
            ax.plot(states[0,i,:,5].detach().cpu().numpy(), label='Velocities')
            ax.plot(pred_states[0,i,:,5].detach().cpu().numpy(), label='Predicted Velocities')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Linear Velocity')
            ax.set_title(f'Rollout {i+1}')
            ax.legend()
        plt.tight_layout()
        canvas.draw()
        s, (width, height) = canvas.print_to_buffer()
        image = np.frombuffer(s, dtype=np.uint8).reshape((height, width, 4))
        self.logger.experiment.log({
            prefix + "_lin_vel": wandb.Image(image, caption="Linear Velocities")
        })
        fig.clear()
        plt.close(fig)

        # Visualize the commands, angular velocities, and predicted angular velocities for each rollout
        fig, axs = plt.subplots(commands.shape[1], 1, figsize=(8, 6*commands.shape[1]))
        canvas = FigureCanvas(fig)
        for i in range(commands.shape[1]):
            ax = axs[i]
            ax.plot(commands[0,i,:,1].detach().cpu().numpy(), label='Commands')
            ax.plot(states[0,i,:,6].detach().cpu().numpy(), label='Angular Velocities')
            ax.plot(pred_states[0,i,:,6].detach().cpu().numpy(), label='Predicted Angular Velocities')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Angular Velocity')
            ax.set_title(f'Rollout {i+1}')
            ax.legend()
        plt.tight_layout()
        canvas.draw()
        s, (width, height) = canvas.print_to_buffer()
        image = np.frombuffer(s, dtype=np.uint8).reshape((height, width, 4))
        self.logger.experiment.log({
            prefix + "_ang_vel": wandb.Image(image, caption="Angular Velocities")
        })
        fig.clear()
        plt.close(fig)
