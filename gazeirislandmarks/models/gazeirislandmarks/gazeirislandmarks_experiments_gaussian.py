from math import ceil
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ExponentialLR, StepLR
import torchvision
import torchvision.transforms.functional as TF
from pathlib import Path
import os.path
import os
import copy

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import matplotlib.pyplot as plt

import gpytorch
import math

# import warnings


from ..irislandmarks.irislandmarks import IrisBlock, IrisBlockBN, IrisLandmarks
from .densenet import DenseNet3
from ..layers import Dense, GaussianDropout, GhostBatchNorm, SeqNorm, GhostBatchNorm2d, SeqNorm2d

from .gazeirislandmarks import GazeIrisLandmarksSimpleFull, yaw_pitch_to_vector, vector_to_yaw_pitch, angular_distance, _norm2d_layer_factory, _norm_layer_factory, GazeIrisLandmarks


class GaussianProcessLayer(gpytorch.models.ApproximateGP):
    def __init__(self, num_dim, grid_bounds=(-10.,10.), grid_size=64):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size, batch_shape=torch.Size([num_dim])
        )

        # Our base variational strategy is a GridInterpolationVariationalStrategy,
        # which places variational inducing points on a Grid
        # We wrap it with a IndependentMultitaskVariationalStrategy so that our output is a vector-valued GP
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.GridInterpolationVariationalStrategy(
                self, grid_size=grid_size, grid_bounds=[grid_bounds],
                variational_distribution=variational_distribution,
            ), num_tasks=num_dim,
        )
        super().__init__(variational_strategy)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
                )
            )
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.grid_bounds = grid_bounds

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class GazeIrisLandmarksFeatureExtractor(GazeIrisLandmarksSimpleFull):
    def __init__(self, config={}, import_from_iris_landmarks_path=None, channels=256):
        super().__init__(config, import_from_iris_landmarks_path, channels)
        del self.head

    def forward(self, x_lr, headpose_lr):
        if self.config["architecture"].get("skip_headpose"):
            headpose_lr = {
                "left": torch.zeros_like(headpose_lr["left"]),
                "right": torch.zeros_like(headpose_lr["right"]),
            }
        b = x_lr["left"].shape[0]
        inter = {}

        for side in ["left", "right"]:
            if self.feature_extranormalize:
                x = (x_lr[side] - self.mean[None,:,None,None]) / self.std[None,:,None,None]
            else:
                x = x_lr[side]
            x = F.pad(x, [0, 1, 0, 1], "constant", 0)

            inter[side] = self.backbone(x)
        
        x_gaze = torch.cat((inter["left"], inter["right"]), 1)
        g_raw = self.gaze_conv(self.split_gaze(x_gaze))
        return torch.cat((torch.reshape(g_raw, (b, -1)), headpose_lr["left"], headpose_lr["right"]), 1)

class GazeIrisLandmarksGaussianProcess(gpytorch.Module):
    def __init__(self, config={}, import_from_iris_landmarks_path=None, channels=256):
        self.config = config
        grid_bounds=(-10., 10.)
        gpytorch.Module.__init__(self)
        
        self.feature_extractor = GazeIrisLandmarksFeatureExtractor(config, import_from_iris_landmarks_path, channels)        

        output_features = self.config["architecture"]["split_gaze"]["output_features"]
        face_distances_features = 2 if self.config["architecture"].get("face_distance", False) else 0
        headpose_features = 4
        
        num_dim = output_features + headpose_features + face_distances_features

        self.gp_layer = GaussianProcessLayer(num_dim=num_dim, grid_bounds=grid_bounds)
        self.grid_bounds = grid_bounds
        self.num_dim = num_dim

        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(self.grid_bounds[0], self.grid_bounds[1])

        # setup likelihood
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def setup_marginal_log_likelihood(self, train_dl):
        self.mll = gpytorch.mlls.PredictiveLogLikelihood(self.likelihood, self.gp_layer, num_data=len(train_dl.dataset))

    def forward(self, x_lr, headpose_lr):
        features = self.feature_extractor(x_lr, headpose_lr)

        # Gaussian Process
        features = self.scale_to_bounds(features)

        features = features.transpose(-1,-2).unsqueeze(-1)
        res = self.gp_layer(features)

        return res
    
    def loss_function(self, prediction_yp, truth_yp, prediction_direction, truth_direction):
        return self.mll(prediction_yp, truth_yp)

    def train(self, mode: bool = True):
        super().train(mode=mode)
        self.feature_extractor.train(mode)
        self.likelihood.train(mode)
    
    def eval(self):
        super().eval()
        self.feature_extractor.eval()
        self.likelihood.eval()

    def to(self, device=None, dtype=None, non_blocking=False):
        super().to(device, dtype, non_blocking)
        self.feature_extractor.to(device, dtype, non_blocking)
        self.likelihood.to(device, dtype, non_blocking)
    
    def cuda(self, device=None):
        super().cuda(device)
        self.feature_extractor.cuda(device)
        self.likelihood.cuda(device)

    def cpu(self):
        super().cpu()
        self.feature_extractor.cpu()
        self.likelihood.cpu()

    def configure_optimizers(self, parameters=None):
        if parameters is None:
            # filter parameters
            feature_extractor_parameters = filter(lambda p: p.requires_grad, self.feature_extractor.parameters())
            trainable_parameters = [
                {'params': feature_extractor_parameters, 'weight_decay': self.config["wd"]},
                {'params': self.gp_layer.hyperparameters(), 'lr': self.config["lr"] * 0.01},
                {'params': self.gp_layer.variational_parameters()},
                {'params': self.likelihood.parameters()},            
            ]
        else:
            trainable_parameters = parameters
        if self.config.get("adam", True) and not self.config.get("sgd", False):
            optimizer = optim.Adam(trainable_parameters, 
                lr=(self.lr or self.learning_rate), 
                weight_decay=0., 
                betas=(self.config.get("adam_b1", 0.9), self.config.get("adam_b2", 0.999))
            )
        elif self.config.get("sgd", False):
            optimizer = torch.optim.SGD(trainable_parameters, 
                lr=(self.lr or self.learning_rate), 
                momentum=self.config.get("sgd_momentum", 0.9), 
                weight_decay=0., 
                nesterov=True if self.config.get("sgd_momentum", 0.9) != 0.0 else False
            )
        elif self.config.get("adabound", False):
            import adabound
            optimizer = adabound.AdaBound(trainable_parameters, 
                lr=self.config["lr"], 
                final_lr=0.1, 
                weight_decay=0., 
                amsbound=self.config.get("amsbound", False),
                betas=(self.config.get("adabound_b1", 0.9), self.config.get("adabound_b2", 0.999))
            )
        elif self.config.get("radam", False):
            optimizer = optim.RAdam(trainable_parameters, 
                lr=self.config["lr"],  
                weight_decay=0., 
                betas=(self.config.get("radam_b1", 0.9), self.config.get("radam_b2", 0.999))
            )
        elif self.config.get("ranger", False):
            from ranger21 import Ranger21
            optimizer = Ranger21(trainable_parameters, 
                lr=self.config["lr"],  
                weight_decay=0., 
                betas=(self.config.get("ranger_b1", 0.95), self.config.get("ranger_b2", 0.999)),
                num_epochs=self.config["max_epochs"],
                num_batches_per_epoch=self.config["num_batches_per_epoch"],
                using_normgc=self.config.get("normgc", True),
                # use_madgrad=True
            )
        elif self.config.get("madgradw", False):
            from .madgrad import MADGradW
            optimizer = MADGradW(trainable_parameters, 
                lr=self.config["lr"],  
                weight_decay=0., 
            )
        else:
            raise ValueError("No valid optimizer chosen")
        if not self.config.get("schedule_lr"):
            return optimizer
        else:
            return {
                'optimizer': optimizer,
                'lr_scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config["schedule_lr"], gamma=0.1)
            }

    # def load_weights(self, path):
    #     self.load_state_dict(torch.load(path))
    #     self.eval()
    
    # def load_weights_from_checkpoint(self, path):
    #     data = torch.load(path)
    #     # self.load_state_dict(data["model_state_dict"])
    #     self.load_state_dict(data["state_dict"])
    #     self.eval()


    # @staticmethod
    # def format_batch_for_forward_pass(batch, flip_left=True):
    #     if not all(batch.get("preprocessed", [False])):
    #         if flip_left:
    #             images_l = torchvision.transforms.functional.hflip(batch["image_l"].permute((0,3,1,2))).float() / 255.0
    #         else:
    #             images_l = batch["image_l"].permute((0,3,1,2)).float() / 255.0
    #         images_r = batch["image_r"].permute((0,3,1,2)).float() / 255.0
    #     else:
    #         images_l = batch["image_l"]
    #         images_r = batch["image_r"]
    #     inputs = ({"left": images_l, "right": images_r}, {"left": batch["left_head_yaw_pitch"].type(images_l.dtype), "right": batch["right_head_yaw_pitch"].type(images_l.dtype)})
    #     return inputs

    def process_step(self, batch):
        pass
        # inputs = type(self).format_batch_for_forward_pass(batch, self.config.get("flip_left", True))
        # outputs = self.forward(*inputs)
        # if batch.get("yaw_pitch") is None:
        #     # yaw_pitch = 0.5 * (batch["left_yaw_pitch"].type(inputs[0]["left"].dtype) + batch["left_yaw_pitch"].type(inputs[0]["left"].dtype))
        #     yaw_pitch = vector_to_yaw_pitch(-F.normalize(yaw_pitch_to_vector(batch["left_yaw_pitch"].type(inputs[0]["left"].dtype)) + yaw_pitch_to_vector(batch["right_yaw_pitch"].type(inputs[0]["left"].dtype)), dim=1))
        # else:
        #     yaw_pitch = batch["yaw_pitch"].type(inputs[0]["left"].dtype)

        # if batch.get("gaze") is None:
        #     real_direction = -0.5 * (batch["left_gaze"] + batch["right_gaze"])
        # else:
        #     real_direction = batch["gaze"].type(inputs[0]["left"].dtype)#.detach()
        # real_direction /= torch.norm(real_direction, dim=1)[:, None]

        # # yaw_pitch = vector_to_yaw_pitch(real_direction).type(inputs[0]["left"].dtype)

        # estimated_yaw_pitch = outputs[2]
        # direction = F.normalize(yaw_pitch_to_vector(estimated_yaw_pitch), dim=1)
        # # direction /= torch.norm(direction, dim=1)[:, None]
        # loss = self.loss_function(outputs[2], yaw_pitch, direction, real_direction)
        # if self.config["person_loss"]:
        #     loss += 1e-2*self.loss_consistency(outputs[3])
        
            
        # error = torch.rad2deg(torch.acos(torch.sum(direction.detach()*real_direction, dim=1)))
        # mean_error = error[error.isnan().logical_not()].mean() # This does not count NaNs

        # return loss, mean_error.detach()

    def generate_histograms(self, writer, global_step, output_grads=True):
        self.feature_extractor.generate_histograms(writer, global_step, output_grads)

    def get_normalization_values(self) -> Tuple[float, float]:
        return self.feature_extractor.config.get("normalized_focal_length", 650), self.config.get("normalized_distance", 600)
