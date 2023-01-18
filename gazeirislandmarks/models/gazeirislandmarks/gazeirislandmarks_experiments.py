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

# import warnings


from ..irislandmarks.irislandmarks import IrisBlock, IrisBlockBN, IrisLandmarks
from .densenet import DenseNet3
from ..layers import Dense, GaussianDropout, GhostBatchNorm, SeqNorm, GhostBatchNorm2d, SeqNorm2d

from .gazeirislandmarks import GazeIrisLandmarksSimpleFull, yaw_pitch_to_vector, vector_to_yaw_pitch, angular_distance, _norm2d_layer_factory, _norm_layer_factory, GazeIrisLandmarks

# NOTE: This is the headpose at split and no MLP
class GazeIrisLandmarksExperiment(GazeIrisLandmarks):
    default_weights_path = str(Path(os.path.dirname(__file__), "sample.pth"))
    def __init__(self, config={}, import_from_iris_landmarks_path=None, channels=256):
        # Base
        super().__init__(config, import_from_iris_landmarks_path, channels)

        del self.split_eye
        del self.split_iris
        del self.eye_landmark_conv
        del self.iris_landmark_conv

        # head layers
        output_features = 2 # yaw pitch
        face_distances_features = 2 if self.config["architecture"].get("face_distance", False) else 0
        if self.config["architecture"].get("headpose", True):
            headpose_features = 4
        else:
            headpose_features = 0

        real_channels = channels + headpose_features

        self.gaze_conv = nn.Sequential(nn.Conv2d(real_channels, output_features, kernel_size=2, stride=2), nn.Flatten())


        if config.get("no_batchnorm"):
            IBlock = IrisBlock
        else:
            IBlock = IrisBlockBN

        if config["train_options"].get("dropout_split", 0) == 0.0:
            dropout_split_layer = nn.Identity()
        elif not config["train_options"].get("dropout_split_gaussian", False):
            dropout_split_layer = nn.Dropout2d(p=config["train_options"].get("dropout_split", 0))
        else:
            dropout_split_layer = GaussianDropout(p=config["train_options"].get("dropout_split", 0))
        
        self.split_gaze = nn.Sequential(
            dropout_split_layer,
            nn.Identity() if config.get("no_batchnorm") else _norm2d_layer_factory(real_channels, config),
            *[IBlock(real_channels, real_channels,  
                dropout=config["train_options"]["dropout"], 
                bn_splits=config["train_options"].get("bn_splits", 0), 
                num_groups=config["train_options"].get("num_groups", 0), 
                dropout_gaussian=config["train_options"].get("dropout_gaussian", False)) 
                for _ in range(config["architecture"]["split_gaze"]["n_layers"][0])
            ],
            IBlock(real_channels,  real_channels,  
                stride=2, 
                dropout=config["train_options"]["dropout"], 
                bn_splits=config["train_options"].get("bn_splits", 0), 
                num_groups=config["train_options"].get("num_groups", 0)
            ),
            *[IBlock(real_channels, real_channels,  
                dropout=config["train_options"]["dropout"], 
                bn_splits=config["train_options"].get("bn_splits", 0), 
                num_groups=config["train_options"].get("num_groups", 0), 
                dropout_gaussian=config["train_options"].get("dropout_gaussian", False)) 
                for _ in range(config["architecture"]["split_gaze"]["n_layers"][1])
            ],
            IBlock(real_channels,  real_channels,  
                stride=2, 
                dropout=config["train_options"]["dropout"], 
                bn_splits=config["train_options"].get("bn_splits", 0), 
                num_groups=config["train_options"].get("num_groups", 0), 
                dropout_gaussian=config["train_options"].get("dropout_gaussian", False)
            ),
            *[IBlock(real_channels, real_channels,  
                dropout=config["train_options"]["dropout"], 
                bn_splits=config["train_options"].get("bn_splits", 0), 
                num_groups=config["train_options"].get("num_groups", 0), 
                dropout_gaussian=config["train_options"].get("dropout_gaussian", False)) 
                for _ in range(config["architecture"]["split_gaze"]["n_layers"][2])
            ],
        )

        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x_lr, headpose_lr):
        if self.config["architecture"].get("skip_headpose"):
            headpose_lr = {
                "left": torch.zeros_like(headpose_lr["left"]),
                "right": torch.zeros_like(headpose_lr["right"]),
            }
        b = x_lr["left"].shape[0]
        inter = {}

        for side in ["left", "right"]:
            if self.normalize:
                x = (x_lr[side] - self.mean[None,:,None,None]) / self.std[None,:,None,None]
            else:
                x = x_lr[side]
            x = F.pad(x, [0, 1, 0, 1], "constant", 0)

            inter[side] = self.backbone(x)
        
        # x_gaze = torch.cat((inter["left"], inter["right"]), 1)

        if self.config.get("headpose", True):
            x_gaze = torch.cat((inter["left"], inter["right"], headpose_lr["left"].unsqueeze(-1).unsqueeze(-1).repeat((1,1,8,8)), headpose_lr["right"].unsqueeze(-1).unsqueeze(-1).repeat((1,1,8,8))), 1)
        else:
            x_gaze = torch.cat((inter["left"], inter["right"]), 1)
        
        g_raw = self.gaze_conv(self.split_gaze(x_gaze))
        # eig = torch.cat((torch.reshape(g_raw, (b, -1)), headpose_lr["left"], headpose_lr["right"]), 1)
        
        # g = self.head(eig)
        # g = g.reshape(b, -1)
        g = g_raw.reshape(b, -1)
        return [None, None, g]

    def _lock_parameters(self):
        pass


def create_backbone(config):
    if config.get("no_batchnorm"):
        IBlock = IrisBlock
    else:
        IBlock = IrisBlockBN
    backbone = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=0, bias=True),
        nn.Identity() if config.get("no_batchnorm") else _norm2d_layer_factory(64, config), 
        nn.PReLU(64),

        IBlock(64, 64,
            dropout=config["train_options"].get("dropout_backbone", 0.0), 
            bn_splits=config["train_options"].get("bn_splits", 0), 
            num_groups=config["train_options"].get("num_groups", 0), 
            dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
        ),
        IBlock(64, 64,
            dropout=config["train_options"].get("dropout_backbone", 0.0), 
            bn_splits=config["train_options"].get("bn_splits", 0), 
            num_groups=config["train_options"].get("num_groups", 0), 
            dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
        ),
        IBlock(64, 64,
            dropout=config["train_options"].get("dropout_backbone", 0.0), 
            bn_splits=config["train_options"].get("bn_splits", 0), 
            num_groups=config["train_options"].get("num_groups", 0), 
            dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
        ),
        IBlock(64, 64,
            dropout=config["train_options"].get("dropout_backbone", 0.0), 
            bn_splits=config["train_options"].get("bn_splits", 0), 
            num_groups=config["train_options"].get("num_groups", 0), 
            dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
        ),
        IBlock(64, 128, stride=2,
            dropout=config["train_options"].get("dropout_backbone", 0.0), 
            bn_splits=config["train_options"].get("bn_splits", 0), 
            num_groups=config["train_options"].get("num_groups", 0), 
            dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
        ),
        IBlock(128, 128,
            dropout=config["train_options"].get("dropout_backbone", 0.0), 
            bn_splits=config["train_options"].get("bn_splits", 0), 
            num_groups=config["train_options"].get("num_groups", 0), 
            dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
        ),
        IBlock(128, 128,
            dropout=config["train_options"].get("dropout_backbone", 0.0), 
            bn_splits=config["train_options"].get("bn_splits", 0), 
            num_groups=config["train_options"].get("num_groups", 0), 
            dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
        ),
        IBlock(128, 128,
            dropout=config["train_options"].get("dropout_backbone", 0.0), 
            bn_splits=config["train_options"].get("bn_splits", 0), 
            num_groups=config["train_options"].get("num_groups", 0), 
            dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
        ),
        IBlock(128, 128,
            dropout=config["train_options"].get("dropout_backbone", 0.0), 
            bn_splits=config["train_options"].get("bn_splits", 0), 
            num_groups=config["train_options"].get("num_groups", 0), 
            dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
        ),
        IBlock(128, 128, stride=2,
            dropout=config["train_options"].get("dropout_backbone", 0.0), 
            bn_splits=config["train_options"].get("bn_splits", 0), 
            num_groups=config["train_options"].get("num_groups", 0), 
            dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
        )
    )
    return backbone

class GazeIrisLandmarksDoubleBackboneExperiment(GazeIrisLandmarksSimpleFull):
    def __init__(self, config={}, import_from_iris_landmarks_path=None, channels=256):
        super().__init__(config, import_from_iris_landmarks_path, channels)
        del self.backbone

        if self.config.get("no_batchnorm"):
            IBlock = IrisBlock
        else:
            IBlock = IrisBlockBN

        self.backbone_left = create_backbone(self.config)
        self.backbone_right = create_backbone(self.config)

        # if self.config["train_options"].get("dropout_backbone", 0) == 0.0:
        #     dropout_backbone_layer = nn.Identity()
        # elif not config["train_options"].get("dropout_backbone_gaussian", False):
        #     dropout_backbone_layer = nn.Dropout2d(p=self.config["train_options"].get("dropout_backbone", 0))
        # else:
        #     dropout_backbone_layer = GaussianDropout(p=self.config["train_options"].get("dropout_backbone", 0))

        # self.backbone_left = 
    
    def forward(self, x_lr, headpose_lr):
        if self.config["architecture"].get("skip_headpose"):
            headpose_lr = {
                "left": torch.zeros_like(headpose_lr["left"]),
                "right": torch.zeros_like(headpose_lr["right"]),
            }
        b = x_lr["left"].shape[0]
        inter = {}

        for side in ["left", "right"]:
            if self.normalize:
                x = (x_lr[side] - self.mean[None,:,None,None]) / self.std[None,:,None,None]
            else:
                x = x_lr[side]
            x = F.pad(x, [0, 1, 0, 1], "constant", 0)

            inter[side] = self.backbone_left(x) if side == "left" else self.backbone_right(x)
        
        x_gaze = torch.cat((inter["left"], inter["right"]), 1)
        g_raw = self.gaze_conv(self.split_gaze(x_gaze))
        eig = torch.cat((torch.reshape(g_raw, (b, -1)), headpose_lr["left"], headpose_lr["right"]), 1)
        
        g = self.head(eig)
        g = g.reshape(b, -1)
        return [None, None, g]





############################ Longer backbone

class GazeIrisLandmarksLongerBackboneExperiment(GazeIrisLandmarksSimpleFull):
    def __init__(self, config={}, import_from_iris_landmarks_path=None, channels=256):
        super().__init__(config, import_from_iris_landmarks_path, channels)

        if config.get("no_batchnorm"):
            IBlock = IrisBlock
        else:
            IBlock = IrisBlockBN
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=0, bias=True),
            nn.Identity() if config.get("no_batchnorm") else _norm2d_layer_factory(16, config), 
            nn.PReLU(16),
            IBlock(16, 16,
                dropout=config["train_options"].get("dropout_backbone", 0.0), 
                bn_splits=config["train_options"].get("bn_splits", 0), 
                num_groups=config["train_options"].get("num_groups", 0), 
                dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
            ),
            IBlock(16, 16,
                dropout=config["train_options"].get("dropout_backbone", 0.0), 
                bn_splits=config["train_options"].get("bn_splits", 0), 
                num_groups=config["train_options"].get("num_groups", 0), 
                dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
            ),
            IBlock(16, 32, stride=1,
                dropout=config["train_options"].get("dropout_backbone", 0.0), 
                bn_splits=config["train_options"].get("bn_splits", 0), 
                num_groups=config["train_options"].get("num_groups", 0), 
                dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
            ),
            IBlock(32, 32,
                dropout=config["train_options"].get("dropout_backbone", 0.0), 
                bn_splits=config["train_options"].get("bn_splits", 0), 
                num_groups=config["train_options"].get("num_groups", 0), 
                dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
            ),
            IBlock(32, 32,
                dropout=config["train_options"].get("dropout_backbone", 0.0), 
                bn_splits=config["train_options"].get("bn_splits", 0), 
                num_groups=config["train_options"].get("num_groups", 0), 
                dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
            ),
            IBlock(32, 64, stride=1,
                dropout=config["train_options"].get("dropout_backbone", 0.0), 
                bn_splits=config["train_options"].get("bn_splits", 0), 
                num_groups=config["train_options"].get("num_groups", 0), 
                dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
            ),
            IBlock(64, 64,
                dropout=config["train_options"].get("dropout_backbone", 0.0), 
                bn_splits=config["train_options"].get("bn_splits", 0), 
                num_groups=config["train_options"].get("num_groups", 0), 
                dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
            ),
            IBlock(64, 64,
                dropout=config["train_options"].get("dropout_backbone", 0.0), 
                bn_splits=config["train_options"].get("bn_splits", 0), 
                num_groups=config["train_options"].get("num_groups", 0), 
                dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
            ),
            IBlock(64, 128, stride=2,
                dropout=config["train_options"].get("dropout_backbone", 0.0), 
                bn_splits=config["train_options"].get("bn_splits", 0), 
                num_groups=config["train_options"].get("num_groups", 0), 
                dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
            ),
            IBlock(128, 128,
                dropout=config["train_options"].get("dropout_backbone", 0.0), 
                bn_splits=config["train_options"].get("bn_splits", 0), 
                num_groups=config["train_options"].get("num_groups", 0), 
                dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
            ),
            IBlock(128, 128,
                dropout=config["train_options"].get("dropout_backbone", 0.0), 
                bn_splits=config["train_options"].get("bn_splits", 0), 
                num_groups=config["train_options"].get("num_groups", 0), 
                dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
            ),
            IBlock(128, 128, stride=2,
                dropout=config["train_options"].get("dropout_backbone", 0.0), 
                bn_splits=config["train_options"].get("bn_splits", 0), 
                num_groups=config["train_options"].get("num_groups", 0), 
                dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
            ),
            IBlock(128, 128,
                dropout=config["train_options"].get("dropout_backbone", 0.0), 
                bn_splits=config["train_options"].get("bn_splits", 0), 
                num_groups=config["train_options"].get("num_groups", 0), 
                dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
            ),
            IBlock(128, 128,
                dropout=config["train_options"].get("dropout_backbone", 0.0), 
                bn_splits=config["train_options"].get("bn_splits", 0), 
                num_groups=config["train_options"].get("num_groups", 0), 
                dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
            ),
        )

# class GazeIrisLandmarksFullFaceExperiment(GazeIrisLandmarksSimpleFull):
#     def __init__(self, config={}, import_from_iris_landmarks_path=None, channels=256):
#         super().__init__(config, import_from_iris_landmarks_path, channels)

#         if config.get("no_batchnorm"):
#             IBlock = IrisBlock
#         else:
#             IBlock = IrisBlockBN
#         self.backbone = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=0, bias=True),
#             nn.Identity() if config.get("no_batchnorm") else _norm2d_layer_factory(16, config), 
#             nn.PReLU(16),
#             IBlock(16, 16,
#                 dropout=config["train_options"].get("dropout_backbone", 0.0), 
#                 bn_splits=config["train_options"].get("bn_splits", 0), 
#                 num_groups=config["train_options"].get("num_groups", 0), 
#                 dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
#             ),
#             IBlock(16, 16,
#                 dropout=config["train_options"].get("dropout_backbone", 0.0), 
#                 bn_splits=config["train_options"].get("bn_splits", 0), 
#                 num_groups=config["train_options"].get("num_groups", 0), 
#                 dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
#             ),
#             IBlock(16, 32, stride=2,
#                 dropout=config["train_options"].get("dropout_backbone", 0.0), 
#                 bn_splits=config["train_options"].get("bn_splits", 0), 
#                 num_groups=config["train_options"].get("num_groups", 0), 
#                 dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
#             ),
#             IBlock(32, 32,
#                 dropout=config["train_options"].get("dropout_backbone", 0.0), 
#                 bn_splits=config["train_options"].get("bn_splits", 0), 
#                 num_groups=config["train_options"].get("num_groups", 0), 
#                 dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
#             ),
#             IBlock(32, 32,
#                 dropout=config["train_options"].get("dropout_backbone", 0.0), 
#                 bn_splits=config["train_options"].get("bn_splits", 0), 
#                 num_groups=config["train_options"].get("num_groups", 0), 
#                 dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
#             ),
#             IBlock(32, 64, stride=2,
#                 dropout=config["train_options"].get("dropout_backbone", 0.0), 
#                 bn_splits=config["train_options"].get("bn_splits", 0), 
#                 num_groups=config["train_options"].get("num_groups", 0), 
#                 dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
#             ),
#             IBlock(64, 64,
#                 dropout=config["train_options"].get("dropout_backbone", 0.0), 
#                 bn_splits=config["train_options"].get("bn_splits", 0), 
#                 num_groups=config["train_options"].get("num_groups", 0), 
#                 dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
#             ),
#             IBlock(64, 64,
#                 dropout=config["train_options"].get("dropout_backbone", 0.0), 
#                 bn_splits=config["train_options"].get("bn_splits", 0), 
#                 num_groups=config["train_options"].get("num_groups", 0), 
#                 dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
#             ),
#             IBlock(64, 128, stride=2,
#                 dropout=config["train_options"].get("dropout_backbone", 0.0), 
#                 bn_splits=config["train_options"].get("bn_splits", 0), 
#                 num_groups=config["train_options"].get("num_groups", 0), 
#                 dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
#             ),
#             IBlock(128, 128,
#                 dropout=config["train_options"].get("dropout_backbone", 0.0), 
#                 bn_splits=config["train_options"].get("bn_splits", 0), 
#                 num_groups=config["train_options"].get("num_groups", 0), 
#                 dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
#             ),
#             IBlock(128, 128,
#                 dropout=config["train_options"].get("dropout_backbone", 0.0), 
#                 bn_splits=config["train_options"].get("bn_splits", 0), 
#                 num_groups=config["train_options"].get("num_groups", 0), 
#                 dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
#             ),
#             IBlock(128, 128, stride=2,
#                 dropout=config["train_options"].get("dropout_backbone", 0.0), 
#                 bn_splits=config["train_options"].get("bn_splits", 0), 
#                 num_groups=config["train_options"].get("num_groups", 0), 
#                 dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
#             ),
#             IBlock(128, 128,
#                 dropout=config["train_options"].get("dropout_backbone", 0.0), 
#                 bn_splits=config["train_options"].get("bn_splits", 0), 
#                 num_groups=config["train_options"].get("num_groups", 0), 
#                 dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
#             ),
#             IBlock(128, 128,
#                 dropout=config["train_options"].get("dropout_backbone", 0.0), 
#                 bn_splits=config["train_options"].get("bn_splits", 0), 
#                 num_groups=config["train_options"].get("num_groups", 0), 
#                 dropout_gaussian=config["train_options"].get("dropout_backbone_gaussian", False)
#             ),
#         )

class GazeIrisLandmarksPinBallLoss(GazeIrisLandmarksSimpleFull):
    def __init__(self, config={}, import_from_iris_landmarks_path=None, channels=256):
        super().__init__(config, import_from_iris_landmarks_path, channels)
        self.quantile_low = 0.1
        self.quantile_high = 1.0 - self.quantile_low

        # change last head layer to out_number = 3
        self.head[-1] = nn.Linear(self.head[-1].in_features, 3, self.head[-1].bias is not None, self.device)
    
    def loss_function(self, prediction_ypv, truth_yp, prediction_direction, truth_direction):
        variance = prediction_ypv[:,2]
        pred_yp = prediction_ypv[:,:2]

        quantile_value_low = truth_yp - (pred_yp - variance[:,None])
        quantile_value_high = truth_yp - (pred_yp + variance[:,None])
        
        loss_low = torch.mean(torch.max(self.quantile_low * quantile_value_low, (self.quantile_low - 1.0)*quantile_value_low))
        loss_high = torch.mean(torch.max(self.quantile_high * quantile_value_high, (self.quantile_high - 1.0)*quantile_value_high))

        return loss_low + loss_high

class GazeIrisLandmarksUncertaintyLoss(GazeIrisLandmarksSimpleFull):
    # FROM: https://towardsdatascience.com/get-uncertainty-estimates-in-neural-networks-for-free-48f2edb82c8f
    def __init__(self, config={}, import_from_iris_landmarks_path=None, channels=256):
        super().__init__(config, import_from_iris_landmarks_path, channels)

        # change last head layer to out_number = 3
        self.head[-1] = nn.Linear(self.head[-1].in_features, 3, self.head[-1].bias is not None, self.device)
    
    def loss_function(self, prediction_ypv, truth_yp, prediction_direction, truth_direction):
        log_dev = prediction_ypv[:,2]
        pred_yp = prediction_ypv[:,:2]

        dev = torch.exp(log_dev)

        if self.config.get("uncertainty_loss_beta") is None:
            return torch.mean(2*log_dev + torch.sum(((pred_yp - truth_yp)/dev[:,None])**2, dim=1))
        else:
            beta = self.config.get("uncertainty_loss_beta") # use beta = 0.0 to get the other behavior
            return torch.mean( torch.pow(dev.detach(), beta) * (log_dev/2.0 + torch.sum(torch.square(pred_yp - truth_yp)/(2.0*dev[:,None]), dim=1)))


class GazeIrisLandmarksHeadposeAtSplit(GazeIrisLandmarksUncertaintyLoss):
    def __init__(self, config={}, import_from_iris_landmarks_path=None, channels=256):
        output_features = 3
        self.headpose_features = 4
        self.embedding_channels = 256
        self.embedding_2d_size = 8

        config["architecture"]["split_gaze"]["output_features"] = output_features
        super().__init__(config, import_from_iris_landmarks_path, channels)

        

        self.headpose_to_embedding = nn.Sequential(
            nn.Linear(self.headpose_features, self.embedding_channels*self.embedding_2d_size*self.embedding_2d_size),
            nn.BatchNorm1d(self.embedding_channels*self.embedding_2d_size*self.embedding_2d_size),
            nn.PReLU(self.embedding_channels*self.embedding_2d_size*self.embedding_2d_size)
        )

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
            if self.normalize:
                x = (x_lr[side] - self.mean[None,:,None,None]) / self.std[None,:,None,None]
            else:
                x = x_lr[side]
            x = F.pad(x, [0, 1, 0, 1], "constant", 0)

            inter[side] = self.backbone(x)

        # deal with headpose
        x_hp = torch.reshape(
            self.headpose_to_embedding(
                torch.cat((headpose_lr["left"], headpose_lr["right"]), 1)
            ),
            (b, self.embedding_channels, self.embedding_2d_size, self.embedding_2d_size)
        )
        
        x_gaze = torch.cat((inter["left"], inter["right"]), 1) + x_hp
        g = self.gaze_conv(self.split_gaze(x_gaze))
        
        g = g.reshape(b, -1)
        return [None, None, g]

class GazeIrisLandmarksHeadposeAtSplitSimple(GazeIrisLandmarksUncertaintyLoss):
    def __init__(self, config={}, import_from_iris_landmarks_path=None, channels=256):
        output_features = 3
        self.headpose_features = 4
        self.embedding_channels = 256
        self.embedding_2d_size = 8

        config["architecture"]["split_gaze"]["output_features"] = output_features
        super().__init__(config, import_from_iris_landmarks_path, channels + self.headpose_features)

        

        self.headpose_to_embedding = nn.Sequential(
            nn.Linear(self.headpose_features, self.headpose_features*self.embedding_2d_size*self.embedding_2d_size),
            nn.BatchNorm1d(self.headpose_features*self.embedding_2d_size*self.embedding_2d_size),
            nn.PReLU(self.headpose_features*self.embedding_2d_size*self.embedding_2d_size)
        )

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
            if self.normalize:
                x = (x_lr[side] - self.mean[None,:,None,None]) / self.std[None,:,None,None]
            else:
                x = x_lr[side]
            x = F.pad(x, [0, 1, 0, 1], "constant", 0)

            inter[side] = self.backbone(x)

        # deal with headpose
        x_hp = torch.reshape(
            self.headpose_to_embedding(
                torch.cat((headpose_lr["left"], headpose_lr["right"]), 1)
            ),
            (b, self.headpose_features, self.embedding_2d_size, self.embedding_2d_size)
        )
        
        x_gaze = torch.cat((inter["left"], inter["right"], x_hp), 1)
        g = self.gaze_conv(self.split_gaze(x_gaze))
        
        g = g.reshape(b, -1)
        return [None, None, g]