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

from typing import (
    Dict,
    Sequence,
    Union,
    Optional,
    List,
    AnyStr,
    Callable
)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import matplotlib.pyplot as plt

# import warnings


from ..irislandmarks.irislandmarks import IrisBlock, IrisBlockBN, IrisLandmarks
from .densenet import DenseNet3
from ..layers import Dense, GaussianDropout, GhostBatchNorm, SeqNorm, GhostBatchNorm2d, SeqNorm2d

def yaw_pitch_to_vector(yaw_pitchs):
        # n = yaw_pitchs.shape[0]
        sin = torch.sin(yaw_pitchs)
        cos = torch.cos(yaw_pitchs)
        return torch.stack([cos[:, 1] * sin[:, 0], sin[:, 1], cos[:, 1] * cos[:, 0]], 1)

def vector_to_yaw_pitch(vectors):
    n = vectors.shape[0]
    out = torch.empty((n, 2), device=vectors.device, dtype=vectors.dtype)
    vectors = torch.divide(vectors, torch.norm(vectors, dim=1).reshape(n, 1))
    out[:, 1] = torch.arcsin(vectors[:, 1])  # theta
    out[:, 0] = torch.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out.squeeze()


def angular_distance(a, b):
        sim = F.cosine_similarity(a, b, eps=1e-6)
        sim = F.hardtanh(sim, -1.0 + 1e-6, 1.0 - 1e-6)
        return torch.rad2deg(torch.acos(sim))

def _norm2d_layer_factory(in_features, config):
    if not config["train_options"].get("bn_splits", 0):
        return nn.BatchNorm2d(in_features)
    elif config["train_options"].get("num_groups", 0):
        return SeqNorm2d(in_features, config["train_options"].get("num_groups", 0), config["train_options"].get("bn_splits", 0))
    else:
        return GhostBatchNorm2d(in_features, config["train_options"].get("bn_splits", 0))

def _norm_layer_factory(in_features, config):
    if not config["train_options"].get("bn_splits", 0):
        return nn.BatchNorm1d(in_features)
    elif config["train_options"].get("num_groups", 0):
        return SeqNorm(in_features, config["train_options"].get("num_groups", 0), config["train_options"].get("bn_splits", 0))
    else:
        return GhostBatchNorm(in_features, config["train_options"].get("bn_splits", 0))

class GazeIrisLandmarks(pl.LightningModule):
    default_weights_path = str(Path(os.path.dirname(__file__), "sample.pth"))
    def __init__(self, config: dict={}, import_from_iris_landmarks_path: str=None, channels: int=256):
        self.normalize = config["augment"].get("norm_colors", False)
        # self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        # self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

        # Base
        super().__init__()
        self.config = config
        self.lr = config["lr"]

        self.skip_headpose: bool = self.config["architecture"].get("skip_headpose", False)
        self.new_norm: bool = self.config.get("new_norm", False)
        self.use_utmv_offset: bool = "utmv" in self.config["hdf5_name"] # hacky but works for now since there isn't a separate flag for utmv
        # selr.lr = self.learning_rate
        self._define_initial_layers()
        
        # self.hparams = {}
        for k in config:
            self.hparams[k] = config[k]
        # self.logger.log_hyperparams(config)

        if import_from_iris_landmarks_path is not None:
            self._import_from_irislandmarks(import_from_iris_landmarks_path)

        self.register_buffer("mean",torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.device), persistent=False)
        self.register_buffer("std",torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.device), persistent=False)

        # Get linear output layer for landmarks
        self.iris_landmark_conv = self.split_iris[8]
        self.eye_landmark_conv = self.split_eye[8]

        # recreate the split branches without the linear output layers
        self.split_iris = nn.Sequential(*[self.split_iris[i] for i in range(8)])
        self.split_eye = nn.Sequential(*[self.split_eye[i] for i in range(8)])

        # Combined
        self.config = config
        for k in config:
            self.hparams[k] = config[k]
        
        # MLP
        if self.config.get("no_batchnorm"):
            IBlock = IrisBlock
        else:
            IBlock = IrisBlockBN

        if self.config["train_options"].get("dropout_split", 0) == 0.0:
            dropout_split_layer = nn.Identity()
        elif not config["train_options"].get("dropout_split_gaussian", False):
            dropout_split_layer = nn.Dropout2d(p=self.config["train_options"].get("dropout_split", 0))
        else:
            dropout_split_layer = GaussianDropout(p=self.config["train_options"].get("dropout_split", 0))

        self.split_gaze = nn.Sequential(
            dropout_split_layer,
            nn.Identity() if self.config.get("no_batchnorm") else _norm2d_layer_factory(channels, self.config),
            *[IBlock(channels,channels, 
                dropout=config["train_options"]["dropout"], 
                bn_splits=self.config["train_options"].get("bn_splits", 0), 
                num_groups=self.config["train_options"].get("num_groups", 0), 
                dropout_gaussian=config["train_options"].get("dropout_gaussian", False)) 
                for _ in range(self.config["architecture"]["split_gaze"]["n_layers"][0])
            ],
            IBlock(channels, channels, 
                stride=2, 
                dropout=config["train_options"]["dropout"], 
                bn_splits=self.config["train_options"].get("bn_splits", 0), 
                num_groups=self.config["train_options"].get("num_groups", 0)
            ),
            *[IBlock(channels,channels, 
                dropout=config["train_options"]["dropout"], 
                bn_splits=self.config["train_options"].get("bn_splits", 0), 
                num_groups=self.config["train_options"].get("num_groups", 0), 
                dropout_gaussian=config["train_options"].get("dropout_gaussian", False)) 
                for _ in range(self.config["architecture"]["split_gaze"]["n_layers"][1])
            ],
            IBlock(channels, channels, 
                stride=2, 
                dropout=config["train_options"]["dropout"], 
                bn_splits=self.config["train_options"].get("bn_splits", 0), 
                num_groups=self.config["train_options"].get("num_groups", 0), 
                dropout_gaussian=config["train_options"].get("dropout_gaussian", False)
            ),
            *[IBlock(channels,channels, 
                dropout=config["train_options"]["dropout"], 
                bn_splits=self.config["train_options"].get("bn_splits", 0), 
                num_groups=self.config["train_options"].get("num_groups", 0), 
                dropout_gaussian=config["train_options"].get("dropout_gaussian", False)) 
                for _ in range(self.config["architecture"]["split_gaze"]["n_layers"][2])
            ],
        )

        self.gaze_conv = nn.Sequential(nn.Conv2d(channels, self.config["architecture"]["split_gaze"]["output_features"], kernel_size=2, stride=2), nn.Flatten())

        if self.config["train_options"].get("dropout_head", 0) == 0.0:
            dropout_head_layer = nn.Identity()
        elif not config["train_options"].get("dropout_head_gaussian", False):
            dropout_head_layer = nn.Dropout2d(p=self.config["train_options"].get("dropout_head", 0))
        else:
            dropout_head_layer = GaussianDropout(p=self.config["train_options"].get("dropout_head", 0))
        
        # head layers
        output_features = self.config["architecture"]["split_gaze"]["output_features"]
        face_distances_features = 2 if self.config["architecture"].get("face_distance", False) else 0
        other_splits_features = 2*(213+15)
        headpose_features = 4

        head_layers = [dropout_head_layer]
        if self.config["architecture"]["head"]["n_layers"] < 2:
            if self.config["architecture"]["head"].get("dense"):
                head_layers.append(
                    Dense(cfg={
                        "in_features": other_splits_features + headpose_features + output_features + face_distances_features,
                        "out_features": 2
                    })
                )
            else:
                head_layers.append(
                    nn.Linear(other_splits_features + headpose_features + output_features + face_distances_features, 2)
                )
        else:
            if self.config["architecture"]["head"].get("dense"):
                head_layers.append(
                    Dense(cfg={
                        "in_features": other_splits_features + headpose_features + output_features + face_distances_features,
                        "out_features": self.config["architecture"]["head"]["hidden_features"]
                    })
                )
                for _ in range(self.config["architecture"]["head"]["n_layers"] - 2):
                    head_layers.append(Dense(cfg={
                        "in_features": self.config["architecture"]["head"]["hidden_features"],
                        "out_features": self.config["architecture"]["head"]["hidden_features"]
                    }))
                head_layers.append(Dense(cfg={
                    "in_features": self.config["architecture"]["head"]["hidden_features"],
                    "out_features": 2
                }))
            else:
                head_layers.append(
                    nn.Linear(other_splits_features + headpose_features + output_features + face_distances_features, self.config["architecture"]["head"]["hidden_features"])
                )
                for _ in range(self.config["architecture"]["head"]["n_layers"] - 2):
                    if self.config["train_options"].get("dropout_within_head", 0.0) != 0.0:
                        head_layers.append(nn.Dropout(p=self.config["train_options"].get("dropout_within_head", 0.0)))
                    head_layers.append(
                        nn.Linear(self.config["architecture"]["head"]["hidden_features"], self.config["architecture"]["head"]["hidden_features"])
                    )
                head_layers.append(
                    nn.Linear(self.config["architecture"]["head"]["hidden_features"], 2)
                )
        self.head = nn.Sequential(*head_layers)

    def forward(self, x_lr, headpose_lr):
        if self.config["architecture"].get("skip_headpose"):
            headpose_lr = {
                "left": torch.zeros_like(headpose_lr["left"]),
                "right": torch.zeros_like(headpose_lr["right"]),
            }
        b = x_lr["left"].shape[0]
        inter = {}

        e_raw_both = {}
        i_raw_both = {}

        e_lr = {}
        i_lr = {}

        for side in ["left", "right"]:
            if self.normalize:
                x = (x_lr[side] - self.mean[None,:,None,None]) / self.std[None,:,None,None]
            else:
                x = x_lr[side]
            x = F.pad(x, [0, 1, 0, 1], "constant", 0)

            inter[side] = self.backbone(x)
            e_raw_both[side] = self.split_eye(inter[side])
            i_raw_both[side] = self.split_iris(inter[side])
            e = self.eye_landmark_conv(e_raw_both[side]) 
            e = e.view(b, -1)
            
            i = self.iris_landmark_conv(i_raw_both[side])
            i = i.reshape(b, -1)
            e_lr[side] = e
            i_lr[side] = i
        
        x_gaze = torch.cat((inter["left"], inter["right"]), 1)
        g_raw = self.gaze_conv(self.split_gaze(x_gaze))
        eig = torch.cat((e_lr["left"], e_lr["right"], i_lr["left"], i_lr["right"], torch.reshape(g_raw, (e_lr["left"].shape[0], -1)), headpose_lr["left"], headpose_lr["right"]), 1)
        
        g = self.head(eig)
        g = g.reshape(b, -1)
        if self.config["person_loss"]:
            return [e_lr, i_lr, g, g_raw[:, :self.config["number_consistent_features"], ...]]
        else:
            return [e_lr, i_lr, g]

    def _define_initial_layers(self):
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=0, bias=True),
            nn.PReLU(64),

            IrisBlock(64, 64),
            IrisBlock(64, 64),
            IrisBlock(64, 64),
            IrisBlock(64, 64),
            IrisBlock(64, 128, stride=2),
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2)
        )
        self.split_eye = nn.Sequential(
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2),
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2),
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            nn.Conv2d(in_channels=128, out_channels=213, kernel_size=2, stride=1, padding=0, bias=True)
        )
        self.split_iris = nn.Sequential(
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2),
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2),
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            nn.Conv2d(in_channels=128, out_channels=15, kernel_size=2, stride=1, padding=0, bias=True)
        )

    def _import_from_irislandmarks(self, irislandmarks_path):
        # Load weights from saved weights of the IrisLandmarks model
        super().load_state_dict(torch.load(irislandmarks_path))

        # Disable grads so as to keep the weights and not modify it during training
        for param in self.parameters():
            param.requires_grad = False

    def _lock_parameters(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        for param in self.split_eye.parameters():
            param.requires_grad = False
        
        for param in self.split_iris.parameters():
            param.requires_grad = False

        for param in self.iris_landmark_conv.parameters():
            param.requires_grad = False
        
        for param in self.eye_landmark_conv.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode=mode)
        # Make sure that whenever we're training, we do not update the initial iris estimation weights
        self._lock_parameters()

    def configure_optimizers(self, parameters=None):
        if parameters is None:
            # filter parameters
            trainable_parameters = filter(lambda p: p.requires_grad, self.parameters())
        else:
            trainable_parameters = parameters
        if self.config.get("adam", True) and not self.config.get("sgd", False):
            optimizer = optim.Adam(trainable_parameters, 
                lr=(self.lr or self.learning_rate), 
                weight_decay=self.config["wd"], 
                betas=(self.config.get("adam_b1", 0.9), self.config.get("adam_b2", 0.999))
            )
        elif self.config.get("sgd", False):
            optimizer = torch.optim.SGD(trainable_parameters, 
                lr=(self.lr or self.learning_rate), 
                momentum=self.config.get("sgd_momentum", 0.9), 
                weight_decay=self.config["wd"], 
                nesterov=True if self.config.get("sgd_momentum", 0.9) != 0.0 else False
            )
        elif self.config.get("adabound", False):
            import adabound
            optimizer = adabound.AdaBound(trainable_parameters, 
                lr=self.config["lr"], 
                final_lr=0.1, 
                weight_decay=self.config["wd"], 
                amsbound=self.config.get("amsbound", False),
                betas=(self.config.get("adabound_b1", 0.9), self.config.get("adabound_b2", 0.999))
            )
        elif self.config.get("radam", False):
            optimizer = optim.RAdam(trainable_parameters, 
                lr=self.config["lr"],  
                weight_decay=self.config["wd"], 
                betas=(self.config.get("radam_b1", 0.9), self.config.get("radam_b2", 0.999))
            )
        elif self.config.get("ranger", False):
            from ranger21 import Ranger21
            optimizer = Ranger21(trainable_parameters, 
                lr=self.config["lr"],  
                weight_decay=self.config["wd"], 
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
                weight_decay=self.config["wd"], 
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

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
    
    def load_weights_from_checkpoint(self, path):
        data = torch.load(path)
        # self.load_state_dict(data["model_state_dict"])
        self.load_state_dict(data["state_dict"])
        self.eval()

    def loss_consistency(self, features):
        """ Assume every sample is of the same person
        """
        # n_persons = torch.max(batch["person"])
        # for p in range(n_persons):
        #     indices = (batch["person"] == p).nonzero()
        #     if len(indices > 2):
        n = torch.norm(features, dim=1)
        # features /= n[:, None]
        return torch.cdist(features.contiguous()/n[:, None], features.contiguous()/n[:, None], p=2).sum()
        # return torch.sum(dist_mat)

    def fold_batch_norm_layers(self):
        new_config = self.config
        new_config["no_batchnorm"] = True
        converted = GazeIrisLandmarks(config=new_config)
        converted.backbone = self.backbone

        converted_i = 0
        if isinstance(self.gaze_conv, nn.Sequential):
            for i in range(len(self.gaze_conv)):
                if isinstance(self.gaze_conv[i], nn.BatchNorm2d):
                    converted.gaze_conv[converted_i] = torch.nn.utils.fuse_conv_bn_eval(self.gaze_conv[i-1], self.gaze_conv[i])
                    continue
                converted.gaze_conv[converted_i] = self.gaze_conv[i]
                converted_i += 1
        else:
            converted.gaze_conv = self.gaze_conv
                


        converted.eye_landmark_conv = self.eye_landmark_conv
        converted.iris_landmark_conv = self.iris_landmark_conv
        converted.split_eye = self.split_eye
        converted.split_iris = self.split_iris

        for i in range(len(self.split_gaze)):
            converted.split_gaze[i] = self.split_gaze[i].fold_bn()

        return converted

    @staticmethod
    def format_batch_for_forward_pass(batch, flip_left=True):
        if not all(batch.get("preprocessed", [False])):
            if flip_left:
                images_l = torchvision.transforms.functional.hflip(batch["image_l"].permute((0,3,1,2))).float() / 255.0
            else:
                images_l = batch["image_l"].permute((0,3,1,2)).float() / 255.0
            images_r = batch["image_r"].permute((0,3,1,2)).float() / 255.0
        else:
            images_l = batch["image_l"]
            images_r = batch["image_r"]
        inputs = ({"left": images_l, "right": images_r}, {"left": batch["left_head_yaw_pitch"].type(images_l.dtype), "right": batch["right_head_yaw_pitch"].type(images_l.dtype)})
        return inputs

    def process_step(self, batch):
        inputs = type(self).format_batch_for_forward_pass(batch, self.config.get("flip_left", True))
        outputs = self.forward(*inputs)
        if batch.get("yaw_pitch") is None:
            # yaw_pitch = 0.5 * (batch["left_yaw_pitch"].type(inputs[0]["left"].dtype) + batch["left_yaw_pitch"].type(inputs[0]["left"].dtype))
            yaw_pitch = vector_to_yaw_pitch(-F.normalize(yaw_pitch_to_vector(batch["left_yaw_pitch"].type(inputs[0]["left"].dtype)) + yaw_pitch_to_vector(batch["right_yaw_pitch"].type(inputs[0]["left"].dtype)), dim=1))
        else:
            yaw_pitch = batch["yaw_pitch"].type(inputs[0]["left"].dtype)

        if batch.get("gaze") is None:
            real_direction = -0.5 * (batch["left_gaze"] + batch["right_gaze"])
        else:
            real_direction = batch["gaze"].type(inputs[0]["left"].dtype)#.detach()
        real_direction /= torch.norm(real_direction, dim=1)[:, None]

        # yaw_pitch = vector_to_yaw_pitch(real_direction).type(inputs[0]["left"].dtype)

        estimated_yaw_pitch = outputs[2]
        direction = F.normalize(yaw_pitch_to_vector(estimated_yaw_pitch), dim=1)
        # direction /= torch.norm(direction, dim=1)[:, None]
        loss = self.loss_function(outputs[2], yaw_pitch, direction, real_direction)
        if self.config["person_loss"]:
            loss += 1e-2*self.loss_consistency(outputs[3])
        
            
        error = torch.rad2deg(torch.acos(torch.sum(direction.detach()*real_direction, dim=1)))
        mean_error = error[error.isnan().logical_not()].mean() # This does not count NaNs

        return loss, mean_error.detach()

    def generate_histograms(self, writer, global_step, output_grads=True):
        for name, weight in self.named_parameters():
            if weight.requires_grad:
                writer.add_histogram(name.replace(".", "/"), weight, global_step)
                if output_grads:
                    writer.add_histogram(f'{name}.grad'.replace(".", "/"), weight.grad, global_step)

    def loss_function(self, prediction_yp, truth_yp, prediction_direction, truth_direction):
        if self.config.get("angular_loss"):
            # loss = angular_distance(prediction_direction, truth_direction).mean() + (10*F.mse_loss(prediction_yp, truth_yp) if self.current_epoch < 3 else 0.0)
            loss = angular_distance(prediction_direction, truth_direction).sum()#.mean() + F.mse_loss(prediction_yp, truth_yp)
        else:
            loss = F.mse_loss(prediction_yp, truth_yp)
        return loss

    def get_normalization_values(self) -> Tuple[float, float]:
        return self.config.get("normalized_focal_length", 650), self.config.get("normalized_distance", 600)
        
        

class GazeIrisLandmarksRegClass(GazeIrisLandmarks):
    default_weights_path = str(Path(os.path.dirname(__file__), "sample_single.pth"))
    def __init__(self, config={}, import_from_iris_landmarks_path=None, channels=256, min_yaw=-90.0, max_yaw=90.0, min_pitch=-90.0, max_pitch=90.0, angular_step=3.0):
        super().__init__(config, import_from_iris_landmarks_path)

        self.pitch = [min_pitch, max_pitch]
        self.yaw = [min_yaw, max_yaw]
        self.angular_step = angular_step
        self.n_steps_pitch = ceil((max_pitch - min_pitch) / angular_step)
        self.n_steps_yaw = ceil((max_yaw - min_yaw) / angular_step)
        self.n_steps = self.n_steps_yaw + self.n_steps_pitch

        if self.config.get("no_batchnorm"):
            IBlock = IrisBlock
        else:
            IBlock = IrisBlockBN

        self.split_gaze = nn.Sequential(
            nn.Dropout2d(p=config["train_options"]["dropout_split"]) if not config["train_options"].get("dropout_split_gaussian", False) else GaussianDropout(p=config["train_options"]["dropout_split"]),
            nn.Identity() if self.config.get("no_batchnorm") else _norm2d_layer_factory(256, self.config),
            *[IBlock(256,256, dropout=config["train_options"]["dropout"], bn_splits=self.config["train_options"].get("bn_splits", 0), num_groups=self.config["train_options"].get("num_groups", 0), dropout_gaussian=config["train_options"].get("dropout_gaussian", False)) for _ in range(self.config["architecture"]["split_gaze"]["n_layers"][0])],
            IBlock(256, 256, stride=2, dropout=config["train_options"]["dropout"], bn_splits=self.config["train_options"].get("bn_splits", 0), num_groups=self.config["train_options"].get("num_groups", 0)),
            *[IBlock(256,256, dropout=config["train_options"]["dropout"], bn_splits=self.config["train_options"].get("bn_splits", 0), num_groups=self.config["train_options"].get("num_groups", 0), dropout_gaussian=config["train_options"].get("dropout_gaussian", False)) for _ in range(self.config["architecture"]["split_gaze"]["n_layers"][1])],
            IBlock(256, 256, stride=2, dropout=config["train_options"]["dropout"], bn_splits=self.config["train_options"].get("bn_splits", 0), num_groups=self.config["train_options"].get("num_groups", 0), dropout_gaussian=config["train_options"].get("dropout_gaussian", False)),
            *[IBlock(256,256, dropout=config["train_options"]["dropout"], bn_splits=self.config["train_options"].get("bn_splits", 0), num_groups=self.config["train_options"].get("num_groups", 0), dropout_gaussian=config["train_options"].get("dropout_gaussian", False)) for _ in range(self.config["architecture"]["split_gaze"]["n_layers"][2])],
        )

        # head layers
        output_features = self.config["architecture"]["split_gaze"]["output_features"]
        face_distances_features = 2 if self.config["architecture"].get("face_distance", False) else 0
        other_splits_features = 2*(213+15)
        headpose_features = 4

        self.gaze_conv = nn.Sequential(nn.Conv2d(channels, output_features, kernel_size=2, stride=2), nn.Flatten())
        head_layers = [nn.Dropout(self.config["train_options"].get("dropout_head", 0)) if not config["train_options"].get("dropout_head_gaussian", False) else GaussianDropout(p=config["train_options"]["dropout_head"])]
        if self.config["architecture"]["head"]["n_layers"] < 2:
            if self.config["architecture"]["head"].get("dense"):
                head_layers.append(
                    Dense(cfg={
                        "in_features": other_splits_features + headpose_features + output_features + face_distances_features,
                        "out_features": self.n_steps
                    })
                )
            else:
                head_layers.append(
                    nn.Linear(other_splits_features + headpose_features + output_features + face_distances_features, self.n_steps)
                )
        else:
            if self.config["architecture"]["head"].get("dense"):
                head_layers.append(
                    Dense(cfg={
                        "in_features": other_splits_features + headpose_features + output_features + face_distances_features,
                        "out_features": self.config["architecture"]["head"]["hidden_features"]
                    })
                )
                for _ in range(self.config["architecture"]["head"]["n_layers"] - 2):
                    head_layers.append(Dense(cfg={
                        "in_features": self.config["architecture"]["head"]["hidden_features"],
                        "out_features": self.config["architecture"]["head"]["hidden_features"]
                    }))
                head_layers.append(Dense(cfg={
                    "in_features": self.config["architecture"]["head"]["hidden_features"],
                    "out_features": self.n_steps
                }))
            else:
                head_layers.append(
                    nn.Linear(other_splits_features + headpose_features + output_features + face_distances_features, self.config["architecture"]["head"]["hidden_features"])
                )
                for _ in range(self.config["architecture"]["head"]["n_layers"] - 2):
                    head_layers.append(
                        nn.Linear(self.config["architecture"]["head"]["hidden_features"], self.config["architecture"]["head"]["hidden_features"])
                    )
                head_layers.append(
                    nn.Linear(self.config["architecture"]["head"]["hidden_features"], self.n_steps)
                )
        self.head = nn.Sequential(*head_layers)

        self.cross_entropy_loss = torch.nn.CrossEntropyLoss().to(self.device)

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

        e_raw_both = {}
        i_raw_both = {}

        e_lr = {}
        i_lr = {}

        for side in ["left", "right"]:
            if self.normalize:
                x = (x_lr[side] - self.mean[None,:,None,None]) / self.std[None,:,None,None]
            else:
                x = x_lr[side]
            x = F.pad(x, [0, 1, 0, 1], "constant", 0)

            inter[side] = self.backbone(x)
            e_raw_both[side] = self.split_eye(inter[side])
            i_raw_both[side] = self.split_iris(inter[side])
            e = self.eye_landmark_conv(e_raw_both[side]) 
            e = e.view(b, -1)
            
            i = self.iris_landmark_conv(i_raw_both[side])
            i = i.reshape(b, -1)
            e_lr[side] = e
            i_lr[side] = i
        
        # if self.config.get("headpose"):
        #     x_gaze = torch.cat((inter["left"], inter["right"], headpose_lr["left"].unsqueeze(-1).unsqueeze(-1).repeat((1,1,8,8)), headpose_lr["right"].unsqueeze(-1).unsqueeze(-1).repeat((1,1,8,8))), 1)
        # else:
        x_gaze = torch.cat((inter["left"], inter["right"]), 1)
        g_raw = self.gaze_conv(self.split_gaze(x_gaze))
        eig = torch.cat((e_lr["left"], e_lr["right"], i_lr["left"], i_lr["right"], torch.reshape(g_raw, (e_lr["left"].shape[0], -1)), headpose_lr["left"], headpose_lr["right"]), 1)
        
        # g = self.gaze_conv(eig)
        g = self.head(eig)
        g = g.reshape(b, -1)
        return [e_lr, i_lr, g]

    def process_step(self, batch):
        inputs = type(self).format_batch_for_forward_pass(batch, self.config.get("flip_left", True))
        outputs = self.forward(*inputs)
        yaw_pitch = batch["yaw_pitch"].type(inputs[0]["left"].dtype)
        yaw_pitch_bin = torch.empty_like(yaw_pitch, dtype=torch.long) # , dtype=torch.long)
        # create bins
        yaw_pitch_bin[:,0] = torch.bucketize(torch.rad2deg(yaw_pitch[:,0]), torch.arange(start=self.yaw[0], end=self.yaw[1], step=self.angular_step, device=self.device)) - 1
        yaw_pitch_bin[:,1] = torch.bucketize(torch.rad2deg(yaw_pitch[:,1]), torch.arange(start=self.pitch[0], end=self.pitch[1], step=self.angular_step, device=self.device)) - 1
        # yaw_pitch_bin[:,0] = F.one_hot(torch.bucketize(torch.rad2deg(yaw_pitch[:,0]), torch.arange(start=-42, end=42, step=3, device=self.device)) - 1, 28)
        # yaw_pitch_bin[:,1] = F.one_hot(torch.bucketize(torch.rad2deg(yaw_pitch[:,1]), torch.arange(start=-42, end=42, step=3, device=self.device)) - 1, 28)

        real_direction = batch["gaze"].type(inputs[0]["left"].dtype).detach()
        real_direction /= torch.norm(real_direction, dim=1)[:, None]

        estimated_yaw_pitch = outputs[2]

        # cross entropy loss
        loss_yaw_bin = self.cross_entropy_loss(estimated_yaw_pitch[:,:self.n_steps_yaw], yaw_pitch_bin[:,0])
        loss_pitch_bin = self.cross_entropy_loss(estimated_yaw_pitch[:,self.n_steps_yaw:], yaw_pitch_bin[:,1])

        # mse loss
        yaw_predicted = F.softmax(estimated_yaw_pitch[:,:self.n_steps_yaw],dim=1)
        pitch_predicted = F.softmax(estimated_yaw_pitch[:,self.n_steps_yaw:],dim=1)

        yaw_pitch_predicted = torch.empty_like(yaw_pitch)
        yaw_pitch_predicted[:,0] = torch.sum(yaw_predicted * torch.arange(start=0, end=self.n_steps_yaw, device=self.device)) * self.angular_step + self.yaw[0]
        yaw_pitch_predicted[:,1] = torch.sum(pitch_predicted * torch.arange(start=0, end=self.n_steps_pitch, device=self.device)) * self.angular_step + self.pitch[0]

        loss_yaw_mse = F.mse_loss(yaw_pitch_predicted[:,0], torch.rad2deg(yaw_pitch[:,0]))
        loss_pitch_mse = F.mse_loss(yaw_pitch_predicted[:,1], torch.rad2deg(yaw_pitch[:,1]))

        loss_yaw = loss_yaw_bin + loss_yaw_mse
        loss_pitch = loss_pitch_bin + loss_pitch_mse
        loss = loss_yaw + loss_pitch

        direction = yaw_pitch_to_vector(estimated_yaw_pitch).detach()
        direction /= torch.norm(direction, dim=1)[:, None]
        # loss = self.loss_function(outputs[2], yaw_pitchts, direction, real_direction)
        
            
        error = torch.rad2deg(torch.acos(torch.sum(direction*real_direction, dim=1)))
        mean_error = error[error.isnan().logical_not()].mean() # This does not count NaNs

        return loss, mean_error.detach()


class GazeIrisLandmarksSimple(GazeIrisLandmarks):
    default_weights_path = str(Path(os.path.dirname(__file__), "sample.pth"))
    def __init__(self, config: dict={}, import_from_iris_landmarks_path: str=None, channels: int=256):
        # Base
        super().__init__(config, import_from_iris_landmarks_path, channels)

        del self.split_eye
        del self.split_iris
        del self.eye_landmark_conv
        del self.iris_landmark_conv

        # head layers
        output_features = self.config["architecture"]["split_gaze"]["output_features"]
        face_distances_features = 2 if self.config["architecture"].get("face_distance", False) else 0
        headpose_features = 4

        head_layers = [nn.Dropout(self.config["train_options"].get("dropout_head", 0)) if not config["train_options"].get("dropout_head_gaussian", False) else GaussianDropout(p=config["train_options"]["dropout_head"])]
        if self.config["architecture"]["head"]["n_layers"] < 2:
            if self.config["architecture"]["head"].get("dense"):
                head_layers.append(
                    Dense(cfg={
                        "in_features": headpose_features + output_features + face_distances_features,
                        "out_features": 2
                    })
                )
            else:
                head_layers.append(
                    nn.Linear(headpose_features + output_features + face_distances_features, 2)
                )
        else:
            if self.config["architecture"]["head"].get("dense"):
                head_layers.append(
                    Dense(cfg={
                        "in_features": headpose_features + output_features + face_distances_features,
                        "out_features": self.config["architecture"]["head"]["hidden_features"]
                    })
                )
                for _ in range(self.config["architecture"]["head"]["n_layers"] - 2):
                    head_layers.append(Dense(cfg={
                        "in_features": self.config["architecture"]["head"]["hidden_features"],
                        "out_features": self.config["architecture"]["head"]["hidden_features"]
                    }))
                head_layers.append(Dense(cfg={
                    "in_features": self.config["architecture"]["head"]["hidden_features"],
                    "out_features": 2
                }))
            else:
                head_layers.append(
                    nn.Linear(headpose_features + output_features + face_distances_features, self.config["architecture"]["head"]["hidden_features"])
                )
                for _ in range(self.config["architecture"]["head"]["n_layers"] - 2):
                    head_layers.append(
                        nn.Linear(self.config["architecture"]["head"]["hidden_features"], self.config["architecture"]["head"]["hidden_features"])
                    )
                head_layers.append(
                    nn.Linear(self.config["architecture"]["head"]["hidden_features"], 2)
                )
        self.head = nn.Sequential(*head_layers)

    def forward(self, x_lr: Dict[str, torch.Tensor], headpose_lr: Dict[str, torch.Tensor]):
        if self.skip_headpose:
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
            x = F.pad(x, [0, 1, 0, 1], "constant", 0.0)

            inter[side] = self.backbone(x)
        
        x_gaze = torch.cat((inter["left"], inter["right"]), 1)
        g_raw = self.gaze_conv(self.split_gaze(x_gaze))
        eig = torch.cat((torch.reshape(g_raw, (b, -1)), headpose_lr["left"], headpose_lr["right"]), 1)
        
        g = self.head(eig)
        g = g.reshape(b, -1)
        return [None, None, g]

    def _lock_parameters(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

class GazeIrisLandmarksSimpleFull(GazeIrisLandmarksSimple):
    def __init__(self, config: dict={}, import_from_iris_landmarks_path: str=None, channels: int=256):
        super().__init__(config, import_from_iris_landmarks_path, channels)

        if not self.config.get("no_batchnorm"):
            for idx, layer in enumerate(self.backbone):
                if isinstance(layer, IrisBlock):
                    self.backbone[idx] = IrisBlockBN.from_irisblock(
                        layer, 
                        dropout=config["train_options"]["dropout"], 
                        bn_splits=self.config["train_options"].get("bn_splits", 0), 
                        num_groups=self.config["train_options"].get("num_groups", 0), 
                        dropout_gaussian=config["train_options"].get("dropout_gaussian", False)
                    )
        
        for param in self.parameters():
            param.requires_grad = True

    def _lock_parameters(self):
        pass

class GazeIrisLandmarksSimpleFullDoubleInput(GazeIrisLandmarksSimpleFull):
    def __init__(self, config={}, import_from_iris_landmarks_path=None, channels=256):
        super().__init__(config, import_from_iris_landmarks_path, channels)
        if self.config.get("no_batchnorm"):
            IBlock = IrisBlock
        else:
            IBlock = IrisBlockBN

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=128, kernel_size=3, stride=2, padding=0, bias=True),
            nn.Identity if self.config.get("no_batchnorm") else _norm2d_layer_factory(128, self.config),
            nn.PReLU(128),

            IBlock(128, 128,
                dropout=config["train_options"]["dropout"], 
                bn_splits=self.config["train_options"].get("bn_splits", 0), 
                num_groups=self.config["train_options"].get("num_groups", 0), 
                dropout_gaussian=config["train_options"].get("dropout_gaussian", False)
            ),
            IBlock(128, 128,
                dropout=config["train_options"]["dropout"], 
                bn_splits=self.config["train_options"].get("bn_splits", 0), 
                num_groups=self.config["train_options"].get("num_groups", 0), 
                dropout_gaussian=config["train_options"].get("dropout_gaussian", False)
            ),
            IBlock(128, 128,
                dropout=config["train_options"]["dropout"], 
                bn_splits=self.config["train_options"].get("bn_splits", 0), 
                num_groups=self.config["train_options"].get("num_groups", 0), 
                dropout_gaussian=config["train_options"].get("dropout_gaussian", False)
            ),
            IBlock(128, 128,
                dropout=config["train_options"]["dropout"], 
                bn_splits=self.config["train_options"].get("bn_splits", 0), 
                num_groups=self.config["train_options"].get("num_groups", 0), 
                dropout_gaussian=config["train_options"].get("dropout_gaussian", False)
            ),
            IBlock(128, 256, stride=2, 
                dropout=config["train_options"]["dropout"], 
                bn_splits=self.config["train_options"].get("bn_splits", 0), 
                num_groups=self.config["train_options"].get("num_groups", 0), 
                dropout_gaussian=config["train_options"].get("dropout_gaussian", False)
            ),
            IBlock(256, 256,
                dropout=config["train_options"]["dropout"], 
                bn_splits=self.config["train_options"].get("bn_splits", 0), 
                num_groups=self.config["train_options"].get("num_groups", 0), 
                dropout_gaussian=config["train_options"].get("dropout_gaussian", False)
            ),
            IBlock(256, 256,
                dropout=config["train_options"]["dropout"], 
                bn_splits=self.config["train_options"].get("bn_splits", 0), 
                num_groups=self.config["train_options"].get("num_groups", 0), 
                dropout_gaussian=config["train_options"].get("dropout_gaussian", False)
            ),
            IBlock(256, 256,
                dropout=config["train_options"]["dropout"], 
                bn_splits=self.config["train_options"].get("bn_splits", 0), 
                num_groups=self.config["train_options"].get("num_groups", 0), 
                dropout_gaussian=config["train_options"].get("dropout_gaussian", False)
            ),
            IBlock(256, 256,
                dropout=config["train_options"]["dropout"], 
                bn_splits=self.config["train_options"].get("bn_splits", 0), 
                num_groups=self.config["train_options"].get("num_groups", 0), 
                dropout_gaussian=config["train_options"].get("dropout_gaussian", False)
            ),
            IBlock(256, 256, stride=2, 
                dropout=config["train_options"]["dropout"], 
                bn_splits=self.config["train_options"].get("bn_splits", 0), 
                num_groups=self.config["train_options"].get("num_groups", 0), 
                dropout_gaussian=config["train_options"].get("dropout_gaussian", False)
            )
        )
    
    def forward(self, x_lr, headpose_lr):
        if self.config["architecture"].get("skip_headpose"):
            headpose_lr = {
                "left": torch.zeros_like(headpose_lr["left"]),
                "right": torch.zeros_like(headpose_lr["right"]),
            }
        b = x_lr["left"].shape[0]

        x = torch.cat((x_lr["left"], x_lr["right"]), 1)
        x = F.pad(x, [0, 1, 0, 1], "constant", 0)
        x = self.backbone(x)
        g_raw = self.gaze_conv(self.split_gaze(x))

        eig = torch.cat((torch.reshape(g_raw, (b, -1)), headpose_lr["left"], headpose_lr["right"]), 1)
        
        g = self.head(eig)
        g = g.reshape(b, -1)
        return [None, None, g]

class GazeIrisLandmarksSimpleNoHead(GazeIrisLandmarksSimple):
    def __init__(self, config={}, import_from_iris_landmarks_path=None, channels=256):
        super().__init__(config, import_from_iris_landmarks_path, channels)
        del self.head
        self.gaze_conv = nn.Sequential(nn.Conv2d(channels, 2, kernel_size=2, stride=2), nn.Flatten())

    def forward(self, x_lr, headpose_lr):
        b = x_lr["left"].shape[0]
        inter = {}

        for side in ["left", "right"]:
            if self.normalize:
                x = (x_lr[side] - self.mean[None,:,None,None]) / self.std[None,:,None,None]
            else:
                x = x_lr[side]
            x = F.pad(x, [0, 1, 0, 1], "constant", 0)

            inter[side] = self.backbone(x)
        
        x_gaze = torch.cat((inter["left"], inter["right"]), 1)
        g_raw = self.gaze_conv(self.split_gaze(x_gaze))
        # eig = torch.cat((torch.reshape(g_raw, (b, -1)), headpose_lr["left"], headpose_lr["right"]), 1)
        
        # g = self.head(eig)
        # g = g.reshape(b, -1)
        return [None, None, g_raw]


from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
from torch.optim.swa_utils import AveragedModel, SWALR
from time import perf_counter
from pytorch_lightning.core.memory import ModelSummary

from ...utilities.general import print_progress_bar
from ...utilities.torchsummary import summary

import uuid

from torch.profiler import schedule as profile_schedule
from torch.profiler import profile, ProfilerActivity

def generate_grad_histograms(model, writer, step):
    for name, weight in model.named_parameters():
        if weight.requires_grad:
            writer.add_histogram(f'{name}.grad'.replace(".", "/"), weight.grad, step)

def log_grad_norm_scalar(model, writer, step):
    for name, module in model.named_children():
        norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), 2) for p in module.parameters()]), 2)
        writer.add_scalar(f'{name}.norm_grad'.replace(".", "/"), norm, step)

def train_custom(config, train_dls, val_dl, test_dl, progress_bar=True):
    def send_batch_to_device(batch, device):
        for k in batch:
            if torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_epochs = config.get("max_epochs")

    model = model_factory(config, device)

    writer = SummaryWriter(comment=config["name"] + str(uuid.uuid4()))

    config["num_batches_per_epoch"] = sum([len(train_dl) for train_dl in train_dls])

    # Get optimzer
    schedulers = []
    per_batch_schedulers = []
    optim_config = model.configure_optimizers()
    if isinstance(optim_config, dict):
        optimizer = optim_config["optimizer"]
        schedulers = [optim_config["lr_scheduler"]]
    else:
        optimizer = optim_config
    
    if config.get("single_step_lr"):
        single_step_lr = lambda epoch: config["single_step_lr"]["factor"] if epoch > config["single_step_lr"]["start_after"] else 1.0
        if config["single_step_lr"].get("per_epoch", True):
            schedulers.append(torch.optim.lr_scheduler.LambdaLR(optimizer, single_step_lr))
        else:
            per_batch_schedulers.append(torch.optim.lr_scheduler.LambdaLR(optimizer, single_step_lr))
    if config.get("step_lr_until"):
        step_lr_until = lambda epoch: config["step_lr_until"]["factor"] ** (epoch // config["step_lr_until"]["step_size"])  if epoch < config["step_lr_until"]["until"] else config["step_lr_until"]["factor"] ** (config["step_lr_until"]["until"] // config["step_lr_until"]["step_size"])
        if config["step_lr_until"].get("per_epoch", True):
            schedulers.append(torch.optim.lr_scheduler.LambdaLR(optimizer, step_lr_until))
        else:
            per_batch_schedulers.append(torch.optim.lr_scheduler.LambdaLR(optimizer, step_lr_until))
    if config.get("step_lr"):
        if config["step_lr"].get("per_epoch", True):
            schedulers.append(StepLR(optimizer, 
                step_size=config["step_lr"]["step_size"], 
                gamma=config["step_lr"]["factor"],
            ))
        else:
            per_batch_schedulers.append(StepLR(optimizer, 
                step_size=config["step_lr"]["step_size"], 
                gamma=config["step_lr"]["factor"]
            ))
    if config.get("cosine_annealing_lr"):
        schedulers.append(CosineAnnealingLR(optimizer, 
            T_max=config["cosine_annealing_lr"]["t_max"], 
            eta_min=config["cosine_annealing_lr"]["eta_min"]
        ))
        if config["cosine_annealing_lr"].get("per_epoch", True):
            schedulers.append(CosineAnnealingLR(optimizer, 
                T_max=config["cosine_annealing_lr"]["t_max"], 
                eta_min=config["cosine_annealing_lr"]["eta_min"]
            ))
        else:
            per_batch_schedulers.append(CosineAnnealingLR(optimizer, 
                T_max=config["cosine_annealing_lr"]["t_max"], 
                eta_min=config["cosine_annealing_lr"]["eta_min"]
            ))
    if config.get("cosine_annealing_warm_restarts"):
        if config["cosine_annealing_warm_restarts"].get("per_epoch", True):
            schedulers.append(CosineAnnealingWarmRestarts(optimizer, 
                T_0=config["cosine_annealing_warm_restarts"]["t_0"], 
                eta_min=config["cosine_annealing_warm_restarts"]["eta_min"]
            ))
        else:
            per_batch_schedulers.append(CosineAnnealingWarmRestarts(optimizer, 
                T_0=config["cosine_annealing_warm_restarts"]["t_0"], 
                eta_min=config["cosine_annealing_warm_restarts"]["eta_min"]
            ))
    if config.get("exponential_lr"):
        if config["exponential_lr"].get("per_epoch", True):
            schedulers.append(ExponentialLR(optimizer, config["exponential_lr"]["gamma"]))
        else:
            per_batch_schedulers.append(ExponentialLR(optimizer, config["exponential_lr"]["gamma"]))
    if config.get("cyclic_lr"):
        if config["cyclic_lr"].get("per_epoch"):
            schedulers.append(torch.optim.lr_scheduler.CyclicLR(optimizer, 
                base_lr=config["lr"], 
                max_lr=config["cyclic_lr"].get("max_lr", config["lr"] * 10), 
                step_size_up=config["cyclic_lr"]["period"]//2, 
                mode=config["cyclic_lr"].get("mode", "triangular"),
                cycle_momentum=False
            ))
        else:
            per_batch_schedulers.append(torch.optim.lr_scheduler.CyclicLR(optimizer, 
                base_lr=config["lr"], 
                max_lr=config["cyclic_lr"].get("max_lr", config["lr"] * 10), 
                step_size_up=config["cyclic_lr"]["period"]//2, 
                mode=config["cyclic_lr"].get("mode", "triangular"),
                cycle_momentum=False
            ))

    if config["swa"]["enabled"]:
        swa_model = AveragedModel(model).to(device)
        if isinstance(config["swa"]["start"], float):
            swa_start = config["swa"]["start"] * max_epochs
        else:
            swa_start = config["swa"]["start"]
        
        if config["swa"].get("scheduler_lr_factor"):
            swa_scheduler = SWALR(optimizer, swa_lr=config["lr"]*config["swa"]["scheduler_lr_factor"])
        else:
            swa_scheduler = SWALR(optimizer, swa_lr=config["swa"]["lr"])
        
    else:
        swa_start = max_epochs + 1

    model.to(device)

    # Do a single forward pass to print summary

    model.eval()
    batch = iter(train_dls[0]).next()
    send_batch_to_device(batch, device)
    # summary(model, type(model).format_batch_for_forward_pass(batch), batch_size=train_dl.batch_size)
    print(ModelSummary(model))

    model.generate_histograms(writer, global_step=-1, output_grads=False)


    # NOTE: TRAINING
    update_step = 0
    # warnings.filterwarnings("ignore")
    last_train_error = None
    last_val_error = None
    current_total_steps = 0

    # def trace_handler(p):
    #     output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    #     print(output)
    #     p.export_chrome_trace("trace_" + str(p.step_num) + ".json")

    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     schedule=torch.profiler.schedule(
    #         wait=10,
    #         warmup=10,
    #         active=10),
    #     on_trace_ready=trace_handler
    # ) as p:
    for e in range(max_epochs):
        total_steps = sum([len(train_dl) for train_dl in train_dls])
        train_loss = 0.0
        train_error = 0.0
        steps = 0
        t1 = perf_counter()
        lr_update_steps = total_steps // config["train_options"]["lr_scheduling_updates_per_epoch"]
        print_progress_bar(0, total_steps, "Train", length=50, print_every_x_percent=1.0)
        if e == swa_start + 1:
            if config["swa"].get("scheduler_lr_factor"):
                swa_scheduler = SWALR(optimizer, swa_lr=config["lr"]*config["swa"]["scheduler_lr_factor"])
            else:
                swa_scheduler = SWALR(optimizer, swa_lr=config["swa"]["lr"])
        for i in torch.randperm(len(train_dls)):
            train_dl = train_dls[i]
            if config.get("split_train_epochs"):
                train_dl.dataset.end_subset()
            for b in train_dl:
                send_batch_to_device(b, device)
                # if steps == 0:
                #     # Output model to tensorboard
                #     writer.add_image('images left eye', torchvision.utils.make_grid(b["image_l"]))
                #     writer.add_image('images right eye', torchvision.utils.make_grid(b["image_r"]))
                #     writer.add_graph(model, b)
                model.train()
                optimizer.zero_grad()
                loss, mean_error = model.process_step(b)
                loss.backward()
                optimizer.step()

                for s in per_batch_schedulers:
                    s.step()
                
                mean_loss = (loss/train_dl.batch_size).detach()
                train_loss += mean_loss
                train_error += mean_error

                if steps % lr_update_steps == 0 and steps != 0:
                    if e > swa_start:
                        # print("updated lr")
                        swa_model.update_parameters(model)
                        if not config["train_options"]["swa"].get("disable_scheduling"):
                            swa_scheduler.step()
                        update_step += 1
                        writer.add_scalar("lr", swa_scheduler.get_last_lr()[0], global_step=update_step)
                    else:
                        if schedulers:
                            # print("updated lr")
                            for s in schedulers:
                                s.step()
                            update_step += 1
                            writer.add_scalar("lr", schedulers[-1].get_last_lr()[0], global_step=update_step)
                    writer.add_scalar("Loss/train_subepoch", mean_loss, global_step=config["train_options"]["lr_scheduling_updates_per_epoch"] * e + update_step)
                    writer.add_scalar("Error/train_subepoch", mean_error, global_step=config["train_options"]["lr_scheduling_updates_per_epoch"] * e + update_step)

                writer.add_scalar("Loss/train", mean_loss, global_step=total_steps) # commented because it overwrites the previous epoch's values
                writer.add_scalar("Error/train", mean_error, global_step=total_steps)
                log_grad_norm_scalar(model, writer, current_total_steps)
                # p.step()
                steps += 1
                current_total_steps += 1
                t2 = perf_counter()
                # print_progress_bar(steps, total_steps, "Train", f'{steps}/{total_steps} [{1/(t2-t1):.2f} it/s] mean loss = {mean_loss:.5f}, mean_error = {mean_error:.2f}')
                print_progress_bar(steps, total_steps, f"Train epoch {e}", f'{steps}/{total_steps} [{steps/(t2-t1):.2f} it/s] mean loss = {mean_loss:.5f}, mean_error = {mean_error:.2f}', length=50, print_every_x_percent=1.0)
                # t1 = t2
        if e > swa_start:
            # print("updated lr")
            swa_model.update_parameters(model)
            swa_scheduler.step()
            update_step += 1
            writer.add_scalar("lr", swa_scheduler.get_last_lr()[0], global_step=update_step)
        else:
            if schedulers:
                # print("updated lr")
                for s in schedulers:
                    s.step()
                update_step += 1
                writer.add_scalar("lr", schedulers[-1].get_last_lr()[0], global_step=update_step)
        writer.add_scalar("Mean_loss/train", train_loss/steps, global_step=e)
        writer.add_scalar("Mean_error/train", train_error/steps, global_step=e)
        last_train_error = train_error/steps
        model.generate_histograms(writer, global_step=e, output_grads=False)
        print()
        print(f'Mean training loss = {train_loss/steps:.5f}, mean training error = {train_error/steps:.2f}')
        print()

        if e > swa_start:
            # update bn running statistics
            steps = 0
            print("Computing new BN due to Stochastic Weight Averaging")
            print_progress_bar(0, total_steps, "BN from averaging", length=50, print_every_x_percent=1.0)
            for b in train_dl:
                send_batch_to_device(b, device)
                swa_model.forward(*type(model).format_batch_for_forward_pass(b))
                steps += 1
                print_progress_bar(steps, total_steps, "BN from averaging", f'{steps}/{total_steps}', length=50, print_every_x_percent=1.0)
            print()
        
        # save after each training epoch if option is toggled
        if config.get("save_each_epoch", True):
            torch.save(
                {
                    "state_dict": swa_model.module.state_dict() if config["swa"]["enabled"] else model.state_dict(),
                    "config": config
                },
                f"{config['name']}_epoch_{e}_mid_training.pth"
            )
            print(f"Saved model under: {os.path.join(os.getcwd(), config['name'])}.pth")

        
        # NOTE: VALIDATION
        if val_dl:
            print("Beginning validation run...")
            if e > swa_start:
                swa_model.eval()
            else:
                model.eval()
            val_loss = 0.0
            val_error = 0.0
            steps = 0
            total_steps = len(val_dl)
            print_progress_bar(0, total_steps, f"Val", length=50, print_every_x_percent=1.0)
            with torch.no_grad():
                t1 = perf_counter()
                for b in val_dl:
                    send_batch_to_device(b, device)
                    if e > swa_start:
                        loss, mean_error = swa_model.module.process_step(b)
                    else:
                        loss, mean_error = model.process_step(b)
                    mean_loss = (loss/val_dl.batch_size).detach()
                    val_loss += mean_loss
                    val_error += mean_error

                    # writer.add_scalar("Loss/val", mean_loss, global_step=steps) # commented because it overwrites the previous epoch's values
                    # writer.add_scalar("Error/val", mean_error, global_step=steps)
                    steps += 1
                    t2 = perf_counter()
                    # t1 = t2
                    # print_progress_bar(steps, total_steps, "Val", f'{steps}/{total_steps} [{1/(t2-t1):.2f} it/s] mean loss = {mean_loss:.5f}, mean_error = {mean_error:.2f}')
                    print_progress_bar(steps, total_steps, f"Val epoch {e}", f'{steps}/{total_steps} [{steps/(t2-t1):.2f} it/s] mean loss = {mean_loss:.5f}, mean_error = {mean_error:.2f}', length=50, print_every_x_percent=1.0)
                
                writer.add_scalar("Mean_loss/val", val_loss/steps, global_step=e)
                writer.add_scalar("Mean_error/val", val_error/steps, global_step=e)
                last_val_error = val_error/steps
                print()
                print(f'Mean validation loss = {val_loss/steps:.5f}, mean validation error = {val_error/steps:.2f}')
                print()
                    


    # NOTE: TEST
    last_test_error = None
    if test_dl:
        print("Beginning test run...")
        if config["swa"]["enabled"]:
            swa_model.eval()
        else:
            model.eval()
        test_loss = 0.0
        test_error = 0.0
        steps = 0
        total_steps = len(test_dl)
        print_progress_bar(0, total_steps, "Test", length=50, print_every_x_percent=1.0)
        with torch.no_grad():
            t1 = perf_counter()
            for b in test_dl:
                send_batch_to_device(b, device)
                if config["swa"]["enabled"]:
                    loss, mean_error = swa_model.module.process_step(b)
                else:
                    loss, mean_error = model.process_step(b)
                mean_loss = (loss/test_dl.batch_size).detach()
                test_loss += mean_loss
                test_error += mean_error

                writer.add_scalar("Loss/test", mean_loss, global_step=steps)
                writer.add_scalar("Error/test", mean_error, global_step=steps)
                steps += 1
                t2 = perf_counter()
                # t1 = t2
                # print_progress_bar(steps, total_steps, "Test", f'{steps}/{total_steps} [{1/(t2-t1):.2f} it/s] mean loss = {mean_loss:.5f}, mean_error = {mean_error:.2f}')
                print_progress_bar(steps, total_steps, "Test", f'{steps}/{total_steps} [{steps/(t2-t1):.2f} it/s] mean loss = {mean_loss:.5f}, mean_error = {mean_error:.2f}', length=50, print_every_x_percent=1.0)

            
            writer.add_scalar("Mean_loss/test", test_loss/steps, global_step=0)
            writer.add_scalar("Mean_error/test", test_error/steps, global_step=0)

            print()
            print(f'Mean test loss = {test_loss/steps:.5f}, mean test error = {test_error/steps:.2f}')
            print()
            if config.get("save"):
                torch.save(
                    {
                        "state_dict": swa_model.module.state_dict() if config["swa"]["enabled"] else model.state_dict(),
                        "config": config
                    },
                    f"{config['name']}.pth"
                )
                print(f"Saved model under: {os.path.join(os.getcwd(), config['name'])}.pth")
            last_test_error = test_error/steps

    final_model = swa_model.module if config["swa"]["enabled"] else model

    # if config["run_options"].get("denorm_run"):
    #     final_model.eval()
    #     with torch.no_grad():
    #         for b in test_dl:
    #             send_batch_to_device(b, device)
    #             final_model
     
    #             loss, mean_error = model.process_step(b)
    #             mean_loss = (loss/test_dl.batch_size).detach()
    #             test_loss += mean_loss
    #             test_error += mean_error

    if test_dl:
        return last_test_error, final_model
    elif val_dl:
        return last_val_error, final_model
    else:
        return last_train_error, final_model
        
            
            



def model_factory(config: dict, device=None) -> GazeIrisLandmarks:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    initialize_from_checkpoint = config.get("init_ckpt")

    initialize_weights_to_irislandmarks = config.get("initialize_weights_to_irislandmarks", True)

    model_type = config.get("type", "normal")
    if model_type == "normal" or model_type == "combined":
        model_class = GazeIrisLandmarks
    elif model_type == "regclass":
        model_class = GazeIrisLandmarksRegClass
    elif model_type == "simple":
        model_class = GazeIrisLandmarksSimple
    elif model_type == "simple_full":
        model_class = GazeIrisLandmarksSimpleFull
    elif model_type == "simple_full_no_head":
        model_class = GazeIrisLandmarksSimpleNoHead
    elif model_type == "simple_full_double_input":
        model_class = GazeIrisLandmarksSimpleFullDoubleInput
    elif model_type == "experimental":
        from .gazeirislandmarks_experiments import GazeIrisLandmarksExperiment
        model_class = GazeIrisLandmarksExperiment
    elif model_type == "exp_double":
        from .gazeirislandmarks_experiments import GazeIrisLandmarksDoubleBackboneExperiment
        model_class = GazeIrisLandmarksDoubleBackboneExperiment
    elif model_type == "exp_longer":
        from .gazeirislandmarks_experiments import GazeIrisLandmarksLongerBackboneExperiment
        model_class = GazeIrisLandmarksLongerBackboneExperiment
    elif model_type == "pinball_loss":
        from .gazeirislandmarks_experiments import GazeIrisLandmarksPinBallLoss
        model_class = GazeIrisLandmarksPinBallLoss
    elif model_type == "uncertainty_loss":
        from .gazeirislandmarks_experiments import GazeIrisLandmarksUncertaintyLoss
        model_class = GazeIrisLandmarksUncertaintyLoss
    elif model_type == "headsplit":
        from .gazeirislandmarks_experiments import GazeIrisLandmarksHeadposeAtSplit
        model_class = GazeIrisLandmarksHeadposeAtSplit
    elif model_type == "headsplit_simple":
        from .gazeirislandmarks_experiments import GazeIrisLandmarksHeadposeAtSplitSimple
        model_class = GazeIrisLandmarksHeadposeAtSplitSimple
    else:
        raise ValueError("model type was not given")

    if initialize_from_checkpoint:
        model = model_class(config)
        model.load_weights_from_checkpoint(initialize_from_checkpoint)
    else:
        model = model_class(config, IrisLandmarks.default_weights_path if initialize_weights_to_irislandmarks else None)

    model.to(device)
    return model

def load_model(path, device=None) -> GazeIrisLandmarks:
    data = torch.load(path, map_location=torch.device(device))
    config = data["config"]
    model = model_factory(config, device)
    model.load_state_dict(data["state_dict"])
    return model
