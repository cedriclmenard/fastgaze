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

def yaw_pitch_to_vector(yaw_pitchs):
        # n = yaw_pitchs.shape[0]
        sin = torch.sin(yaw_pitchs)
        cos = torch.cos(yaw_pitchs)
        return torch.stack([cos[:, 1] * sin[:, 0], sin[:, 1], cos[:, 1] * cos[:, 0]], 1)

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

class GazeIrisLandmarksAdapted(pl.LightningModule):
    default_weights_path = str(Path(os.path.dirname(__file__), "sample.pth"))
    def __init__(self, config={}, import_from_iris_landmarks_path=None, channels=256):
        self.normalize = config.get("normalize_colors", False)
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        # self.input_width = config.get("input_width", 64)
        # self.input_height = config.get("input_height", 64)

        # Base
        super().__init__()
        self.config = config
        self.lr = config["lr"]
        # selr.lr = self.learning_rate
        self._define_initial_layers()
        
        # self.hparams = {}
        for k in config:
            self.hparams[k] = config[k]
        # self.logger.log_hyperparams(config)

        # if not import_from_iris_landmarks_path is None:
        #     self._import_from_irislandmarks(import_from_iris_landmarks_path)

        # Get linear output layer for landmarks
        self.iris_landmark_conv = self.split_iris[8]
        self.eye_landmark_conv = self.split_eye[8]

        # recreate the split branches without the linear output layers
        self.split_iris = nn.Sequential(*[self.split_iris[i] for i in range(8)])
        self.split_eye = nn.Sequential(*[self.split_eye[i] for i in range(8)])

        # Combined
        channels=256
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
            nn.Identity() if self.config.get("no_batchnorm") else _norm2d_layer_factory(256, self.config),
            *[IBlock(256,256, 
                dropout=config["train_options"]["dropout"], 
                bn_splits=self.config["train_options"].get("bn_splits", 0), 
                num_groups=self.config["train_options"].get("num_groups", 0), 
                dropout_gaussian=config["train_options"].get("dropout_gaussian", False)) 
                for _ in range(self.config["architecture"]["split_gaze"]["n_layers"][0])
            ],
            IBlock(256, 256, 
                stride=2, 
                dropout=config["train_options"]["dropout"], 
                bn_splits=self.config["train_options"].get("bn_splits", 0), 
                num_groups=self.config["train_options"].get("num_groups", 0)
            ),
            *[IBlock(256,256, 
                dropout=config["train_options"]["dropout"], 
                bn_splits=self.config["train_options"].get("bn_splits", 0), 
                num_groups=self.config["train_options"].get("num_groups", 0), 
                dropout_gaussian=config["train_options"].get("dropout_gaussian", False)) 
                for _ in range(self.config["architecture"]["split_gaze"]["n_layers"][1])
            ],
            IBlock(256, 256, 
                stride=2, 
                dropout=config["train_options"]["dropout"], 
                bn_splits=self.config["train_options"].get("bn_splits", 0), 
                num_groups=self.config["train_options"].get("num_groups", 0), 
                dropout_gaussian=config["train_options"].get("dropout_gaussian", False)
            ),
            *[IBlock(256,256, 
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
                x = (x_lr[side] - self.mean) / self.std
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
                num_batches_per_epoch=self.config["num_batches_per_epoch"]
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
        yaw_pitch = batch["yaw_pitch"].type(inputs[0]["left"].dtype)

        if batch.get("gaze") is None:
            real_direction = -0.5 * (batch["left_gaze"] + batch["right_gaze"])
        else:
            real_direction = batch["gaze"].type(inputs[0]["left"].dtype)#.detach()
        real_direction /= torch.norm(real_direction, dim=1)[:, None]

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