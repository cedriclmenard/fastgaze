from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor
import torchvision
import higher
import torchmetrics
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from .gazeirislandmarks import yaw_pitch_to_vector, GazeIrisLandmarks, CombinedGazeIrisLandmarks
from ..irislandmarks.irislandmarks import IrisLandmarks

class AngularAccuracy(torchmetrics.CosineSimilarity):
    def compute(self) -> Tensor:
        return torch.acos(super().compute()) * 180 / np.pi

# Adapted from: https://github.com/rcmalli/lightning-maml/blob/main/src/pl/model.py

class BaseModel(pl.LightningModule):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.save_hyperparameters()

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.
        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        raise NotImplementedError

    def step(self, train: bool, batch: Any):
        raise NotImplementedError

    def training_step(self, batch: Any, batch_idx: int):
        outer_loss, inner_loss, outer_acc, inner_acc = self.step(True, batch)

        self.log_dict(
            {"metatrain/inner_loss": inner_loss.item(),
             "metatrain/inner_accuracy": inner_acc.compute()},
            on_epoch=False,
            on_step=True,
            prog_bar=False
        )
        self.log_dict(
            {"metatrain/outer_loss": outer_loss.item(),
             "metatrain/outer_accuracy": outer_acc.compute()},
            on_epoch=False,
            on_step=True,
            prog_bar=True
        )

    def validation_step(self, batch: Any, batch_idx: int):
        torch.set_grad_enabled(True)
        self.cnn.train()
        outer_loss, inner_loss, outer_acc, inner_acc = self.step(False, batch)
        self.log_dict(
            {"metaval/inner_loss": inner_loss.item(),
             "metaval/inner_accuracy": inner_acc.compute()},
            prog_bar=False
        )
        self.log_dict(
            {"metaval/outer_loss": outer_loss.item(),
             "metaval/outer_accuracy": outer_acc.compute()},
            prog_bar=True
        )

    def test_step(self, batch: Any, batch_idx: int):
        torch.set_grad_enabled(True)
        self.cnn.train()
        outer_loss, inner_loss, outer_acc, inner_acc = self.step(False, batch)
        self.log_dict(
            {"metatest/outer_loss": outer_loss.item(),
             "metatest/inner_loss": inner_loss.item(),
             "metatest/inner_accuracy": inner_acc.compute(),
             "metatest/outer_accuracy": outer_acc.compute()},

        )


# batch should have image_l and image_r under "left" and "right" and already preprocessed (float and /255.0)
# should have gaze_direction defined too

class MAMLGazeIrisLandmark(BaseModel):
    def __init__(self, cnn, cfg, *args, **kwargs) -> None:
        super().__init__(cfg=cfg, *args, **kwargs)
        self.cnn = cnn
        self.cnn = self.cnn.to(device=self.device)
        if not cfg.get("inner_sgd"):
            self.inner_optimizer = optim.Adam(self.cnn.parameters(), lr=cfg["outer_lr"], weight_decay=cfg["outer_wd"])
        else:
            self.inner_optimizer = optim.SGD(self.cnn.parameters(), lr=cfg["outer_lr"], momentum=0.9, weight_decay=cfg["outer_wd"])
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        metric = AngularAccuracy(reduction="mean") # TODO: check if really mean
        self.train_inner_accuracy = metric.clone()
        self.train_outer_accuracy = metric.clone()
        self.val_inner_accuracy = metric.clone()
        self.val_outer_accuracy = metric.clone()
        self.test_inner_accuracy = metric.clone()
        self.test_outer_accuracy = metric.clone()

    def forward(self, x):
        return self.cnn(x)

    def step(self, train: bool, batch: Any):
        self.cnn.zero_grad()
        # self.cnn._lock_parameters()
        outer_optimizer = self.optimizers()
        # train_inputs, train_targets = batch['support']
        # test_inputs, test_targets = batch['query']

        # train_inputs = train_inputs.to(device=self.device)
        # train_targets = train_targets.to(device=self.device)
        # test_inputs = test_inputs.to(device=self.device)
        # test_targets = test_targets.to(device=self.device)

        metric = AngularAccuracy(reduction="mean")
        # outer_loss = torch.tensor(0., device=self.device)
        # inner_loss = torch.tensor(0., device='cpu')
        outer_accuracy = metric.clone()
        inner_accuracy = metric.clone()
        # for task_idx, task in enumerate(batch):
            # support = task["support"]
            # query = task["query"]
        support = batch[0]["support"]
        query = batch[0]["query"]
        track_higher_grads = True if train else False
        with higher.innerloop_ctx(self.cnn, self.inner_optimizer,
                                    copy_initial_weights=False,
                                    track_higher_grads=track_higher_grads) as (
                fmodel, diffopt):
            # fmodel._lock_parameters()
            for k in range(self.cfg["num_inner_steps"]):
                support_yaw_pitch = fmodel(support)
                loss = F.mse_loss(support_yaw_pitch, fmodel.select_gaze_angles(support))
                diffopt.step(loss)

            with torch.no_grad():
                support_yaw_pitch = fmodel(support)
                support_preds = yaw_pitch_to_vector(support_yaw_pitch)
                inner_loss = F.mse_loss(support_yaw_pitch,
                                            fmodel.select_gaze_angles(support))
                inner_accuracy.update(support_preds, fmodel.select_gaze_direction(support))

            query_yaw_pitch = fmodel(query)
            outer_loss = F.mse_loss(query_yaw_pitch, fmodel.select_gaze_angles(query))
            with torch.no_grad():
                query_preds = yaw_pitch_to_vector(query_yaw_pitch)
                outer_accuracy.update(query_preds, fmodel.select_gaze_direction(query))

        if train:
            self.manual_backward(outer_loss, outer_optimizer)
            outer_optimizer.step()

        # outer_loss.div_(task_idx + 1)
        # inner_loss.div_(task_idx + 1)

        return outer_loss, inner_loss, outer_accuracy, inner_accuracy

    def configure_optimizers(self):
        # outer_optimizer
        if not self.cfg.get("outer_sgd"):
            outer_optimizer = optim.Adam(self.parameters(), lr=self.cfg["outer_lr"], weight_decay=self.cfg["outer_wd"])
        else:
            outer_optimizer = optim.SGD(self.parameters(), lr=self.cfg["outer_lr"], momentum=0.9, weight_decay=self.cfg["outer_wd"])
        if not self.cfg.get("outer_schedule_lr"):
            return outer_optimizer
        else:
            return {
                'optimizer': outer_optimizer,
                'lr_scheduler': optim.lr_scheduler.StepLR(outer_optimizer, step_size=self.cfg["outer_schedule_lr"], gamma=0.3)
            }

    # def init_meta_train(self):
    #     for child in self.cnn.children():
    #         if isinstance(child, torch.nn.BatchNorm2d):
    #             # for param in child.parameters():
    #             #     param.requires_grad = False
    #             child.eval()

    # def train(self, mode: bool = True):
    #     super().train(mode=mode)
    #     # Make sure that whenever we're training, we do not update the initial iris estimation weights
    #     self.cnn._lock_parameters()

# TODO: finish, using: https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch/blob/master/few_shot_learning_system.py
# line
class MAMLppGazeIrisLandmark(MAMLGazeIrisLandmark):
    def __init__(self, cnn, cfg, *args, **kwargs) -> None:
        super().__init__(cnn, cfg, *args, **kwargs)
        self.cnn = cnn
        self.cfg = cfg
        self.inner_lr_weights = torch.nn.Parameter(torch.tensor([1.0 for _ in range(self.cfg["num_inner_steps"])]))
        

    def step(self, train: bool, batch: Any):
        self.cnn.zero_grad()
        self.zero_grad()
        outer_optimizer = self.optimizers()

        metric = AngularAccuracy(reduction="mean")
        # outer_loss = torch.tensor(0., device=self.device)
        # inner_loss = torch.tensor(0., device='cpu')
        outer_accuracy = metric.clone()
        inner_accuracy = metric.clone()

        support = batch[0]["support"]
        query = batch[0]["query"]
        track_higher_grads = True if train else False
        with higher.innerloop_ctx(self.cnn, self.inner_optimizer,
                                    copy_initial_weights=False,
                                    track_higher_grads=track_higher_grads) as (
                fmodel, diffopt):
            # fmodel._lock_parameters()
            for k in range(self.cfg["num_inner_steps"]):
                support_yaw_pitch = fmodel(support)
                loss = F.mse_loss(support_yaw_pitch, fmodel.select_gaze_angles(support))
                diffopt.step(loss)
                query_yaw_pitch = fmodel(query)
                outer_loss_mamlpp = F.mse_loss(query_yaw_pitch, fmodel.select_gaze_angles(query))
                with torch.no_grad():
                    query_preds = yaw_pitch_to_vector(query_yaw_pitch).detach()
                    outer_accuracy.update(query_preds, fmodel.select_gaze_direction(query))

                if train:
                    self.manual_backward(self.inner_lr_weights[k] * outer_loss_mamlpp, outer_optimizer, retain_graph=True)
                    # outer_loss_mamlpp.grad *= self.inner_lr_weights[k]
                    outer_optimizer.step()

            with torch.no_grad():
                support_yaw_pitch = fmodel(support)
                support_preds = yaw_pitch_to_vector(support_yaw_pitch).detach()
                inner_loss = F.mse_loss(support_yaw_pitch,
                                            fmodel.select_gaze_angles(support))
                inner_accuracy.update(support_preds, fmodel.select_gaze_direction(support))

            query_yaw_pitch = fmodel(query)
            outer_loss = F.mse_loss(query_yaw_pitch, fmodel.select_gaze_angles(query))
            with torch.no_grad():
                query_preds = yaw_pitch_to_vector(query_yaw_pitch).detach()
                outer_accuracy.update(query_preds, fmodel.select_gaze_direction(query))

        # if train:
        #     self.manual_backward(outer_loss, outer_optimizer)
        #     outer_optimizer.step()

        # outer_loss.div_(task_idx + 1)
        # inner_loss.div_(task_idx + 1)

        return outer_loss, inner_loss, outer_accuracy, inner_accuracy

class GazeIrisLandmarksInnerMAML(GazeIrisLandmarks):
    def forward(self, x):
        t = x["right"].dtype
        return super().forward(x["right"], x["right_head_yaw_pitch"].type(t))[2]
    
    def select_gaze_direction(self, data):
        t = data["right"].dtype
        return data["right_gaze"].type(t)
    
    def select_gaze_angles(self, data):
        t = data["right"].dtype
        return data["right_yaw_pitch"].type(t)
    


class CombinedGazeIrisLandmarksInnerMAML(CombinedGazeIrisLandmarks):
    def forward(self, x):
        t = x["right"].dtype
        return super().forward({"left": x["left"], "right": x["right"]}, {"left": x["left_head_yaw_pitch"].type(t), "right": x["right_head_yaw_pitch"].type(t)})[2]
    
    def select_gaze_direction(self, data):
        t = data["right"].dtype
        return 0.5 * (data["right_gaze"].type(t) + data["left_gaze"].type(t))
    
    def select_gaze_angles(self, data):
        t = data["right"].dtype
        return 0.5 * (data["right_yaw_pitch"].type(t) + data["left_yaw_pitch"].type(t))

def train(config, train_dl, val_dl, test_dl, progress_bar=True):
    """train a gazeirislandmark model using parameters given in the config dictionary.
    Configuration options:
        name (str): Base name for model checkpointing. Defaults to "gazeirislandmarks"
        save_path (str): Save path for model checkpointing. Defaults to "checkpoints/"
        max_epochs (int, optional): Maximum number of epochs to run.
        resume_from_checkpoint (str, optional): Path to checkpoint to resume training from.
        initialize_from_checkpoint (str, optional): Path to checkpoint to initialize weights from.
        type (str): Either "single" or "combined", used to select the model class to train.
        no_val (bool, optional): Skip validation checking and model checkpointing based on validation performance. Defaults to False.
        test (bool, optional): Perform modeling testing after complete training. Defaults to True.
        gpus (int, optional): Specify the number of gpus to use. Defaults to 1.
        lr (float, optional): Specify the learning rate.
        wd (float, optional): Specify the weight decay penalty
        angular_loss (bool, optional): Use angular loss function with the yaw and pitch MSE loss
        dropout (float, optional): Specify dropout probability.
        schedule_lr (str, optional): Specify if and what learning rate scheduling to use
        


    Args:
        config (dict): see above for all options and configurations
        train_dl (torch.utils.data.DataLoader): Training dataset dataloader
        val_dl (torch.utils.data.DataLoader): Validation dataset dataloader
        test_dl (torch.utils.data.DataLoader): Test dataset dataloader
        progress_bar (bool, optional): Display a progress bar. Defaults to True.

    Raises:
        ValueError: If any configuration errors are found.
    """
    base_name = config.get("name", "gazeirislandmarks")
    save_path = config.get("save_path", "checkpoints/")
    max_epochs = config.get("max_epochs")
    resume_from_checkpoint = config.get("resume_ckpt")
    initialize_from_checkpoint = config.get("init_ckpt")
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        filename=base_name + '_{epoch}-{metaval/outer_accuracy:.2f}',
        monitor="metaval/outer_accuracy",
        mode="min",
        verbose=True
    )
    if config["type"] == "single":
        model_class = GazeIrisLandmarksInnerMAML
    elif config["type"] == "combined":
        model_class = CombinedGazeIrisLandmarksInnerMAML
    else:
        raise ValueError("model type was not given")
    
    if initialize_from_checkpoint:
        inner_model = model_class(config)
        inner_model.load_weights_from_checkpoint(initialize_from_checkpoint)
    else:
        inner_model = model_class(config, IrisLandmarks.default_weights_path)

    model = MAMLGazeIrisLandmark(inner_model, config)
    # model = MAMLppGazeIrisLandmark(inner_model, config)
    model.automatic_optimization = False

    logger = TensorBoardLogger("tb_logs", name=base_name)
    logger.log_hyperparams(config)
    trainer = pl.Trainer(   gpus=config["gpus"] if config.get("gpus") else 1, 
                            callbacks=[checkpoint_callback] if not config.get("no_val") else None, 
                            max_epochs=max_epochs, 
                            resume_from_checkpoint=resume_from_checkpoint, 
                            logger=logger,
                            progress_bar_refresh_rate=(1 if progress_bar else 0),
                            stochastic_weight_avg=True,
                            # gradient_clip_val=0.5, gradient_clip_algorithm="value",
                            precision=16)
    # )
                            # profiler="simple")
    
    if not config.get("test"):
        trainer.fit(model, train_dl, val_dl)
    else:
        try:
            trainer.fit(model, train_dl, val_dl)
        except KeyboardInterrupt:
            print("Catched Keyboard Interrupt! Stop training and run test")
        finally:
            trainer.test(test_dataloaders=test_dl)
    # trainer.save_checkpoint()