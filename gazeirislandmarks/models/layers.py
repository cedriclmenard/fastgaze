import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms.functional as TF

class Print(nn.Module):
    def __init__(self, description=None):
        self.description = description
        super(Print, self).__init__()

    def forward(self, x):
        if not self.description is None:
            print(self.description)
        print(x.shape)
        return x

class Dense(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        layers = [nn.Linear(in_features=cfg["in_features"], out_features=cfg["out_features"], bias=cfg.get("bias", True))]
        
        if cfg.get("bn", False):
            layers.append(nn.BatchNorm1d(cfg["out_features"]))
        layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


# From: https://www.kaggle.com/cepheidq/gaussian-dropout-for-pytorch?scriptVersionId=76880867
class GaussianDropout(nn.Module):

    def __init__(self, p: float):
        """
        Multiplicative Gaussian Noise dropout with N(1, p/(1-p))
        It is NOT (1-p)/p like in the paper, because here the
        noise actually increases with p. (It can create the same
        noise as the paper, but with reversed p values)

        Source:
        Dropout: A Simple Way to Prevent Neural Networks from Overfitting
        https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf

        :param p: float - determines the the standard deviation of the
        gaussian noise, where sigma = p/(1-p).
        """
        super().__init__()
        assert 0 <= p < 1
        self.t_mean = torch.ones((0,))
        self.shape = ()
        self.p = p
        self.t_std = self.compute_std()

    def compute_std(self):
        return self.p / (1 - self.p)

    def forward(self, t_hidden):
        if self.training and self.p > 0.:
            if self.t_mean.shape != t_hidden.shape:
                self.t_mean = torch.ones_like(input=t_hidden
                                              , dtype=t_hidden.dtype
                                              , device=t_hidden.device)
            elif self.t_mean.device != t_hidden.device:
                self.t_mean = self.t_mean.to(device=t_hidden.device, dtype=t_hidden.dtype)

            t_gaussian_noise = torch.normal(self.t_mean, self.t_std)
            t_hidden = t_hidden.mul(t_gaussian_noise)
        return t_hidden

# impl from: https://github.com/nixx14/Ghost-BatchNormalisation-/blob/master/GhostBatchNorm.ipynb
class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight=True, bias=True):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias

class GhostBatchNorm2d(BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(num_features * self.num_splits))
        self.register_buffer('running_var', torch.ones(num_features * self.num_splits))

    def train(self, mode=True):
        if (self.training is True) and (mode is False):  # lazily collate stats when we are going to use them
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
        return super().train(mode)

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            return F.batch_norm(
                input.view(-1, C * self.num_splits, H, W), self.running_mean, self.running_var,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
        else:
            return F.batch_norm(
                input, self.running_mean[:self.num_features], self.running_var[:self.num_features],
                self.weight, self.bias, False, self.momentum, self.eps)

class SeqNorm2d(nn.Module):
    def __init__(self, num_features, num_groups, num_bn_splits, **kw):
        super().__init__()
        self.m = nn.Sequential(
            torch.nn.GroupNorm(num_groups, num_features),
            GhostBatchNorm2d(num_features, num_bn_splits)
        )
    
    def train(self, mode=True):
        self.m.train(mode=mode)
        return self
    
    def forward(self, input):
        return self.m(input)
        
class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight=True, bias=True):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias

class GhostBatchNorm(BatchNorm):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(num_features * self.num_splits))
        self.register_buffer('running_var', torch.ones(num_features * self.num_splits))

    def train(self, mode=True):
        if (self.training is True) and (mode is False):  # lazily collate stats when we are going to use them
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
        return super().train(mode)

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            return F.batch_norm(
                input.view(-1, C * self.num_splits, H, W), self.running_mean, self.running_var,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
        else:
            return F.batch_norm(
                input, self.running_mean[:self.num_features], self.running_var[:self.num_features],
                self.weight, self.bias, False, self.momentum, self.eps)

class SeqNorm(nn.Module):
    def __init__(self, num_features, num_groups, num_bn_splits, **kw):
        super().__init__()
        self.m = nn.Sequential(
            torch.nn.GroupNorm(num_groups, num_features),
            GhostBatchNorm(num_features, num_bn_splits)
        )
    
    def train(self, mode=True):
        self.m.train(mode=mode)
        return self
    
    def forward(self, input):
        return self.m(input)