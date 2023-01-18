import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import os.path

from ..layers import BatchNorm, GhostBatchNorm, SeqNorm, GaussianDropout, BatchNorm2d, GhostBatchNorm2d, SeqNorm2d

def _norm_layer_factory(in_features, bn_splits, num_groups):
    if not bn_splits:
        return nn.BatchNorm2d(in_features)
    elif num_groups:
        return SeqNorm2d(in_features, num_groups, bn_splits)
    else:
        return GhostBatchNorm2d(in_features, bn_splits)

class IrisBlock(nn.Module):
    """This is the main building block for architecture"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, **kw):
        super(IrisBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        

        # My impl
        self.stride = stride
        self.channel_pad = out_channels - in_channels
        
        padding = (kernel_size - 1) // 2
        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)

        self.convAct = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=int(out_channels/2), kernel_size=stride, stride=stride, padding=0, bias=True),
            nn.PReLU(int(out_channels/2))
        )
        self.dwConvConv = nn.Sequential(
            nn.Conv2d(in_channels=int(out_channels/2), out_channels=int(out_channels/2), 
                      kernel_size=kernel_size, stride=1, padding=padding,  # Padding might be wrong here
                      groups=int(out_channels/2), bias=True),
            nn.Conv2d(in_channels=int(out_channels/2), out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.act = nn.PReLU(out_channels)

    def forward(self, x):
        h = self.convAct(x)
        if self.stride == 2:
            
            x = self.max_pool(x)
        
        h = self.dwConvConv(h)

        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)
        
        return self.act(h + x)

class IrisBlockBN(nn.Module):
    """This is the main building block for architecture"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, dropout: float = 0.0, bn_splits: int = 0, dropout_gaussian=False, num_groups=0, **kw):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        super(IrisBlockBN, self).__init__()
        

        # My impl
        self.stride = stride
        self.channel_pad = out_channels - in_channels

        # dp = dropout
        
        padding = (kernel_size - 1) // 2
        if stride == 2:
            # self.max_pool = nn.Sequential(nn.MaxPool2d(kernel_size=stride, stride=stride), nn.Dropout2d(p=dp))
            self.max_pool = nn.Sequential(nn.MaxPool2d(kernel_size=stride, stride=stride), nn.BatchNorm2d(in_channels))
        else:
            self.max_pool = nn.Identity()

        self.convAct = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=int(out_channels/2), kernel_size=stride, stride=stride, padding=0, bias=False), # False because it will be removed due to the batchnorm layer after
            # nn.BatchNorm2d(int(out_channels/2)) if bn_splits == 0 else GhostBatchNorm(int(out_channels/2), bn_splits),
            _norm_layer_factory(int(out_channels/2), bn_splits, num_groups),
            # nn.Dropout2d(p=dp),
            nn.PReLU(int(out_channels/2))
        )
        self.dwConvConv = nn.Sequential(
            nn.Conv2d(in_channels=int(out_channels/2), out_channels=int(out_channels/2), 
                      kernel_size=kernel_size, stride=1, padding=padding,  # Padding might be wrong here
                      groups=int(out_channels/2), bias=True), # False because it will be removed due to the batchnorm layer after
            # nn.BatchNorm2d(int(out_channels/2)) if bn_splits == 0 else GhostBatchNorm(int(out_channels/2), bn_splits),
            # nn.Dropout2d(p=dp),
            nn.Conv2d(in_channels=int(out_channels/2), out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False), # False because it will be removed due to the batchnorm layer after
            # nn.BatchNorm2d(out_channels) if bn_splits == 0 else GhostBatchNorm(out_channels, bn_splits),
            _norm_layer_factory(out_channels, bn_splits, num_groups),
            # nn.Dropout2d(p=dp),
        )
        if dropout == 0.0:
            dropout_layer = nn.Identity()
        elif not dropout_gaussian:
            dropout_layer = nn.Dropout2d(p=dropout)
        else:
            dropout_layer = GaussianDropout(p=dropout)
        
        self.act = nn.Sequential(
            dropout_layer,
            nn.PReLU(out_channels)
        )

    def forward(self, x):
        h = self.convAct(x)
        if self.stride == 2:
            
            x = self.max_pool(x)
        
        h = self.dwConvConv(h)

        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0.0)
        
        return self.act(h + x)

    def fold_bn(self):
        no_bn = IrisBlock(self.in_channels, self.out_channels, self.kernel_size, self.stride)
        if self.stride == 2:
            no_bn.max_pool = nn.MaxPool2d(kernel_size=self.stride, stride=self.stride)
        
        # no_bn.convAct = self.convAct
        # convert bn
        conv = self.convAct[0]
        bn = self.convAct[1]
        # w, b = torch.nn.utils.fuse_conv_bn_eval(conv.weight, conv.bias, bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)
        no_bn.convAct[0] = torch.nn.utils.fuse_conv_bn_eval(conv, bn)
        
        no_bn.convAct[1].weight = self.convAct[3].weight # use PReLU parameters
        # no_bn.convAct[0].weight = w # use converted weight and bias
        # no_bn.convAct[0].bias = b

        # no_bn.dwConvConv = self.dwConvConv
        no_bn.dwConvConv[0] = torch.nn.utils.fuse_conv_bn_eval(self.dwConvConv[0], self.dwConvConv[1])
        no_bn.dwConvConv[1] = torch.nn.utils.fuse_conv_bn_eval(self.dwConvConv[3], self.dwConvConv[4])
        no_bn.act = self.act
        return no_bn
    
    @staticmethod
    def from_irisblock(irisblock: IrisBlock, dropout: float = 0.0, bn_splits: int = 0, dropout_gaussian=False, num_groups=0, **kw):
        with torch.no_grad():
            irisblockbn = IrisBlockBN(irisblock.in_channels, irisblock.out_channels, irisblock.kernel_size, irisblock.stride, dropout, bn_splits, dropout_gaussian, num_groups)

            # Conv
            irisblockbn.convAct[0] = nn.Conv2d(in_channels=irisblock.in_channels, out_channels=int(irisblock.out_channels/2), kernel_size=irisblock.stride, stride=irisblock.stride, padding=0, bias=True)
            irisblockbn.convAct[0].weight.copy_(irisblock.convAct[0].weight)
            irisblockbn.convAct[0].bias.copy_(irisblock.convAct[0].bias)

            # PReLU
            irisblockbn.convAct[2].weight.copy_(irisblock.convAct[1].weight)

            # Conv
            irisblockbn.dwConvConv[0].weight.copy_(irisblock.dwConvConv[0].weight)
            irisblockbn.dwConvConv[0].bias.copy_(irisblock.dwConvConv[0].bias)
            irisblockbn.dwConvConv[1] = nn.Conv2d(in_channels=int(irisblock.out_channels/2), out_channels=irisblock.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
            irisblockbn.dwConvConv[1].weight.copy_(irisblock.dwConvConv[1].weight)
            irisblockbn.dwConvConv[1].bias.copy_(irisblock.dwConvConv[1].bias)

            # PReLU
            irisblockbn.act[1].weight.copy_(irisblock.act.weight)

            return irisblockbn


class IrisLandmarks(nn.Module):
    """The IrisLandmark face landmark model from MediaPipe.

    Because we won't be training this model, it doesn't need to have
    batchnorm layers. These have already been "folded" into the conv 
    weights by TFLite.

    The conversion to PyTorch is fairly straightforward, but there are 
    some small differences between TFLite and PyTorch in how they handle
    padding on conv layers with stride 2.

    This version works on batches, while the MediaPipe version can only
    handle a single image at a time.
    """
    default_weights_path = os.path.dirname(__file__) + "/irislandmarks.pth"

    def __init__(self):
        super(IrisLandmarks, self).__init__()

        # self.num_coords = 228
        # self.x_scale = 64.0
        # self.y_scale = 64.0
        self.min_score_thresh = 0.75

        self._define_layers()

    def _define_layers(self):
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

        
        
    def forward(self, x):
        # TFLite uses slightly different padding on the first conv layer
        # than PyTorch, so do it manually.
        x = F.pad(x, [0, 1, 0, 1], "constant", 0)
        b = x.shape[0]      # batch size, needed for reshaping later

        x = self.backbone(x)            # (b, 128, 8, 8)
        
        e = self.split_eye(x)           # (b, 213, 1, 1)    
        e = e.view(b, -1)               # (b, 213)
        
        i = self.split_iris(x)          # (b, 15, 1, 1)
        i = i.reshape(b, -1)            # (b, 15)
        
        return [e, i]

    def _device(self):
        """Which device (CPU or GPU) is being used by this model?"""
        return self.backbone[0].weight.device
    
    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
    
    def load_weights_from_checkpoint(self, path):
        data = torch.load(path)
        # self.load_state_dict(data["model_state_dict"])
        self.load_state_dict(data[0])
        self.eval()

    
    def _preprocess(self, x):
        """Converts the image pixels to the range [-1, 1]."""
        # return x.float() / 127.5 - 1.0
        return x.float() / 255.0 # NOTE: Seems to work better

    def predict_on_image(self, img):
        """Makes a prediction on a single image.

        Arguments:
            img: a NumPy array of shape (H, W, 3) or a PyTorch tensor of
                 shape (3, H, W). The image's height and width should be 
                 64 pixels.

        Returns:
            A tensor with face detections.
        """
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute((2, 0, 1))
        
        return self.predict_on_batch(img.unsqueeze(0))

    def predict_on_batch(self, x):
        """Makes a prediction on a batch of images.

        Arguments:
            x: a NumPy array of shape (b, H, W, 3) or a PyTorch tensor of
               shape (b, 3, H, W). The height and width should be 64 pixels.

        Returns:
            A list containing a tensor of face detections for each image in 
            the batch. If no faces are found for an image, returns a tensor
            of shape (0, 17).

        Each face detection is a PyTorch tensor consisting of 17 numbers:
            - ymin, xmin, ymax, xmax
            - x,y-coordinates for the 6 keypoints
            - confidence score
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).permute((0, 3, 1, 2))
            # x = torch.from_numpy(x)

        assert x.shape[1] == 3
        assert x.shape[2] == 64
        assert x.shape[3] == 64

        # 1. Preprocess the images into tensors:
        x = x.to(self._device())
        if x.dtype != torch.float32:
            x = self._preprocess(x)

        # 2. Run the neural network:
        with torch.no_grad():
            out = self.__call__(x)

        # 3. Postprocess the raw predictions:
        eye, iris = out

        return eye.view(-1, 71, 3), iris.view(-1, 5, 3)

    @staticmethod
    def get_default_weights_path():
        return str(Path(Path(__file__).parent, "irislandmarks.pth"))
