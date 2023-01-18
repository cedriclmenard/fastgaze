import torch
import torchvision
import torchvision.transforms.functional as F
import numpy as np

from torch import Tensor
from functools import partial
from PIL import Image

if hasattr(F, "get_image_size"):
    get_image_size = F.get_image_size
else:
    get_image_size = F._get_image_size



class RandomAffine(torchvision.transforms.RandomAffine):
    def get_constant_transform(self):
        def constant_transform(img):
            fill = self.fill
            if isinstance(img, Tensor):
                if isinstance(fill, (int, float)):
                    fill = [float(fill)] * F._get_image_num_channels(img)
                else:
                    fill = [float(f) for f in fill]

            img_size = get_image_size(img)

            ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)
            return F.affine(img, *ret, interpolation=self.interpolation, fill=fill)
        
        return constant_transform
        # return lambda img: F.affine(img, *ret, interpolation=self.interpolation, fill=fill)
        # return F.affine(img, *ret, interpolation=self.interpolation, fill=fill)

class ColorJitter(torchvision.transforms.ColorJitter):
    def get_constant_transform(self):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        def l_func(img):
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    img = F.adjust_brightness(img, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    img = F.adjust_contrast(img, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    img = F.adjust_saturation(img, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    img = F.adjust_hue(img, hue_factor)

            return img
        return l_func

class GaussianBlur(torchvision.transforms.GaussianBlur):
    def get_constant_transform(self):
        def constant_transform(img):
            # not constant but doesn't matter much
            sigma = self.get_params(self.sigma[0], self.sigma[1])
            return F.gaussian_blur(img, self.kernel_size, [sigma, sigma])
        
        return constant_transform

class GammaJitter(torch.nn.Module):
    def __init__(self, stdev_gamma):
        super().__init__()
        self.stdev_gamma = stdev_gamma
    
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        return F.adjust_gamma(img, torch.abs(torch.empty(1).normal_(1, self.stdev_gamma)).item())

    def get_constant_transform(self):
        gamma = torch.abs(torch.empty(1).normal_(1, self.stdev_gamma)).item()
        def constant_transform(img):
            return F.adjust_gamma(img, gamma)
        return constant_transform


def get_padding(image):    
    w, h = image.size
    max_wh = np.max([w, h])
    h_padding = (max_wh - w) / 2
    v_padding = (max_wh - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding

class PadToSquare(torch.nn.Module):
    def __init__(self, padding_mode='constant', fill=0):
        assert isinstance(fill, (int, float, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return F.pad(img, get_padding(img), self.fill, self.padding_mode)


def gaussian_noise(img, sigma):
    """
    Args:
        img (PIL Image or Tensor): Input image.

    Returns:
        PIL Image or Tensor: Color jittered image.
    """
    is_pil = False
    if isinstance(img, Image.Image):
        is_pil = True
        img = F.pil_to_tensor(img)

    assert isinstance(img, torch.Tensor)

    dtype = img.dtype
    if not img.is_floating_point():
        img = img.to(torch.float32)
    
    out = torch.clamp(img + sigma * torch.randn_like(img), min=0.0, max=255.0)
    
    
    if out.dtype != dtype:
        out = out.to(dtype)
    
    if is_pil:
        out = F.to_pil_image(out)
        
    return out

class GaussianNoise(torch.nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma
    
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        return gaussian_noise(img, self.sigma)

    def get_constant_transform(self):
        def constant_transform(img):
            return gaussian_noise(img, self.sigma)
        return constant_transform