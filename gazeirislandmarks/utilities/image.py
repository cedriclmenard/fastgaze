import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import cv2

def show_image(im, savePath=None, show=True):
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(im, cmap='gray', vmin=0, vmax=255)
    ax.axis('off')
    if not savePath is None:
        plt.savefig(savePath)
    if show:
        plt.show()

def show_mesh_on_image(img, detections, highlights=None, showIndices=True, savePath=None):
    plt.imshow(img, zorder=1)
    x, y = detections[:, 0], detections[:, 1]
    plt.scatter(x, y, zorder=2, s=1.0, c='b')
    if showIndices:
        for i in range(detections.shape[0]):
            plt.text(detections[i,0], detections[i,1], str(i))
    if not highlights is None:
        x_h, y_h = detections[highlights, 0], detections[highlights, 1]
        plt.scatter(x_h, y_h, zorder=3, s=1.0, c='r')
    plt.axis("off")
    if not savePath is None:
        plt.savefig(savePath)
    plt.show()

def show_mesh(detections, highlights=None, showIndices=True, savePath=None):
    x, y = detections[:, 0], detections[:, 1]
    plt.scatter(x, y, zorder=2, s=1.0, c='b')
    if showIndices:
        for i in range(detections.shape[0]):
            plt.text(detections[i,0], detections[i,1], str(i))
    if not highlights is None:
        x_h, y_h = detections[highlights, 0], detections[highlights, 1]
        plt.scatter(x_h, y_h, zorder=3, s=1.0, c='r')
    plt.axis("off")
    if not savePath is None:
        plt.savefig(savePath)
    plt.show()

def show_dual_meshes_on_image(img, detections, highlights=None, showIndices=True, savePath=None):
    plt.imshow(img, zorder=1)
    x, y = detections[:, 0], detections[:, 1]
    plt.scatter(x, y, zorder=2, s=2.0, c='y')
    if showIndices:
        for i in range(detections.shape[0]):
            plt.text(detections[i,0], detections[i,1], str(i))
    if not highlights is None:
        x_h, y_h = highlights[:, 0], highlights[:, 1]
        plt.scatter(x_h, y_h, zorder=3, s=2.0, c='r')
    plt.axis("off")
    if not savePath is None:
        plt.savefig(savePath)
    plt.show()


def show_dual_images(img1, img2, show=True, block=True):
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(img1, cmap='gray', vmin=0, vmax=255)
    ax.axis('off')
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(img2, cmap='gray', vmin=0, vmax=255)
    ax.axis('off')

    if show:
        plt.show(block=block)

def get_width_height(image):
    if isinstance(image, np.ndarray):
        h = image.shape[0]
        w = image.shape[1]
    elif torch.is_tensor(image):
        h = image.shape[-2]
        w = image.shape[-1]
    elif isinstance(image, Image.Image):
        w, h = image.size
    return w, h

def square_center_crop_resize(image, square=128):
    if isinstance(image, np.ndarray):
        h = image.shape[0]
        w = image.shape[1]
        h_c = h//2
        w_c = w//2
        if h > w:
            left = 0
            right = w
            top = h_c - w/2
            bottom = top + w
            scale = w / square
        else:
            left = w_c - h/2
            right = left + h
            top = 0
            bottom = h
            scale = h / square
        img_small = cv2.resize(image[top:bottom, left:right, ...], (square,square))
        return img_small, scale, left, top
    elif torch.is_tensor(image):
        h = image.shape[-2]
        w = image.shape[-1]
        h_c = h//2
        w_c = w//2
        if h > w:
            left = 0
            right = w
            top = h_c - w//2
            bottom = top + w
            scale = w / square
        else:
            left = w_c - h//2
            right = left + h
            top = 0
            bottom = h
            scale = h / square
        img_small = F.interpolate(image[..., top:bottom, left:right].unsqueeze(0), (square,square), mode="bilinear").squeeze()
        return img_small, scale, -left, -top
    elif isinstance(image, Image.Image):
        w_o, h_o = image.size
        im = TF.center_crop(image, max(image.width, image.height))
        w_c, h_c = im.size
        im = TF.resize(im, (square,square))
        w, h = im.size
        x_offset = (w_c - w_o)/2.0
        y_offset = (h_c - h_o)/2.0
        scale = w_c/float(w)
        return im, scale, x_offset, y_offset # should do: (point*scale) + (x_offset, y_offset)