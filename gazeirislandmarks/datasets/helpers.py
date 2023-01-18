import torch
import torchvision
from PIL import Image
import numpy as np
from random import shuffle
from itertools import chain
import h5py
# import hdf5plugin
import io
import pickle
import os.path
import torchvision.transforms.functional as TF
import transformations as tr
import cv2
import json
from pathlib import Path
import types
from torch.utils.data._utils.collate import default_collate
from bisect import bisect_left
from torch import default_generator

from enum import Enum

from typing import (
    # Callable,
    # Dict,
    # Generic,
    # Iterable,
    # Iterator,
    # List,
    Optional,
    # Sequence,
    # Tuple,
    # TypeVar,
)

from ..utilities.torchutils import multiprocess_run_on_dataset
from ..utilities.general import progress_bar
from ..utilities.geometry import vector_to_yaw_pitch

default_normalized_focal = 650
default_normalized_distance = 600
class NormalizationType(Enum):
    ORIGINAL=1
    NEW=2


class PersonDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
    
    def get_number_of_persons(self):
        raise NotImplementedError

    def get_person_dataset(self, p):
        raise NotImplementedError

def random_split_person_dataset(dataset, split: float = 0.8, np_test: int = 1, return_indices=False, save_indices_path=None, load_if_available=False):
    if load_if_available and save_indices_path:
        if os.path.isfile(save_indices_path):
            return split_person_dataset(dataset, pickle.load(open(save_indices_path, "rb")))
    npersons = dataset.get_number_of_persons()
    indices = list(range(npersons))
    shuffle(indices)
    np_train = int(npersons*split)
    np_val = npersons - np_train - np_test
    if np_val < 1:
        np_val = 1
        np_train = npersons - np_val - np_test
    
    train_datasets = []
    val_datasets = []
    test_datasets = []

    train_ps = []
    val_ps = []
    test_ps = []
    for i in range(np_train):
        p = indices[i]
        train_ps.append(p)
        train_datasets.append(dataset.get_person_dataset(p))
    for i in range(np_train, np_train + np_val):
        p = indices[i]
        val_ps.append(p)
        val_datasets.append(dataset.get_person_dataset(p))
    for i in range(np_train + np_val, np_train + np_val + np_test):
        p = indices[i]
        test_ps.append(p)
        test_datasets.append(dataset.get_person_dataset(p))
    
    train_ds = torch.utils.data.ConcatDataset(train_datasets)
    val_ds = torch.utils.data.ConcatDataset(val_datasets)
    test_ds = torch.utils.data.ConcatDataset(test_datasets)
    person_indices = {
        "train": train_ps,
        "val": val_ps,
        "test": test_ps
    }
    print(person_indices)
    if save_indices_path:
        pickle.dump(person_indices, open(save_indices_path, "wb"))
    if return_indices:
        return train_ds, val_ds, test_ds, person_indices
    else:
        return train_ds, val_ds, test_ds

def leave_one_out_split(dataset, val_index=None, test_index=None):
    npersons = dataset.get_number_of_persons()
    
    train_datasets = []
    val_datasets = []
    test_datasets = []

    for p in range(npersons):
        if p == val_index:
            val_datasets.append(dataset.get_person_dataset(p))
        if p == test_index:
            test_datasets.append(dataset.get_person_dataset(p))
        if p != val_index and p != test_index:
            train_datasets.append(dataset.get_person_dataset(p))

    
    train_ds = torch.utils.data.ConcatDataset(train_datasets)
    val_ds = torch.utils.data.ConcatDataset(val_datasets)
    test_ds = torch.utils.data.ConcatDataset(test_datasets)
    return train_ds, val_ds, test_ds

def split_person_dataset(dataset, indices):
    train_datasets = []
    val_datasets = []
    test_datasets = []

    for p in indices["train"]:
        train_datasets.append(dataset.get_person_dataset(p))
    for p in indices["val"]:
        val_datasets.append(dataset.get_person_dataset(p))
    for p in indices["test"]:
        test_datasets.append(dataset.get_person_dataset(p))
    
    train_ds = torch.utils.data.ConcatDataset(train_datasets)
    val_ds = torch.utils.data.ConcatDataset(val_datasets)
    test_ds = torch.utils.data.ConcatDataset(test_datasets)

    return train_ds, val_ds, test_ds

class GaussianNoise():
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std
    
    def forward(self, array):
        if isinstance(array, torch.Tensor):
            tensor = array
        else:
            tensor = torchvision.transforms.functional.to_tensor(array)
        return torchvision.transforms.functional.to_pil_image((tensor + torch.randn(tensor.size()) * self.std + self.mean).type(torch.uint8))
    
    def __call__(self, tensor):
        return self.forward(tensor)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def apply_transform_to_sample(sample, transform):
    for k in ["image_l", "image_r", "image"]:
        if k in sample:
            if isinstance(sample[k], np.ndarray):
                sample[k] = np.array(transform(Image.fromarray(sample[k])))
            else:
                sample[k] = transform(sample[k])
    return sample

def apply_transform_to_sample_constant(sample, transform):
    transform_constant = transform.get_constant_transform()
    for k in ["image_l", "image_r", "image"]:
        if k in sample:
            if isinstance(sample[k], np.ndarray):
                sample[k] = np.array(transform_constant(Image.fromarray(sample[k])))
            else:
                sample[k] = transform_constant(sample[k])
    return sample

class TransformDataset(PersonDataset):
    def __init__(self, dataset: torch.utils.data.Dataset, transform, keep_original=True, apply_constantly_on_sample=False):
        self.transform = transform
        self.dataset = dataset
        self.keep_original = keep_original
        self.apply_constantly_on_sample = apply_constantly_on_sample
    
    def __len__(self):
        if self.keep_original:
            return 2*len(self.dataset)
        else:
            return len(self.dataset)

    def __getitem__(self, idx):
        if self.apply_constantly_on_sample:
            apply_transform = apply_transform_to_sample_constant
        else:
            apply_transform = apply_transform_to_sample
        
        if self.keep_original:
            if idx % 2 == 0:
                return self.dataset[idx//2]
            else:
                return apply_transform(self.dataset[idx//2], self.transform)
        else:
            return apply_transform(self.dataset[idx], self.transform)
    
    def get_number_of_persons(self):
        return self.dataset.get_number_of_persons()

    def get_person_dataset(self, p):
        return TransformDataset(self.dataset.get_person_dataset(p), self.transform, self.keep_original, self.apply_constantly_on_sample)

class TransformPoseRandomDataset(PersonDataset):
    def __init__(self, dataset: torch.utils.data.Dataset, stdeviation, fields):
        self.dataset = dataset
        self.stdeviation = stdeviation
        self.fields = fields
    
    def __len__(self):
            return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        for k in self.fields:
            if k in sample:
                if isinstance(sample[k], np.ndarray):
                    sample[k] += np.random.normal(scale=self.stdeviation, size=sample[k].shape)
                else:
                    sample[k] += torch.randn(sample[k].shape) * self.stdeviation
        return sample
    
    def get_number_of_persons(self):
        return self.dataset.get_number_of_persons()

    def get_person_dataset(self, p):
        return TransformPoseRandomDataset(self.dataset.get_person_dataset(p), self.stdeviation)

class TransformTorchvisionDataset(PersonDataset):
    def __init__(self, dataset: torch.utils.data.Dataset, transform):
        self.transform = transform
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        for key in ["image_l", "image_r", "image"]:
            if sample.get(key) is not None:
                if isinstance(sample[key], np.ndarray):
                    sample[key] = np.array(self.transform(TF.to_pil_image(sample[key])))
                else:
                    sample[key] = self.transform(sample[key])
        return sample
    
    def get_number_of_persons(self):
        return self.dataset.get_number_of_persons()

    def get_person_dataset(self, p):
        return TransformTorchvisionDataset(self.dataset.get_person_dataset(p), self.transform)

# class AugmentedDataset(torch.utils.data.Dataset):

#     def __init__(self, dataset: torch.utils.data.Dataset, grayscale: bool, noisy: int = 0, noise_mean: float = 0.0, noise_std: float = 1.0):
#         self.dataset = dataset
#         self.n = len(dataset)
#         self.grayscale = grayscale
#         self.noisy = noisy

#         # tsfm = []
#         self.t_grayscale = torchvision.transforms.Grayscale(num_output_channels=3)
#         self.t_noise = GaussianNoise(noise_mean, noise_std)

#         if grayscale:
#             self.n *= 2
#             # tsfm.append(torchvision.transforms.Grayscale(num_output_channels=3))
        
#         if noisy != 0:
#             self.n += noisy*len(self.dataset)
    
#     def __len__(self):
#         return self.n

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         ds_scale = self.n // len(self.dataset)
#         i = idx // ds_scale
#         i_aug = idx - i*ds_scale
#         if i_aug == 0:
#             # normal image
#             return self.dataset[i]
#         elif self.grayscale and i_aug % ds_scale == 1:
#             # grayscale image
#             sample = self.dataset[i]
#             if "image" in sample:
#                 sample["image"] = np.array(self.t_grayscale(Image.fromarray(sample["image"])), copy=False).copy()
#             if "image_l" in sample:
#                 sample["image_l"] = np.array(self.t_grayscale(Image.fromarray(sample["image_l"])), copy=False).copy()
#             if "image_r" in sample:
#                 sample["image_r"] = np.array(self.t_grayscale(Image.fromarray(sample["image_r"])), copy=False).copy()
#             return sample
#         else:
#             sample = self.dataset[i]
#             if "image" in sample:
#                 sample["image"] = np.array(self.t_noise(Image.fromarray(sample["image"])), copy=False).copy()
#             if "image_l" in sample:
#                 sample["image_l"] = np.array(self.t_noise(Image.fromarray(sample["image_l"])), copy=False).copy()
#             if "image_r" in sample:
#                 sample["image_r"] = np.array(self.t_noise(Image.fromarray(sample["image_r"])), copy=False).copy()
#             return sample


class InMemoryDataset(PersonDataset):
    def __init__(self, dataset):
        self.samples = [s for s in dataset]
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.samples[idx]

    def get_number_of_persons(self):
        return self.dataset.get_number_of_persons()

    def get_person_dataset(self, p):
        return InMemoryDataset(self.dataset.get_person_dataset(p))


def load_worker_InMemoryParallelDataset(dataset):
            return [s for s in dataset]
class InMemoryParallelDataset(InMemoryDataset):
    def __init__(self, dataset, processes=4):
        list_of_list_of_samples = multiprocess_run_on_dataset(processes, load_worker_InMemoryParallelDataset, dataset)
        self.samples = list(chain(*list_of_list_of_samples))

class PreprocessedDataset(PersonDataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.dataset[idx]
        sample["image_l"] = torchvision.transforms.functional.hflip(TF.to_tensor(sample["image_l"]))
        sample["image_r"] = TF.to_tensor(sample["image_r"])

        sample["preprocessed"] = True
        return sample
    
    def get_number_of_persons(self):
        return self.dataset.get_number_of_persons()

    def get_person_dataset(self, p):
        return PreprocessedDataset(self.dataset.get_person_dataset(p))

def sample_to_hdf5(hfile, sample, compression=False, image_format=None):
    if type(sample) is list:
        for i, v in enumerate(sample):
            if type(v) is dict:
                sample_to_hdf5(hfile.create_group(str(i)), v)
            else:
                if compression:
                    try:
                        hfile.create_dataset(str(i), data=v, compression="gzip", compression_opts=5)
                    except TypeError:
                        hfile.create_dataset(str(i), data=v)
                else:
                    hfile.create_dataset(str(i), data=v)
    elif type(sample) is dict:
        for k, v in sample.items():
            if type(v) is dict:
                sample_to_hdf5(hfile.create_group(k), v)
            else:
                if type(v) is np.ndarray and v.dtype == np.uint8 and image_format:
                    # this is an image, compress accordingly
                    # maybe use JPEG-LS filter? See: https://support.hdfgroup.org/services/filters.html
                    with io.BytesIO() as output:
                        if image_format == "JPEG":
                            Image.fromarray(v).save(output, format=image_format, quality=95)
                        else:
                            Image.fromarray(v).save(output, format=image_format)
                        dset = hfile.create_dataset(k, data=np.fromstring(output.getvalue(), dtype='uint8'))
                        # hfile.create_dataset(k, data=v, compression=32018) #JPEG-LS
                elif type(v) is Image:
                    with io.BytesIO() as output:
                        if image_format == "JPEG":
                            v.save(output, format=image_format, quality=95)
                        else:
                            v.save(output, format=image_format)
                        dset = hfile.create_dataset(k, data=np.fromstring(output.getvalue(), dtype='uint8'))
                elif compression:
                    try:
                        hfile.create_dataset(k, data=v, compression="gzip", compression_opts=5)
                    except TypeError:
                        hfile.create_dataset(k, data=v)
                else:
                    hfile.create_dataset(k, data=v)
    else:
        raise ValueError

def hdf5_to_sample(hfile, image_format=None):
    sample = {}
    for k, v in hfile.items():
        if isinstance(v, h5py.Dataset):
            if image_format and v.dtype == np.uint8:
                sample[k] = np.array(Image.open(io.BytesIO(v[()])))
            elif image_format and v.dtype == "O" and v.size == 100:
                sample[k] = np.array(Image.open(io.BytesIO(v[0])))
            else:
                sample[k] = v[()] 
        else:
            sample[k] = hdf5_to_sample(v)
    return sample

        

def convert_to_hdf5(dataset, output_path, compression=False):
    hf = h5py.File(output_path, "w")

    i = 0
    for s in progress_bar(dataset, prefix="Progress:", suffix="Complete", length=50):
        g = hf.create_group(str(i))
        sample_to_hdf5(g, s, compression)
        i+=1


def __collate(batch):
    return batch[0]

def convert_to_personhdf5(dataset, output_path, compression=False, num_workers=1, augmented=False, image_format=None):
    with h5py.File(output_path, "w") as hf:

        for p in range(dataset.get_number_of_persons()):
            if num_workers > 1:
                # def collate(batch):
                #     return batch[0]
                if augmented:
                    # tmp = AugmentedDataset(dataset.get_person_dataset(p), True, 2, 0.0, 3.0)
                    raise NotImplementedError
                else:
                    tmp = dataset.get_person_dataset(p)
                d = torch.utils.data.DataLoader(tmp, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=__collate)
            else:
                if augmented:
                    # d = AugmentedDataset(dataset.get_person_dataset(p), True, 2, 0.0, 3.0)
                    raise NotImplementedError
                else:
                    d = dataset.get_person_dataset(p)
            gp = hf.create_group(str(p))
            i = 0
            for s in progress_bar(d, prefix="Progress:", suffix="Complete", length=50):
                if s is None:
                    continue
                g = gp.create_group(str(i))
                sample_to_hdf5(g, s, compression, image_format)
                i+=1

def convert_to_personhdf5_try_skip(dataset, output_path, compression=False, augmented=False, image_format=None):
    with h5py.File(output_path, "w") as hf:

        for p in range(dataset.get_number_of_persons()):
            if augmented:
                # d = AugmentedDataset(dataset.get_person_dataset(p), True, 2, 0.0, 3.0)
                raise NotImplementedError
            else:
                d = dataset.get_person_dataset(p)
            gp = hf.create_group(str(p))
            i = 0
            for idx in progress_bar(range(len(d)), prefix="Progress:", suffix="Complete", length=50):
                try:
                    s = d[idx]
                    if s is None:
                        continue
                    g = gp.create_group(str(i))
                    sample_to_hdf5(g, s, compression, image_format)
                    i+=1
                except:
                    print(f"Error getting sample at index {idx}")
                    continue
                


    

class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, path=None, hdf=None, in_memory=False):
        self.in_memory = in_memory
        assert path or hdf
        if path:
            self.path = path
            self.hf = h5py.File(path, "r")
        else:
            self.hf = hdf

        if in_memory:
            self.hf = self.hf[:]
        self.n = len(self.hf.items())
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if idx >= self.n:
            raise IndexError()
        return hdf5_to_sample(self.hf.get(str(idx)))
    
    def reopen_file(self):
        if not self.in_memory:
            self.hf = h5py.File(self.path, "r")

class HDF5PersonDataset(PersonDataset):
    def __init__(self, path, image_format=None, in_memory=False):
        self.hf = None
        self.in_memory = in_memory
        self.path = path
        # self.hf = h5py.File(path, "r")
        with h5py.File(path, "r") as hf:
            self.np = len(hf.items())
            self.n = 0
            self.ranges = []
            for p in range(self.np):
                l = len(hf.get(str(p)).items())
                self.ranges.append(range(self.n, self.n + l))
                self.n += l

            if in_memory:
                self.in_memory_file = io.BytesIO()
                with h5py.File(self.in_memory_file, "w") as tmp_hf:
                    for d in hf.keys():
                        hf.copy(d, tmp_hf)

                hf = h5py.File(self.in_memory_file, "r")
            
            self.image_format = image_format
    
    def __len__(self):
        return self.n
    
    def get_number_of_persons(self):
        return self.np

    def get_person_dataset(self, p):
        return torch.utils.data.Subset(self, self.ranges[p])
    
    def __getitem__(self, idx):
        if self.hf is None:
            self.hf = h5py.File(self.path, "r")
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if idx >= self.n:
            raise IndexError()
        def get_person(i):
            for p, r in enumerate(self.ranges):
                if i in r:
                    return p
        p = get_person(idx)
        return hdf5_to_sample(self.hf.get(str(p)+"/"+str(idx - self.ranges[p].start)), image_format=self.image_format)
    
    def reopen_file(self):
        if not self.in_memory:
            self.hf = h5py.File(self.path, "r")
        else:
            self.hf = h5py.File(self.in_memory_file, "r")

def append_to_hdf5(path_to, dataset_from):
    hf_to = h5py.File(path_to, 'a')
    i = len(hf_to.items())
    
    for s in dataset_from:
        g = hf_to.create_group(str(i))
        sample_to_hdf5(g, s)
        i+=1

def dataset_persons_to_list_of_ranges(dataset):
    npersons = dataset.get_number_of_persons()
    lengths = []
    for p in range(npersons):
        lengths.append(len(dataset.get_person_dataset(p)))
    
    ranges = []
    for l in lengths:
        if ranges:
            ranges.append(range(ranges[-1].stop, ranges[-1].stop + l))
        else:
            ranges.append(range(l))
    return ranges

class PersonSplitDataset(PersonDataset):
    def __init__(self, dataset, list_of_ranges):
        self.dataset = dataset
        self.ranges = list_of_ranges
    
    def __len__(self):
        return len(self.dataset)
    
    def get_number_of_persons(self):
        return len(self.ranges)

    def get_person_dataset(self, p):
        return torch.utils.data.Subset(self.dataset, self.ranges[p])
    
    def __getitem__(self, idx):
        return self.ds[idx]


def normalize_img(img, gaze_origin, head_rotation, gaze_target, roi_size, cam_matrix, focal_new=default_normalized_focal, distance_new=default_normalized_distance):
    # gaze_origin = target_3d
    # gc = gaze target
    # NOTE: this is the new method (doesn't scale the gaze angles)

    distance = np.linalg.norm(gaze_origin)
    z_scale = distance_new / distance
    cam_new = np.array([[focal_new, 0, roi_size[0] / 2],
                        [0.0, focal_new, roi_size[1] / 2],
                        [0, 0, 1.0]])
    scale_mat = np.array([[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, z_scale]])
    h_rx = head_rotation[:, 0].squeeze()
    forward = (gaze_origin / distance).squeeze()
    down = np.cross(forward, h_rx)
    down = down / np.linalg.norm(down)
    right = np.cross(down, forward)
    right = right / np.linalg.norm(right)

    rot_mat = np.c_[right, down, forward].T
    warp_mat = (cam_new @ scale_mat) @ (rot_mat @ np.linalg.inv(cam_matrix))
    img_warped = cv2.warpPerspective(img, warp_mat, roi_size)

    gaze_direction = gaze_target.squeeze() - gaze_origin.squeeze()
    gaze_direction /= np.linalg.norm(gaze_direction)
    n_gaze_direction = rot_mat.dot(gaze_direction)
    n_head_rotation = rot_mat @ head_rotation # head pose in the new normalized camera reference frame

    return img_warped, cv2.Rodrigues(n_head_rotation)[0].squeeze(), n_gaze_direction

def normalize_gaze(gaze_origin, head_rotation, gaze_target, distance_new=default_normalized_distance):

    distance = np.linalg.norm(gaze_origin)
    z_scale = distance_new / distance
    scale_mat = np.array([[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, z_scale]])
    h_rx = head_rotation[:, 0].squeeze()
    forward = (gaze_origin / distance).squeeze()
    down = np.cross(forward, h_rx)
    down = down / np.linalg.norm(down)
    right = np.cross(down, forward)
    right = right / np.linalg.norm(right)

    rot_mat = np.array([right.T, down.T, forward.T])

    # rotation normalization
    cnv_mat = scale_mat @ rot_mat
    h_rnew = cnv_mat @ head_rotation
    hrnew = cv2.Rodrigues(h_rnew)[0].reshape((3,))
    htnew = cnv_mat @ gaze_origin.squeeze()

    # gaze vector normalization
    gaze_target_new = cnv_mat @ gaze_target
    gvnew = gaze_target_new.squeeze() - htnew
    gvnew = gvnew / np.linalg.norm(gvnew)

    return hrnew, gvnew

def denormalize_gaze(n_gv, gaze_origin, head_rotation, distance_new=default_normalized_distance):

    distance = np.linalg.norm(gaze_origin)
    z_scale = distance_new / distance
    scale_mat = np.array([[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, z_scale]])
    h_rx = head_rotation[:, 0].squeeze()
    forward = (gaze_origin / distance).squeeze()
    down = np.cross(forward, h_rx)
    down = down / np.linalg.norm(down)
    right = np.cross(down, forward)
    right = right / np.linalg.norm(right)

    rot_mat = np.array([right.T, down.T, forward.T])

    # rotation normalization
    cnv_mat = scale_mat @ rot_mat
    gv = np.linalg.inv(cnv_mat).dot(n_gv)
    gv /= np.linalg.norm(gv)
    return gv

def normalize_gaze_new(gaze_origin, head_rotation, gaze_target):
    distance = np.linalg.norm(gaze_origin)
    h_rx = head_rotation[:, 0].squeeze()
    forward = (gaze_origin / distance).squeeze()
    down = np.cross(forward, h_rx)
    down = down / np.linalg.norm(down)
    right = np.cross(down, forward)
    right = right / np.linalg.norm(right)

    rot_mat = np.c_[right, down, forward].T

    gaze_direction = gaze_target.squeeze() - gaze_origin.squeeze()
    gaze_direction /= np.linalg.norm(gaze_direction)
    n_gaze_direction = rot_mat.dot(gaze_direction)
    n_head_rotation = rot_mat @ head_rotation # head pose in the new normalized camera reference frame

    return cv2.Rodrigues(n_head_rotation)[0].squeeze(), n_gaze_direction

def denormalize_gaze_new(n_gv, gaze_origin, head_rotation):
    distance = np.linalg.norm(gaze_origin)
    h_rx = head_rotation[:, 0].squeeze()
    forward = (gaze_origin / distance).squeeze()
    down = np.cross(forward, h_rx)
    down = down / np.linalg.norm(down)
    right = np.cross(down, forward)
    right = right / np.linalg.norm(right)

    rot_mat = np.c_[right, down, forward].T

    gv = rot_mat.T.dot(n_gv)
    gv /= np.linalg.norm(gv)

    return gv

def denormalize_points(points, head_rot_mat, eye_position_3d, roi_size, cam_matrix, focal_new=default_normalized_focal, distance_new=default_normalized_distance):
    distance = np.linalg.norm(eye_position_3d)
    z_scale = distance_new / distance
    cam_new = np.array([[focal_new, 0, roi_size[0] / 2],
                        [0.0, focal_new, roi_size[1] / 2],
                        [0, 0, 1.0]])
    scale_mat = np.array([[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, z_scale]])
    h_rx = head_rot_mat[:, 0]
    forward = (eye_position_3d / distance)
    down = np.cross(forward, h_rx)
    down = down / np.linalg.norm(down)
    right = np.cross(down, forward)
    right = right / np.linalg.norm(right)
    rot_mat = np.array([right.T, down.T, forward.T])

    warp_mat = (cam_new @ scale_mat) @ (rot_mat @ np.linalg.inv(cam_matrix))
    return cv2.perspectiveTransform(np.expand_dims(points, 1), np.linalg.inv(warp_mat)).squeeze()



def split_faze_gazecapture(dataset, verbose=False):
    data = json.load(open(Path(Path(__file__).parent.parent.parent, "data", "faze_gazecapture_split.json")))

    ids = dataset.get_persons_ids()
    train_datasets = []
    val_datasets = []
    test_datasets = []
    for pid in data["train"]:
        try:
            p = ids.index(pid)
        except ValueError:
            if verbose:
                print("Could not find subject with id: %s" % pid)
            continue
        train_datasets.append(dataset.get_person_dataset(p))
    
    for pid in data["val"]:
        try:
            p = ids.index(pid)
        except ValueError:
            if verbose:
                print("Could not find subject with id: %s" % pid)
            continue
        val_datasets.append(dataset.get_person_dataset(p))

    for pid in data["test"]:
        try:
            p = ids.index(pid)
        except ValueError:
            if verbose:
                print("Could not find subject with id: %s" % pid)
            continue
        test_datasets.append(dataset.get_person_dataset(p))

    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)
    if test_datasets:
        test_dataset = torch.utils.data.ConcatDataset(test_datasets)
    else:
        test_dataset = None
    
    return train_dataset, val_dataset, test_dataset

def split_faze_gazecapture_hdf5(dataset, verbose=False):
    def get_persons_ids(self):
        pids = []
        for p in range(self.get_number_of_persons()):
            sample = hdf5_to_sample(self.hf.get(str(p)+"/0"), image_format=self.image_format)
            pids.append([s for s in sample["image_path"].decode("utf-8").split("/") if s][-3])
        return pids

    _ = dataset[0] # dummy read to get hf file
    dataset.get_persons_ids = types.MethodType(get_persons_ids, dataset)
    ret = split_faze_gazecapture(dataset, verbose)
    dataset.hf = None # make sure to None back to prevent pickle errors when passing the dataset to the dataloaders
    dataset.get_persons_ids = None
    return ret

def split_faze_mpii(dataset, split_ratio=0.8):
    train_datasets = []
    val_datasets = []
    test_datasets = []
    for p in range(dataset.get_number_of_persons()):
        dataset_p = dataset.get_person_dataset(p)
        indices = list(range(len(dataset.get_person_dataset(p))))
        n_samples = len(dataset.get_person_dataset(p))
        n_split = int(0.8 * (n_samples-500))
        train_datasets.append(torch.utils.data.Subset(dataset_p, indices[:n_split]))
        val_datasets.append(torch.utils.data.Subset(dataset_p, indices[n_split:-500]))
        test_datasets.append(torch.utils.data.Subset(dataset_p, indices[-500:]))

    train_datasets = torch.utils.data.ConcatDataset(train_datasets)
    val_datasets = torch.utils.data.ConcatDataset(val_datasets)
    test_datasets = torch.utils.data.ConcatDataset(test_datasets)
    return train_datasets, val_datasets, test_datasets

class PersonTaskDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: PersonDataset, num_support_samples=5, num_query_samples=None):
        self.dataset = dataset
        self.num_support_samples = num_support_samples
        self.num_query_samples = num_query_samples
        

    def __len__(self):
        return self.dataset.get_number_of_persons()
    
    def __getitem__(self, idx):
        pds = self.dataset.get_person_dataset(idx)
        indices = torch.randperm(len(pds))
        if self.num_query_samples and self.num_query_samples < (len(pds) - self.num_support_samples):
            num_query_samples = self.num_query_samples
        else:
            num_query_samples = len(pds) - self.num_support_samples

        batch = {}
        
        # Reformat
        batch["support"] = default_collate([pds[indices[i]] for i in range(self.num_support_samples)])
        batch["query"] = default_collate([pds[indices[i]] for i in range(num_query_samples)])
        batch["support"]["left"] = batch["support"].pop("image_l")
        batch["support"]["right"] = batch["support"].pop("image_r")
        batch["query"]["left"] = batch["query"].pop("image_l")
        batch["query"]["right"] = batch["query"].pop("image_r")
        
        return batch

class PersonConcatDataset(PersonDataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.n_per_p = [len(d) for d in datasets]
        self.cummulative_n = np.cumsum(self.n_per_p)
        self.length = sum(self.n_per_p)
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        def get_val(lst, v):
            if lst[-1] < v:
                return None
            idx = bisect_left(lst, v, hi=len(lst) - 1)
            if lst[idx] == v:
                return idx + 1
            return idx

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        p = get_val(self.cummulative_n, idx)
        idx_in_p = idx - (self.cummulative_n[p-1] if p != 0 else 0)
        return self.datasets[p][idx_in_p]
    
    def get_number_of_persons(self):
        return len(self.datasets)

    def get_person_dataset(self, p):
        return self.datasets[p]


class MultiEpochDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, splits, generator: Optional[torch.Generator] = default_generator):
        self.generator = generator
        self.dataset = dataset
        self.splits = splits
        self.length = len(self.dataset)
        self.generate_random_splits()
        self.current_subset = 0
        
    
    def __len__(self):
        return len(self.split_indices[self.current_subset])

    def __getitem__(self, idx):
        return self.dataset[self.split_indices[self.current_subset][idx]]

    def end_subset(self):
        self.current_subset += 1
        if self.current_subset >= self.splits:
            self.current_subset = 0
            self.generate_random_splits()


    def generate_random_splits(self):
        split_size = self.length // self.splits
        indices = torch.randperm(self.length, generator=self.generator)
        self.split_indices = torch.split(indices, split_size)

class MultiPersonsConcatDataset(PersonConcatDataset):
    def __init__(self, datasets):
        self.datasets = [d.get_person_dataset(p) for d in datasets for p in range(d.get_number_of_persons())]
        self.n_per_p = [len(d) for d in self.datasets]
        self.cummulative_n = np.cumsum(self.n_per_p)
        self.length = sum(self.n_per_p)

class NormalizationSelectDataset(PersonDataset):
    def __init__(self, dataset: PersonDataset, normalization_type: NormalizationType, add_face_distances=False, noise_head_yaw_pitch_stdev=0.0, noise_head_distance_stdev=0.0):
        self.dataset = dataset
        self.norm_type = normalization_type
        self.add_face_distances = add_face_distances
        self.noise_head_yaw_pitch_stdev = noise_head_yaw_pitch_stdev
        self.noise_head_distance_stdev = noise_head_distance_stdev
    
    def __len__(self):
        return len(self.dataset)

    def get_number_of_persons(self):
        return self.dataset.get_number_of_persons()
    
    def get_person_dataset(self, p):
        return NormalizationSelectDataset(self.dataset.get_person_dataset(p), self.norm_type)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]

        gaze_target = sample["target"]
        headpose = sample["hr"][:3,:3]

        left_eye = sample["left_eye_position"]
        right_eye = sample["right_eye_position"]
        face = 0.5 * (left_eye + right_eye)

        if self.norm_type == NormalizationType.ORIGINAL:
            left_headpose, left_gaze = normalize_gaze(left_eye*1000, headpose, gaze_target*1000)
            right_headpose, right_gaze = normalize_gaze(right_eye*1000, headpose, gaze_target*1000)
            n_headpose, gaze = normalize_gaze(face*1000, headpose, gaze_target*1000)
        elif self.norm_type == NormalizationType.NEW:
            left_headpose, left_gaze = normalize_gaze_new(left_eye*1000, headpose, gaze_target*1000)
            right_headpose, right_gaze = normalize_gaze_new(right_eye*1000, headpose, gaze_target*1000)
            n_headpose, gaze = normalize_gaze_new(face*1000, headpose, gaze_target*1000)
        new_sample = {
            "left_head_yaw_pitch": vector_to_yaw_pitch(cv2.Rodrigues(left_headpose)[0][:,2]) + np.random.randn(2) * self.noise_head_yaw_pitch_stdev,
            "right_head_yaw_pitch": vector_to_yaw_pitch(cv2.Rodrigues(right_headpose)[0][:,2]) + np.random.randn(2) * self.noise_head_yaw_pitch_stdev,
            "gaze": -gaze,
            "yaw_pitch": vector_to_yaw_pitch(-gaze),
            "image_l": sample["image_l"],
            "image_r": sample["image_r"],
            "target": gaze_target,
            "hr": headpose,
        }

        if self.add_face_distances:
            new_sample["left_head_yaw_pitch"] = np.append(new_sample["left_head_yaw_pitch"], np.linalg.norm(left_eye) + np.random.randn(1) * self.noise_head_distance_stdev)
            new_sample["right_head_yaw_pitch"] = np.append(new_sample["right_head_yaw_pitch"], np.linalg.norm(right_eye) + np.random.randn(1) * self.noise_head_distance_stdev)
        
        return new_sample

def k_fold_split(dataset, k=3, random=False, fold_index=0):
    npersons = dataset.get_number_of_persons()
    person_range = np.array(np.arange(0, npersons-1))
    if random:
        person_range = np.random.shuffle(person_range)

    n_per_fold = round(npersons / k)
    p_per_fold = [[] for i in range(k)]
    count = 0
    fold = 0

    for p in person_range:
        count += 1
        p_per_fold[fold].append(p)
        if count >= n_per_fold:
            count = 0
            fold += 1

    train_datasets = []
    val_datasets = []
    test_datasets = []

    for fold_p in np.array(p_per_fold)[np.arange(k) != fold_index]: # reverse order
        for p in fold_p:
            train_datasets.append(dataset.get_person_dataset(p))
    
    for fold_p in np.array(p_per_fold)[np.arange(k) == fold_index]:
        for p in fold_p:
            val_datasets.append(dataset.get_person_dataset(p))
            test_datasets.append(dataset.get_person_dataset(p))

    train_ds = torch.utils.data.ConcatDataset(train_datasets)
    val_ds = torch.utils.data.ConcatDataset(val_datasets)
    test_ds = torch.utils.data.ConcatDataset(test_datasets)
    return train_ds, val_ds, test_ds