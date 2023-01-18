from matplotlib.pyplot import show
from pytorch_lightning.loggers import tensorboard
import torch
import os
from functools import partial
import pickle
import torchvision
from os.path import expanduser
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from gazeirislandmarks.datasets import (
    GazeCaptureDataset,
    MPIIGazeDataset, 
    MPIIFaceGazeDataset,
    random_split_person_dataset, 
    InMemoryParallelDataset, 
    InMemoryDataset,
    HDF5PersonDataset, 
    hdf5_to_sample, leave_one_out_split, 
    split_person_dataset, 
    TransformDataset, 
    PreprocessedDataset,
    HDF5Dataset,
    TransformPoseRandomDataset,
    NormalizationSelectDataset,
    NormalizationType
)
from gazeirislandmarks.models import train_custom
from gazeirislandmarks.datasets.helpers import (
    PersonConcatDataset,
    split_person_dataset, 
    split_faze_gazecapture, 
    split_faze_gazecapture_hdf5, 
    split_faze_mpii,
    MultiEpochDataset,
    TransformTorchvisionDataset,
    k_fold_split
)
from gazeirislandmarks.datasets.transforms import (
    RandomAffine,
    ColorJitter,
    GammaJitter,
    GaussianBlur,
    PadToSquare,
    GaussianNoise
)


@hydra.main(config_path="conf", config_name="default")
def train_custom_main(cfg: DictConfig) -> None:
    print("Working directory : {}".format(os.getcwd()))
    print(OmegaConf.to_yaml(cfg))
    cfg["dataset_path"] = expanduser(cfg["dataset_path"])
    if cfg["run_options"]["num_workers"] < 0:
        cfg["run_options"]["num_workers"] = os.cpu_count()
    config = {
        "name": cfg["name"],
        "dataset_path": cfg["dataset_path"],
        "architecture": cfg["architecture"],
        "train_options": cfg["train_options"] 
    }
    # config = copy.deepcopy(cfg)
    # config["architecture"] = cfg["architecture"] # TODO: remove after complete conversion/refactor
    

    for k in cfg["architecture"]:
        config[k] = cfg["architecture"][k]
    for k in cfg["train_options"]:
        config[k] = cfg["train_options"][k]
    for k in cfg["run_options"]:
        config[k] = cfg["run_options"][k]
    for k in cfg["dataset"]:
        config[k] = cfg["dataset"][k]

    # Fix some sweeper issues
    if cfg["architecture"]["split_gaze"]["n_layers_mod"] != 0:
        config["split_gaze"]["n_layers"] = [cfg["architecture"]["split_gaze"]["n_layers_mod"]]*3

    

    if cfg["run_options"].get("faze_split", False):
        split_func_mpii = split_faze_mpii
        split_func_gc = split_faze_gazecapture if not cfg["dataset"]["hdf5_name"] else split_faze_gazecapture_hdf5
    elif not cfg["run_options"].get("load_split", False):
        if not cfg["run_options"].get("leave_out"):
            split_func_mpii = partial(random_split_person_dataset, split=cfg["run_options"]["split"], save_indices_path=cfg["name"] + "_mpii_split.pkl", load_if_available=False)
            split_func_gc = partial(random_split_person_dataset, split=cfg["run_options"]["split"], save_indices_path=cfg["name"] + "_gc_split.pkl", load_if_available=False)
        else:
            val_index = cfg["run_options"]["leave_out"].get("idx_val")
            test_index = cfg["run_options"]["leave_out"].get("idx_test")
            split_func_mpii = partial(leave_one_out_split, val_index=val_index, test_index=test_index)
            split_func_gc = partial(leave_one_out_split, val_index=val_index, test_index=test_index)

        # if not cfg["run_options"].get("leave_out", False):
        #     split_func_mpii = partial(random_split_person_dataset, split=cfg["run_options"]["split"], save_indices_path=cfg["name"] + "_mpii_split.pkl", load_if_available=False)
        #     split_func_gc = partial(random_split_person_dataset, split=cfg["run_options"]["split"], save_indices_path=cfg["name"] + "_gc_split.pkl", load_if_available=False)
        # else:
        #     if cfg["run_options"]["leave_out_val"] < 0:
        #         val_person = cfg["run_options"]["leave_out"] - 1
        #         if val_person < 0:
        #             val_person = cfg["run_options"]["leave_out"] + 1
        #     else:
        #         val_person = cfg["run_options"]["leave_out_val"]
        #         if val_person == cfg["run_options"]["leave_out"]:
        #             val_person += 1
        #     split_func_mpii = partial(leave_one_out_split, person_index=cfg["run_options"]["leave_out"], validation_person_index=val_person)
        #     split_func_gc = partial(leave_one_out_split, person_index=cfg["run_options"]["leave_out"], validation_person_index=val_person)
    else:
        split_func_mpii = partial(split_person_dataset, indices=pickle.load(open(cfg["run_options"]["load_split"] + "_mpii_split.pkl", "rb")))
        split_func_gc = partial(split_person_dataset, indices=pickle.load(open(cfg["run_options"]["load_split"] + "_mpii_split.pkl", "rb")))

    # select dataset
    if cfg["dataset"]["hdf5_name"]:
        dataset = HDF5PersonDataset(cfg["dataset_path"] + cfg["dataset"]["hdf5_name"], image_format="JPEG", in_memory=cfg["run_options"]["memory"])
    else:
        if cfg["dataset"]["mpiig"]:
            dataset = MPIIGazeDataset(cfg["dataset_path"], as_dataloader=True, square=True, square_size=720, undistort=True)
        elif cfg["dataset"]["mpiifg"]:
            dataset = MPIIFaceGazeDataset(cfg["dataset_path"], as_dataloader=True, square=True, square_size=720, undistort=True,
                        use_more_annotations=False, gaze_dataset_eye_middle_path=os.path.join(cfg["dataset_path"], "..", "MPIIGaze"))
        elif cfg["dataset"]["gc"]:
            dataset = GazeCaptureDataset(cfg["dataset_path"], as_dataloader=True, square=True, square_size=720, use_more_annotations=False, undistort=True)

    if cfg["run_options"].get("k_fold_validation", 0) == 0:
        if cfg["dataset"]["gc"]:
            train_dataset, val_dataset, test_dataset = split_func_gc(dataset)
        elif cfg["dataset"]["mpiig"] or cfg["dataset"]["mpiifg"]:
            train_dataset, val_dataset, test_dataset = split_func_mpii(dataset)
    else:
        k_folds = cfg["run_options"]["k_fold_validation"]
        k_folds_idx = cfg["run_options"].get("k_fold_validation_idx", 0)
        k_fold_random = cfg["run_options"].get("k_fold_validation_random", False)
        train_dataset, val_dataset, test_dataset = k_fold_split(dataset, k=k_folds, random=k_fold_random, fold_index=k_folds_idx)
    
    if config["train_options"]["person_loss"]:
        if cfg["run_options"]["val"]:
            train_datasets = train_dataset.datasets
            val_dataset = PersonConcatDataset(val_dataset.datasets)
        else:
            train_datasets = train_dataset.datasets + val_dataset.datasets
            val_dataset = PersonConcatDataset(test_dataset.datasets)
        test_dataset = PersonConcatDataset(test_dataset.datasets)
    else:
        # Convert to person format
        if cfg["run_options"]["val"]:
            train_datasets = [PersonConcatDataset(train_dataset.datasets)]
            val_dataset = PersonConcatDataset(val_dataset.datasets)
        else:
            train_datasets = [PersonConcatDataset(train_dataset.datasets + val_dataset.datasets)]
            val_dataset = PersonConcatDataset(test_dataset.datasets)
        test_dataset = PersonConcatDataset(test_dataset.datasets)

    if cfg["run_options"].get("resize_images") is not None:
        resize_transform = torchvision.transforms.Compose([
            PadToSquare(),
            torchvision.transforms.Resize((64,64))
        ])

    for i, train_dataset in enumerate(train_datasets):
        if cfg["run_options"]["augment"]["enabled"]:
            if cfg["run_options"]["augment"]["color_jitter"]:
                brightness = cfg["run_options"]["augment"]["color_jitter_brightness"]
                contrast = cfg["run_options"]["augment"]["color_jitter_contrast"]
                saturation = cfg["run_options"]["augment"]["color_jitter_saturation"]
                hue = cfg["run_options"]["augment"]["color_jitter_hue"]
                train_dataset = TransformDataset(train_dataset, ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue), keep_original=False, apply_constantly_on_sample=True) 
            
            if cfg["run_options"]["augment"]["random_affine"]:
                degrees = cfg["run_options"]["augment"]["random_affine_degrees_variation"]
                translate = [cfg["run_options"]["augment"]["random_affine_translate"], cfg["run_options"]["augment"]["random_affine_translate"]]
                scale = [1.0 - cfg["run_options"]["augment"]["random_affine_scale_variation"], 1.0 + cfg["run_options"]["augment"]["random_affine_scale_variation"]]
                train_dataset = TransformDataset(train_dataset, RandomAffine(degrees=degrees, translate=translate, scale=scale, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, fill=[124, 116, 103]), keep_original=False, apply_constantly_on_sample=True)
            
            if cfg["run_options"]["augment"]["grayscale"]:
                train_dataset = TransformDataset(train_dataset, torchvision.transforms.Grayscale(3))
            
            if cfg["run_options"]["augment"].get("dev_head_yaw_pitch"):
                train_dataset = TransformPoseRandomDataset(train_dataset, cfg["run_options"]["augment"].get("dev_head_yaw_pitch"), ["left_head_yaw_pitch", "right_head_yaw_pitch"])
            
            if cfg["run_options"]["augment"].get("dev_gamma"):
                train_dataset = TransformDataset(train_dataset, GammaJitter(cfg["run_options"]["augment"].get("dev_gamma")))
            
            if cfg["run_options"]["augment"].get("gaussian_blur"):
                train_dataset = TransformDataset(
                    train_dataset, 
                    GaussianBlur(
                        cfg["run_options"]["augment"]["gaussian_blur"].get("kernel_size", 3),
                        (
                            cfg["run_options"]["augment"]["gaussian_blur"].get("sigma_1", 3),
                            cfg["run_options"]["augment"]["gaussian_blur"].get("sigma_2", 3)
                        )
                    )
                )
            
            if cfg["run_options"]["augment"].get("gaussian_noise_sigma", 0.0) != 0.0 :
                train_dataset = TransformDataset(
                    train_dataset, 
                    GaussianNoise(
                        cfg["run_options"]["augment"]["gaussian_noise_sigma"]
                    )
                )

            if cfg["run_options"]["preprocess"]:
                train_dataset = PreprocessedDataset(train_dataset)
                val_dataset = PreprocessedDataset(val_dataset)
                test_dataset = PreprocessedDataset(test_dataset)

        if cfg["run_options"].get("new_norm", "None") != "None":
            if cfg["run_options"]["new_norm"]:
                train_dataset = NormalizationSelectDataset(train_dataset, NormalizationType.NEW, config["architecture"].get("face_distance", False), cfg["run_options"]["augment"].get("dev_head_yaw_pitch", 0.0), cfg["run_options"]["augment"].get("dev_head_distance", 0.0))
            else:
                train_dataset = NormalizationSelectDataset(train_dataset, NormalizationType.ORIGINAL, config["architecture"].get("face_distance", False), cfg["run_options"]["augment"].get("dev_head_yaw_pitch", 0.0), cfg["run_options"]["augment"].get("dev_head_distance", 0.0))
        
        if cfg["run_options"].get("resize_images") is not None:
            train_dataset = TransformTorchvisionDataset(train_dataset, resize_transform)
        
        if cfg["train_options"].get("split_train_epochs"):
            train_dataset = MultiEpochDataset(train_dataset, splits=config["train_options"].get("split_train_epochs"))

        train_datasets[i] = train_dataset

        
    
    # if not cfg["run_options"]["val"]:
    #     train_dataset = PersonConcatDataset([train_dataset, val_dataset])
    #     val_dataset = test_dataset

    if cfg["run_options"].get("new_norm", "None") != "None":
        if cfg["run_options"]["new_norm"]:
            val_dataset = NormalizationSelectDataset(val_dataset, NormalizationType.NEW, config["architecture"].get("face_distance", False))
            test_dataset = NormalizationSelectDataset(test_dataset, NormalizationType.NEW, config["architecture"].get("face_distance", False))
        else:
            val_dataset = NormalizationSelectDataset(val_dataset, NormalizationType.ORIGINAL, config["architecture"].get("face_distance", False))
            test_dataset = NormalizationSelectDataset(test_dataset, NormalizationType.ORIGINAL, config["architecture"].get("face_distance", False))
    
    if cfg["run_options"].get("resize_images") is not None:
        val_dataset = TransformTorchvisionDataset(val_dataset, resize_transform)
        test_dataset = TransformTorchvisionDataset(test_dataset, resize_transform)

    if cfg["run_options"]["num_workers"]:
        def worker_init_fn(idx):
            def step_dataset_tree(d):
                if hasattr(d, "dataset"):
                    if isinstance(d.dataset, HDF5Dataset) or isinstance(d.dataset, HDF5PersonDataset):
                        d.dataset.reopen_file()
                    else:
                        step_dataset_tree(d.dataset)
                elif hasattr(d, "datasets"):
                    for sd in d.datasets:
                        step_dataset_tree(sd)
            info = torch.utils.data.get_worker_info()
            d = info.dataset
            step_dataset_tree(d)
        
        train_dl = [
            torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], 
            shuffle=True, num_workers=cfg["run_options"]["num_workers"], prefetch_factor=4, 
            worker_init_fn=worker_init_fn, pin_memory=True, drop_last=True if cfg["train_options"].get("bn_splits",0) else False) for train_dataset in train_datasets
        ]
        val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=cfg["run_options"]["num_workers"], prefetch_factor=4, worker_init_fn=worker_init_fn, pin_memory=True, drop_last=True if cfg["train_options"].get("bn_splits",0) else False)
        test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=cfg["run_options"]["num_workers"], prefetch_factor=4, worker_init_fn=worker_init_fn, pin_memory=True, drop_last=True if cfg["train_options"].get("bn_splits",0) else False)
    else:
        train_dl = [
            torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], 
            shuffle=True, num_workers=cfg["run_options"]["num_workers"], pin_memory=True, drop_last=True if cfg["train_options"].get("bn_splits",0) else False)
            for train_dataset in train_datasets
        ]
        val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=cfg["run_options"]["num_workers"], pin_memory=True, drop_last=True if cfg["train_options"].get("bn_splits",0) else False)
        test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=cfg["run_options"]["num_workers"], pin_memory=True, drop_last=True if cfg["train_options"].get("bn_splits",0) else False)
    
    # config["no_batchnorm"] = True
    # if config["train_options"]["person_loss"]:
    #     train_custom(config, train_dl, val_dl, test_dl)
    # else:
    #     return train_custom(config, train_dl[0], val_dl, test_dl)

    if cfg.get("debug_augmentation"):
        from gazeirislandmarks.utilities.image import show_dual_images
        # import keyboard 
        for tdl in train_dl:
            for s in tdl:
                for i in range(s["image_l"].shape[0]):
                    show_dual_images(s["image_r"][i,...], s["image_l"][i,...], show=True, block=True)
                    
    test_error, model = train_custom(config, train_dl, val_dl, test_dl)

    return test_error

if __name__ == '__main__':
    train_custom_main()