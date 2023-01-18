import numpy as np
import torch
from bisect import bisect_left
import os
import scipy.io as sio
from glob import glob
import torchvision.transforms.functional as TF
from PIL import Image, ExifTags
import cv2
from pathlib import Path
import pickle

from ..utilities.geometry import compute_yaw_pitch, vector_to_yaw_pitch_gaze, vector_to_yaw_pitch_head, yaw_pitch_to_vector
from .helpers import normalize_img, PersonDataset

def get_camera_matrix(focal_length_mm=85.0):
    # This is based on the info of a Canon EOS Rebel T3i with a Canon EF-S 18–135 mm IS f/3.5–5.6 zoom lens as per the documentations of the dataset
    sensor_width = 22.3
    sensor_height = 14.9
    image_width = 5184
    image_height = 3456
    fx_px = focal_length_mm / sensor_width * image_width
    fy_px = focal_length_mm / sensor_height * image_height
    cam_matrix = np.eye(3)
    cam_matrix[0,0] = fx_px
    cam_matrix[1,1] = fy_px
    cam_matrix[0,2] = image_width/2
    cam_matrix[1,2] = image_height/2
    return cam_matrix


def get_focal_length_from_pil_exif(image: Image):
    exif = {ExifTags.TAGS[k]: v for k, v in image._getexif().items() if k in ExifTags.TAGS}
    return exif["FocalLength"]

class ColumbiaDataset(PersonDataset):
    @staticmethod
    def process_sample(person, idx_in_p, annotations=None, keep_full_image=False):
        image_path = person["image_paths"][idx_in_p]
        name_split = os.path.basename(image_path).split(".")[0].split("_")
        identifier = name_split[0]
        distance = float(name_split[1][:-1])
        head_horizontal = float(name_split[2][:-1]) # yaw is -head_horizontal to keep in line with other datasets
        vertical = float(name_split[3][:-1])
        horizontal = float(name_split[4][:-1])
        face = np.array([0., 0., 2.5])
        
        # gaze = -yaw_pitch_to_vector(np.radians(np.array([[horizontal - head_horizontal, vertical]])))
        gaze = -yaw_pitch_to_vector(np.radians(np.array([[-horizontal, vertical]])))
        face_dir = yaw_pitch_to_vector(np.radians(np.array([[-head_horizontal, 0]])))
        camera_dir = yaw_pitch_to_vector(np.radians(np.array([[head_horizontal, 0]])))
        down = np.array([0.,1.,0.])
        right = np.cross(down, face_dir)
        right /= np.linalg.norm(right)
        hr = np.c_[right, down, face_dir]

        # # line with point [0,0,2.5] (head position) and direction g (gaze) intersects plane at [0,0,0] with normal [0,0,1] (plane with targets)
        # # line parametrization is: g * t + [0,0,2.5]. Therefore, we check what t gives g_z * t + 2.5 = 0 (plane position).
        # t = -(distance + 0.5)/gaze[2]

        # target = gaze * t + np.array([0,0, distance + .5])

        # # camera position
        # camera_position = face - camera_dir * distance
        # camera_down = np.array([0.,1.,0.])
        # camera_right = np.cross(camera_down, camera_dir)
        # camera_right /= np.linalg.norm(camera_right)
        # camera_rotation = np.c_[camera_right, camera_down, camera_dir]


        # target_in_cam = camera_rotation.T @ (target - camera_position)
        # face_in_cam = camera_rotation.T @ (face - camera_position)
        # gaze_in_cam = (target_in_cam - face_in_cam) / np.linalg.norm(target_in_cam - face_in_cam)

        gaze_in_cam = gaze
        face_in_cam = np.array([0,0,distance])
        target_in_cam = face_in_cam + gaze_in_cam * (distance + 0.5)


        image = Image.open(image_path)
        M = get_camera_matrix(get_focal_length_from_pil_exif(image))

        sample = {
            "image": np.array(image),#.convert("RGB"),
            "image_path": image_path,
            "gaze": gaze_in_cam,
            "target": target_in_cam,
            "face": face_in_cam,
            "D": np.zeros((4,)),
            "M": M,
            "hr": hr
        }
        if annotations is not None:
            p = Path(image_path)
            im_id = str(Path(p.parent.parent.name, p.parent.name, p.name))
            pose_data = annotations[im_id]

            image_r, right_headpose, right_gaze = normalize_img(np.array(image), pose_data["right_eye_center"]*1000, pose_data["hR"], target_in_cam*1000, (64,64), M)
            image_l, left_headpose, left_gaze = normalize_img(np.array(image), pose_data["left_eye_center"]*1000, pose_data["hR"], target_in_cam*1000, (64,64), M)

            sample["left_eye_position"] = pose_data["left_eye_center"]/1000.0
            sample["right_eye_position"] = pose_data["right_eye_center"]/1000.0
            sample["hr"] = pose_data["hR"]
            sample["image_l"] = image_l
            sample["image_r"] = image_r
            sample["right_headpose"] = right_headpose
            sample["left_headpose"] = left_headpose
            sample["right_gaze"] = right_gaze
            sample["left_gaze"] = left_gaze

            if not keep_full_image:
                del sample["image"]


        return sample

    
    def __init__(self, path, annotations_file=None, keep_full_image_with_annotations=False):

        self.path = path
        self.keep_full_image_with_annotations = keep_full_image_with_annotations

        person_paths = sorted(glob(os.path.join(path, "*")))
        
        self.persons = []
        self.n_per_p = []
        
        for p in person_paths:
            num_samples = 0
            person = {
                "path": p,
                "image_paths": []
            }
            for im_name in sorted(glob(os.path.join(p, "*.jpg"))):
                person["image_paths"].append(im_name)
                num_samples += 1
            self.persons.append(person)
            self.n_per_p.append(num_samples)

        self.np = len(self.persons)
        self.cumulative_n = np.cumsum(self.n_per_p)
        self.n = sum(self.n_per_p)

        self.annotations = pickle.load(open(annotations_file, "rb")) if annotations_file is not None else None

    def get_number_of_persons(self):
        return self.np

    def get_persons_ids(self):
        return [p["id"] for p in self.persons]
    
    def get_person_dataset(self, p):
        return ColumbiaDatasetSinglePerson(self.persons[p], self.annotations, self.keep_full_image_with_annotations)

    def __len__(self):
        return self.n
    
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

        p = get_val(self.cumulative_n, idx)
        idx_in_p = idx - (self.cumulative_n[p-1] if p != 0 else 0)

        # sample = MPIIGazeDataset.process_one_sample(idx_in_p, self.persons[p], self.as_dataloader, self.square, self.square_size, self.undistort, self.rtgene_normalization)
        # image_path = self.persons[p]["image_paths"][idx_in_p]
        # name_split = os.path.basename(image_path).split(".")[0].split("_")
        # identifier = name_split[0]
        # distance = float(name_split[1][:-1])
        # head_horizontal = float(name_split[2][:-1])
        # vertical = float(name_split[3][:-1])
        # horizontal = float(name_split[4][:-1])
        
        # gaze = -yaw_pitch_to_vector(np.radians(np.array([[horizontal - head_horizontal, vertical]])))

        # # line with point [0,0,2.5] (head position) and direction g (gaze) intersects plane at [0,0,0] with normal [0,0,1] (plane with targets)
        # # line parametrization is: g * t + [0,0,2.5]. Therefore, we check what t gives g_z * t + 2.5 = 0 (plane position).
        # t = -2.5/gaze[2]

        # target = gaze * t + np.array([0,0,2.5])

        # image = Image.open(image_path)
        # M = get_camera_matrix(get_focal_length_from_pil_exif(image))

        # sample = {
        #     "image": image,#.convert("RGB"),
        #     "image_path": image_path,
        #     "gaze": gaze,
        #     "target": target,
        #     "face": np.array([0., 0., 2.5]),
        #     "D": np.zeros((4,)),
        #     "M": M
        # }
        sample = type(self).process_sample(self.persons[p], idx_in_p, self.annotations, self.keep_full_image_with_annotations)

        return sample

class ColumbiaDatasetSinglePerson(torch.utils.data.Dataset):
    def __init__(self, person, annotations, keep_full_image_with_annotations):
        self.n = len(person["image_paths"])
        self.annotations = annotations
        self.person = person
        self.keep_full_image_with_annotations = keep_full_image_with_annotations

    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = ColumbiaDataset.process_sample(self.person, idx, self.annotations, self.keep_full_image_with_annotations)
        return sample
