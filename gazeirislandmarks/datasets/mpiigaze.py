import numpy as np
import torch
from bisect import bisect_left
import os
import scipy.io as sio
from glob import glob
import torchvision.transforms.functional as TF
from PIL import Image
import cv2

from ..utilities.geometry import compute_yaw_pitch, vector_to_yaw_pitch_gaze, vector_to_yaw_pitch_head
from .helpers import normalize_img, PersonDataset

class MPIIGazeDataset(PersonDataset):
    # static variables
    cached_M = None
    cached_D = None
    cached_map1 = None
    cached_map2 = None

    # with adjustment to the camera matrix, per: https://stackoverflow.com/questions/59477398/how-does-cropping-an-image-affect-camera-calibration-intrinsics
    @staticmethod
    def crop_to_square(image, face_center, size=None):
        smallest_side = min(image.shape[0], image.shape[1])
        if image.shape[0] == smallest_side:
            ymin = 0
            xmin = max(int(face_center[0] - smallest_side/2), 0)
        else:
            xmin = 0
            ymin = max(int(face_center[1] - smallest_side/2), 0)
        cropped = image[ymin:(ymin + smallest_side), xmin:(xmin + smallest_side), :]
        if not size is None:
            return TF.resize(Image.fromarray(cropped), (size, size)), xmin, ymin, size/cropped.shape[0]
        return cropped, xmin, ymin, 1.0

    @staticmethod
    def process_one_sample(idx, person, as_dataloader=False, square=False, square_size=None, undistort=True, rtgene_normalization=False):
            # Start with calibration
            M = np.copy(person["M"])
            D = np.copy(person["D"])
            image_path = person["images"][idx]
            target = person["targets"][idx]
            # image = np.array(imageio.imread(image_path))
            image = Image.open(image_path)
            # ratio = 1

            if rtgene_normalization:
                if undistort:
                    if not np.array_equal(MPIIGazeDataset.cached_M, M) or not np.array_equal(MPIIGazeDataset.cached_D, D):
                        M_new, roi = cv2.getOptimalNewCameraMatrix(M, D, (image.height, image.width), alpha=0)
                        MPIIGazeDataset.cached_map1, MPIIGazeDataset.cached_map2 = cv2.initUndistortRectifyMap(M, D, None, M_new, (image.width, image.height), cv2.CV_32FC1)
                        M = M_new
                    image = Image.fromarray(cv2.remap(np.array(image), MPIIGazeDataset.cached_map1, MPIIGazeDataset.cached_map2, interpolation=cv2.INTER_LINEAR))
                image_r, right_headpose, right_gaze = normalize_img(np.array(image), person["poses"][idx]["right_eye_center"], person["poses"][idx]["hR"], target*1000, (64,64), M)
                image_l, left_headpose, left_gaze = normalize_img(np.array(image), person["poses"][idx]["left_eye_center"], person["poses"][idx]["hR"], target*1000, (64,64), M)
                yaw_pitch_r = vector_to_yaw_pitch_gaze(-right_gaze)
                yaw_pitch_l = vector_to_yaw_pitch_gaze(-left_gaze)
                head_yaw_pitch_r = vector_to_yaw_pitch_head(cv2.Rodrigues(right_headpose)[0][:,2])
                head_yaw_pitch_l = vector_to_yaw_pitch_head(cv2.Rodrigues(left_headpose)[0][:,2])
                yaw_pitch = (yaw_pitch_r + yaw_pitch_l) / 2.0
                head_yaw_pitch = vector_to_yaw_pitch_head(person["poses"][idx]["hR"][:,2])

                real_yaw_pitch = vector_to_yaw_pitch_gaze(target - person["poses"][idx]["face"]/1000)
                out = {
                        "M": M, 
                        "D": D.squeeze(), 
                        "target":target, 
                        "face": person["poses"][idx]["face"]/1000,
                        # "real_face": person["poses"][idx]["face"]/1000,
                        "image_path": image_path,
                        "image_l": np.array(image_l, copy=False).copy(),
                        "image_r": np.array(image_r, copy=False).copy(),
                        "yaw_pitch": yaw_pitch.astype(np.float32),
                        "real_yaw_pitch": real_yaw_pitch.astype(np.float32),
                        "right_headpose": right_headpose,
                        "left_headpose": left_headpose,
                        "right_gaze": -right_gaze,
                        "left_gaze": -left_gaze,
                        "left_yaw_pitch": vector_to_yaw_pitch_gaze(-left_gaze),
                        "right_yaw_pitch": vector_to_yaw_pitch_gaze(-right_gaze),
                        "right_eye_position": person["poses"][idx]["right_eye_center"]/1000,
                        "left_eye_position": person["poses"][idx]["left_eye_center"]/1000,
                        "left_head_yaw_pitch": head_yaw_pitch_l,
                        "right_head_yaw_pitch": head_yaw_pitch_r,
                        "head_yaw_pitch": head_yaw_pitch,
                        "hr": person["poses"][idx]["hR"],
                    }
                if not as_dataloader:
                    out["image"] = np.array(image)
                return out
            else:
                if undistort:
                    # w = image.width
                    # ratio = 1
                    # while (w/2 - M[0,2])/(w/2) < -0.1: # correct for resized image not fitting with camera matrix
                    #     ratio *= 2
                    #     w = ratio * image.width
                    #     if not ((w/2 - M[0,2])/(w/2) < -0.1):
                    #         break
                    # while (w/2 - M[0,2])/(w/2) > 0.1:
                    #     ratio /= 2
                    #     w = ratio * image.width
                    #     if not ((w/2 - M[0,2])/(w/2) > 0.1):
                    #         break

                    # image = TF.resize(image, [image.height * ratio, image.width * ratio])
                    if not np.array_equal(MPIIGazeDataset.cached_M, M) or not np.array_equal(MPIIGazeDataset.cached_D, D):
                        M_new, roi = cv2.getOptimalNewCameraMatrix(M, D, (image.height, image.width), alpha=0)
                        MPIIGazeDataset.cached_map1, MPIIGazeDataset.cached_map2 = cv2.initUndistortRectifyMap(M, D, None, M_new, (image.width, image.height), cv2.CV_32FC1)
                        M = M_new
                    image = Image.fromarray(cv2.remap(np.array(image), MPIIGazeDataset.cached_map1, MPIIGazeDataset.cached_map2, interpolation=cv2.INTER_LINEAR))

                # if undistort:
                #     image = Image.fromarray(cv2.undistort(np.array(image), M, D))

                face_center = person["faces"][idx]
                landmarks = person["landmarks"][idx]# * ratio

                if square:
                    image, dx, dy, s = MPIIGazeDataset.crop_to_square(np.array(image), landmarks[:4, :].mean(axis=0), square_size)
                    # adjust camera matrix
                    M[0,2] -= dx
                    M[1,2] -= dy
                    # landmarks[:, 0] -= dx
                    # landmarks[:, 1] -= dy
                    if s != 1.0:
                        M[:2, :3] *= s
                        # landmarks *= s
                else:
                    dx = 0
                    dy = 0
                    s = 1.0

                # Compute image_l and image_r
                left_eye_width = (landmarks[9,0] - landmarks[6,0])*3
                right_eye_width = (landmarks[3,0] - landmarks[0,0])*3
                left_eye_center = (landmarks[9,:] + landmarks[6,:])/2.0
                right_eye_center = (landmarks[3,:] + landmarks[0,:])/2.0
                left_eye_lt = left_eye_center - left_eye_width/2
                right_eye_lt = right_eye_center - right_eye_width/2

                yaw_pitch = compute_yaw_pitch(face_center, target)
                gaze_direction_mirror = target - face_center
                gaze_direction_mirror[0] = -gaze_direction_mirror[0]
                yaw_pitch_mirror = vector_to_yaw_pitch_gaze(gaze_direction_mirror)

                if undistort:
                    left_eye_lt = np.array(left_eye_lt, dtype=np.float)
                    left_eye_lt = cv2.undistortPoints(left_eye_lt, M, D, P=M).squeeze()

                    right_eye_lt = np.array(right_eye_lt, dtype=np.float)
                    right_eye_lt = cv2.undistortPoints(right_eye_lt, M, D, P=M).squeeze()

                image_l = TF.resized_crop(image, left_eye_lt[1], left_eye_lt[0], left_eye_width, left_eye_width, (64,64))
                image_r = TF.resized_crop(image, right_eye_lt[1], right_eye_lt[0], right_eye_width, right_eye_width, (64,64))

                if as_dataloader:
                    return {
                        "M": M, 
                        "D": D.squeeze(), 
                        "target":target, 
                        "face": face_center, 
                        "image_path": image_path,
                        "image_l": np.array(image_l),
                        "image_r": np.array(image_r),
                        "yaw_pitch": yaw_pitch.astype(np.float32),
                        "yaw_pitch_mirror": yaw_pitch_mirror.astype(np.float32)
                    }
                else:
                    return {
                        "M": M, 
                        "D": D.squeeze(), 
                        "image": np.array(image), 
                        "target":target, 
                        "face": face_center, 
                        "image_path": image_path,
                        "image_l": np.array(image_l),
                        "image_r": np.array(image_r),
                        "yaw_pitch": yaw_pitch.astype(np.float32),
                        "yaw_pitch_mirror": yaw_pitch_mirror.astype(np.float32),
                        "x_offset": dx,
                        "y_offset": dy,
                        "scale_offset": s
                    }
    
    def __init__(self, path, as_dataloader=False, square=False, square_size=None, undistort=False, use_rtgene_normalization=False):

        self.as_dataloader = as_dataloader
        self.path = path
        self.square = square
        self.square_size = square_size
        self.undistort = undistort
        self.rtgene_normalization = use_rtgene_normalization

        self.face_model = sio.loadmat(os.path.join(path, '6 points-based face model.mat'))["model"]

        # List person directories
        personPaths = sorted(glob(path + "/Data/Original/*/"))
        
        self.n_per_p = []
        self.n = 0
        self.cumulative_n = []
        self.persons = []
        for p in personPaths:
            imagePaths = []
            gaze_targets = []
            face_centers = []
            landmarks = []
            poses = []
            for d in sorted(glob(p + "/day*/")):
                annotationsFile = d + "annotation.txt"
                with open(annotationsFile, "r") as f:
                    lines = f.readlines()
                    for l in lines:
                        data = l.split(" ")
                        gaze_target = np.array(data[26:29], dtype=np.float32)/1000.0
                        face_center = (np.array(data[35:38], dtype=np.float32) + np.array(data[38:41], dtype=np.float32))/2/1000.0
                        landmark = np.array(data[0:24], dtype=np.float32).reshape((12,2))


                        # Test RT-GENE
                        headpose_hr = np.array(data[29:32], dtype=np.float32)
                        headpose_ht = np.array(data[32:35], dtype=np.float32)
                        hR = cv2.Rodrigues(headpose_hr)[0]
                        Fc = np.dot(hR, self.face_model)
                        Fc = headpose_ht.T[:, np.newaxis] + Fc

                        right_eye_center = 0.5 * (Fc[:, 0] + Fc[:, 1])
                        left_eye_center = 0.5 * (Fc[:, 2] + Fc[:, 3])

                        poses.append({
                            "headpose_hr": headpose_hr,
                            "headpose_ht": headpose_ht,
                            "hR": hR,
                            "right_eye_center": right_eye_center,
                            "left_eye_center": left_eye_center,
                            "face": 0.5 * (right_eye_center + left_eye_center)
                        })

                        # imagePaths.append(imagePath)
                        gaze_targets.append(gaze_target)
                        face_centers.append(face_center)
                        landmarks.append(landmark)
                # List all images
                imagePaths += sorted(glob(d + "/*.jpg"))

            identifier = os.path.basename(os.path.normpath(p))
            calibrationMat = sio.loadmat(p + "Calibration/Camera.mat")

            person = {
                "id": identifier,
                "path": p,
                "M": calibrationMat["cameraMatrix"],
                "D": calibrationMat["distCoeffs"],
                "n": len(imagePaths),
                "images": imagePaths,
                "targets": np.array(gaze_targets, dtype=np.float32),
                "faces": np.array(face_centers, dtype=np.float32),
                "landmarks": landmarks,
                "poses": poses
            }
            n = len(imagePaths)
            self.n_per_p.append(n)
            self.n += n
            self.cumulative_n.append(self.n)
            self.persons.append(person)
        
        self.np = len(self.persons)

    def get_number_of_persons(self):
        return self.np

    def get_persons_ids(self):
        return [p["id"] for p in self.persons]
    
    def get_person_dataset(self, p):
        return MPIIGazeDatasetSinglePerson(self.persons[p], self.as_dataloader, self.square, self.square_size, self.undistort, self.rtgene_normalization)

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

        sample = MPIIGazeDataset.process_one_sample(idx_in_p, self.persons[p], self.as_dataloader, self.square, self.square_size, self.undistort, self.rtgene_normalization)
        return sample

class MPIIGazeDatasetSinglePerson(torch.utils.data.Dataset):
    def __init__(self, person, as_dataloader=False, square=False, square_size=None, undistort=False, rtgene_normalization=False):
        self.as_dataloader = as_dataloader
        self.n = len(person["images"])
        self.person = person
        self.square = square
        self.square_size = square_size
        self.undistort = undistort
        self.rtgene_normalization = rtgene_normalization

    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = MPIIGazeDataset.process_one_sample(idx, self.person, self.as_dataloader, self.square, self.square_size, self.undistort, self.rtgene_normalization)
        return sample

    def get_camera_matrices(self):
        return self.person["M"], self.person["D"]
