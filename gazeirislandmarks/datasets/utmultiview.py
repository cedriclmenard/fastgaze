import numpy as np
import torch
from bisect import bisect_left
import os
import scipy.io as sio
from glob import glob
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
import os.path
from zipfile import ZipFile

import scipy.linalg

from ..utilities.geometry import compute_yaw_pitch, vector_to_yaw_pitch_gaze, vector_to_yaw_pitch_head
from .helpers import normalize_img, PersonDataset
from pytransform3d import rotations

# def rq(A): 
#  Q,R = qr(flipud(A).T)
#  R = flipud(R.T)
#  Q = Q.T 
#  return R[:,::-1],Q[::-1,:]

def separate_projective_matrix(P):
    # I love it when people can't be bothered annotating their dataset as they should :)
    PH = np.append(P, np.array([[0,0,0,1]]), axis=0)
    r, q = scipy.linalg.rq(PH)
    fx = np.abs(r[0,0])
    fy = np.abs(r[1,1])
    cx = np.abs(r[0,2])
    cy = np.abs(r[1,2])

    Pi = np.array([[fx, 0, cx], [0,fy,cy],[0,0,1]])
    PiH = np.array([[fx, 0, cx,0], [0,fy,cy,0],[0,0,1,0],[0,0,0,1]])

    T = np.linalg.inv(PiH) @ (PH)
    return Pi, T

def separate_projective_matrix2(P):
    # https://ksimek.github.io/2012/08/14/decompose/
    # http://web.archive.org/web/20161116083912/http://www.janeriksolem.net/2011/03/rq-factorization-of-camera-matrices.html

    K, R = scipy.linalg.rq(P[:,:3])

    # make diagonal of K positive
    T = np.diag(np.sign(np.diag(K)))
    K = np.dot(K, T)
    R = np.dot(T, R) # T is its own inverse
    
    # t = - np.linalg.inv(P[:,:3]).dot(P[:,3])
    t = np.linalg.inv(K) @ P[:,3]
    return K, R, t

def lines_to_array(lines):
    if len(lines) == 1:
        return np.fromstring(lines[0][1:-1], sep=" ")
    if len(lines) == 2:
        first_line = np.fromstring(lines[0][2:-1], sep=" ")
        data = np.empty((2,len(first_line)))
        data[0,:] = first_line
        data[1,:] = np.fromstring(lines[0][2:-2], sep=" ")
        return data
    first_line = np.fromstring(lines[0][2:-1], sep=" ")
    data = np.empty((len(lines),len(first_line)))
    data[0,:] = first_line
    for i in range(1,len(lines)-1):
        data[i,:] = np.fromstring(lines[i][2:-1], sep=" ")
    data[-1,:] = np.fromstring(lines[-1][2:-2], sep=" ")
    return data


class UTMultiviewDataset(PersonDataset):

    @staticmethod
    def process_one_sample(idx, person, as_dataloader=False, rtgene_normalization=False):
            # Start with calibration
            M = np.copy(person["M"][idx])
            # D = np.copy(person["D"])
            image_path = person["images"][idx]
            target = person["targets"][idx]
            # image = np.array(imageio.imread(image_path))
            image = Image.open(image_path)
            # ratio = 1

            if rtgene_normalization:
                image_r, right_headpose, right_gaze = normalize_img(np.array(image), person["right_eye_centers"][idx], person["head_rs"][idx], target, (64,64), M, focal_new=800)
                image_l, left_headpose, left_gaze = normalize_img(np.array(image), person["left_eye_centers"][idx], person["head_rs"][idx], target, (64,64), M, focal_new=800)
                yaw_pitch_r = vector_to_yaw_pitch_gaze(-right_gaze)
                yaw_pitch_l = vector_to_yaw_pitch_gaze(-left_gaze)
                head_yaw_pitch_r = vector_to_yaw_pitch_head(cv2.Rodrigues(right_headpose)[0][:,2])
                head_yaw_pitch_l = vector_to_yaw_pitch_head(cv2.Rodrigues(left_headpose)[0][:,2])
                yaw_pitch = (yaw_pitch_r + yaw_pitch_l) / 2.0
                head_yaw_pitch = vector_to_yaw_pitch_head(person["head_rs"][idx][:,2])

                real_yaw_pitch = vector_to_yaw_pitch_gaze(target - person["faces"][idx])
                out = {
                        "M": M, 
                        "D": np.zeros((4,)), 
                        "target":target/1000, 
                        "face": person["faces"][idx]/1000,
                        # "real_face": person["poses"][idx]["face"]/1000,
                        "image_path": image_path,
                        "image_l": np.array(image_l, copy=False).copy(),
                        "image_r": np.array(image_r, copy=False).copy(),
                        "yaw_pitch": yaw_pitch.astype(np.float32),
                        "real_yaw_pitch": real_yaw_pitch.astype(np.float32),
                        "right_headpose": right_headpose,
                        "left_headpose": left_headpose,
                        "hr": person["head_rs"][idx],
                        # "ht": person["head_ts"][idx],
                        "right_gaze": -right_gaze,
                        "left_gaze": -left_gaze,
                        "left_yaw_pitch": vector_to_yaw_pitch_gaze(-left_gaze),
                        "right_yaw_pitch": vector_to_yaw_pitch_gaze(-right_gaze),
                        "right_eye_position": person["right_eye_centers"][idx]/1000,
                        "left_eye_position": person["left_eye_centers"][idx]/1000,
                        "left_head_yaw_pitch": head_yaw_pitch_l,
                        "right_head_yaw_pitch": head_yaw_pitch_r,
                        "head_yaw_pitch": head_yaw_pitch
                    }
                if not as_dataloader:
                    out["image"] = np.array(image)
                return out
            else:
                raise NotImplementedError
    
    def __init__(self, path, as_dataloader=False, use_rtgene_normalization=False, **kw):

        self.as_dataloader = as_dataloader
        self.path = path
        self.rtgene_normalization = use_rtgene_normalization

        # self.face_model = sio.loadmat(os.path.join(path, '6 points-based face model.mat'))["model"]

        # List person directories
        persons_path = sorted(glob(path + "/*/"))
        
        self.n_per_p = []
        self.n = 0
        self.cumulative_n = []
        self.persons = []
        self.features_samples = []
        self.headpose_samples = []
        for p in persons_path:
            image_paths = []
            gaze_targets = []
            face_centers = []
            right_eye_centers = []
            left_eye_centers = []
            head_rs = []
            head_ts = []
            camera_matrices = []
            
            
            with open(os.path.join(p, "raw", "monitor.txt"), "r") as f:
                lines = f.readlines()
                # monitor_t = np.fromstring(data[1][1:-1], sep=" ")
                # monitor_r = np.empty((3,3))
                # monitor_r[0,:] = np.fromstring(data[2][2:-1], sep=" ")
                # monitor_r[1,:] = np.fromstring(data[3][2:-1], sep=" ")
                # monitor_r[2,:] = np.fromstring(data[4][2:-2], sep=" ")
                monitor_t = lines_to_array([lines[1]])
                monitor_r = lines_to_array(lines[2:])




            with open(os.path.join(p, "raw", "gazedata.csv"), "r") as f:
                gaze_data = f.readlines()
            
            del gaze_data[0]

            for line in gaze_data:
                data = line.split(",")
                sample_number = data[0]
                g_m = np.zeros((3,))
                g_m[0] = float(data[1])
                g_m[1] = float(data[2])

                g_w = monitor_r @ g_m + monitor_t

                sample_dir = os.path.join(p, "raw", "img" + sample_number.zfill(3))

                with open(os.path.join(sample_dir, "headpose.txt"), "r") as f:
                    lines = f.readlines()
                    head_t = lines_to_array([lines[1]])
                    head_r = lines_to_array(lines[2:5])
                    features = lines_to_array(lines[6:])

                self.features_samples.append(features)
                self.headpose_samples.append(head_r)

                for i in range(8):
                    # camera
                    with open(os.path.join(sample_dir, "cparams", "0"*7 + str(i) + ".txt"), "r") as f:
                        lines = f.readlines()
                        camera_p = np.array([np.fromstring(l, sep=" ") for l in lines[1:]])
                        # https://www.wolframalpha.com/input?i=%7B%7Bf_x%2C+0%2C+c_x%7D%2C%7B0%2C+f_y%2C+c_y%7D%2C%7B0%2C0%2C1%7D%7D+*+%7B%7Br_1%2C+r_2%2C+r_3%2C+t_x%7D%2C+%7Br_4%2Cr_5%2Cr_6%2Ct_y%7D%2C%7Br_7%2Cr_8%2Cr_9%2Ct_z%7D%7D
                        # {{f_x, 0, c_x},{0, f_y, c_y},{0,0,1}} * {{r_1, r_2, r_3, t_x}, {r_4,r_5,r_6,t_y},{r_7,r_8,r_9,t_z}}
                        # {{f_x r_1 + c_x r_7, f_x r_2 + c_x r_8, f_x r_3 + c_x r_9, f_x t_x + c_x t_z}, {f_y r_4 + c_y r_7, f_y r_5 + c_y r_8, f_y r_6 + c_y r_9, f_y t_y + c_y t_z}, {r_7, r_8, r_9, t_z}}
                        #
                        # https://www.wolframalpha.com/input?i=%7B%7Bf_x%2C0%2Cc_x%2C0%7D%2C%7B0%2Cf_y%2Cc_y%2C0%7D%2C%7B0%2C0%2C1%2C0%7D%2C%7B0%2C0%2C0%2C1%7D%7D+*+%7B%7B1%2C0%2C0%2Ct_x%7D%2C%7B0%2C1%2C0%2Ct_y%7D%2C%7B0%2C0%2C1%2Ct_z%7D%2C%7B0%2C0%2C0%2C1%7D%7D
                        # C, T = separate_projective_matrix(camera_p)
                        C, R, t = separate_projective_matrix2(camera_p)
                        T = np.eye(4,4)
                        T[:3,:3] = R
                        T[:3,3] = t
                        camera_matrices.append(C)

                        image_paths.append(os.path.join(sample_dir, "images", "0"*7 + str(i) + ".jpg"))
                        gaze_targets.append((T @ np.append(g_w,1))[:3])
                        head_rs.append(T[:3,:3] @ head_r)
                        ht_cam = (T @ np.append(head_t, 1))
                        ht_cam /= ht_cam[-1]
                        head_ts.append(ht_cam[:3])
                        
                        left_eye = (T @ np.append((features[2,:] + features[3,:])/2.0, 1))[:3]
                        right_eye = (T @ np.append((features[0,:] + features[1,:])/2.0, 1))[:3]

                        left_eye_centers.append(left_eye)
                        right_eye_centers.append(right_eye)

                        face_centers.append((left_eye + right_eye)/2.0)

            identifier = os.path.basename(os.path.normpath(p))

            person = {
                "id": identifier,
                "path": p,
                "M": camera_matrices,
                "n": len(image_paths),
                "images": image_paths,
                "targets": np.array(gaze_targets, dtype=np.float32),
                "faces": np.array(face_centers, dtype=np.float32),
                "head_rs": head_rs,
                "head_ts": head_ts,
                "right_eye_centers": right_eye_centers,
                "left_eye_centers": left_eye_centers
            }
            n = len(image_paths)
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
        return UTMultiviewDatasetSinglePerson(self.persons[p], self.as_dataloader, self.rtgene_normalization)

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

        sample = UTMultiviewDataset.process_one_sample(idx_in_p, self.persons[p], self.as_dataloader, self.rtgene_normalization)
        return sample

    def get_face_landmarks(self):
        return self.features_samples, self.headpose_samples

class UTMultiviewDatasetSinglePerson(torch.utils.data.Dataset):
    def __init__(self, person, as_dataloader=False, rtgene_normalization=False):
        self.as_dataloader = as_dataloader
        self.n = len(person["identifier"])
        self.person = person
        self.rtgene_normalization = rtgene_normalization

    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = UTMultiviewDataset.process_one_sample(idx, self.person, self.as_dataloader, self.rtgene_normalization)
        return sample

    def get_camera_matrices(self):
        return self.person["M"], np.zeros((4,))


class UTMultiviewFromSynthDataset(PersonDataset):

    @staticmethod
    def get_image(person_path, sample_idx, subsample_idx, folder, side):
        zfile = ZipFile(os.path.join(person_path, folder, str(sample_idx).zfill(3) + "_" + side + ".zip"))
        image_names = zfile.namelist()
        return Image.open(zfile.open(image_names[subsample_idx]))

    def __init__(self, path, eval=False, eval_as_last_person=False, **kw):

        self.path = path
        self.eval = eval

        # self.face_model = sio.loadmat(os.path.join(path, '6 points-based face model.mat'))["model"]

        # List person directories
        persons_path = sorted(glob(path + "/*/"))

        self.folder = "test" if eval else "synth"
        self.eval_as_last_person = eval_as_last_person
        
        self.n_per_p = []
        self.n = 0
        self.cumulative_n = []
        self.persons = []
        for p in persons_path:
            
            # samples = sorted(glob(os.join(p, "*_left.csv")))
            n_samples = len(glob(os.path.join(p, self.folder, "*_left.csv")))

            total_num_samples = 0

            person = {
                "left": {
                    "gaze_dirs": [],
                    "rot_vecs": [],
                    "trans_vecs": [],
                    "images": [],
                    "image_paths": []
                },
                "right": {
                    "gaze_dirs": [],
                    "rot_vecs": [],
                    "trans_vecs": [],
                    "images": [],
                    "image_paths": []
                }
            }

            for sample_number in range(n_samples):
                subsamples_per_samples = None
                for side in ["left", "right"]:
                    with open(os.path.join(p, self.folder, str(sample_number).zfill(3) + "_" + side + ".csv"), "r") as f:

                        lines = f.readlines()
                        subsamples_per_samples = len(lines) - len(lines)//12

                        for idx in range(len(lines)):
                            
                            left_right_idx = idx // 12
                            up_down_idx = idx - left_right_idx * 12
                            
                            if side == "right":
                                left_right_idx = 11 - left_right_idx

                            if left_right_idx % 12 == 11:
                                continue

                            actual_idx = up_down_idx + left_right_idx * 12
                            # actual_idx = idx

                            # offset = rotations.active_matrix_from_angle(1, np.radians(-3.0) if side == "left" else np.radians(3.0))[:3,:3]

                            l = lines[actual_idx]
                            d = np.fromstring(l, sep=",")
                            person[side]["gaze_dirs"].append(d[:3])
                            person[side]["rot_vecs"].append(d[3:6])
                            # person[side]["rot_vecs"].append(cv2.Rodrigues(offset @ cv2.Rodrigues(d[3:6])[0])[0])
                            person[side]["trans_vecs"].append(d[6:9])
                            person[side]["image_paths"].append(
                                {
                                    "sample_idx": sample_number,
                                    "subsample_idx": idx,
                                    "person_path": p
                                }
                            )
                total_num_samples += subsamples_per_samples
                self.n += subsamples_per_samples
            # person["id"] = sample_number.zfill(3)
            self.n_per_p.append(total_num_samples)
            self.persons.append(person)
        self.np = len(self.persons)
        self.cumulative_n = np.cumsum(self.n_per_p)

        if eval_as_last_person:
            folder = "test"
            # redo it all
            person = {
                    "left": {
                        "gaze_dirs": [],
                        "rot_vecs": [],
                        "trans_vecs": [],
                        "images": [],
                        "image_paths": []
                    },
                    "right": {
                        "gaze_dirs": [],
                        "rot_vecs": [],
                        "trans_vecs": [],
                        "images": [],
                        "image_paths": []
                    }
                }
            total_num_samples = 0
            for p in persons_path:
            
                # samples = sorted(glob(os.join(p, "*_left.csv")))
                n_samples = len(glob(os.path.join(p, folder, "*_left.csv")))

                

                

                for sample_number in range(n_samples):
                    subsamples_per_samples = None
                    for side in ["left", "right"]:
                        with open(os.path.join(p, folder, str(sample_number).zfill(3) + "_" + side + ".csv"), "r") as f:
                            lines = f.readlines()
                            subsamples_per_samples = len(lines)
                            # data = lines_to_array(lines)

                            for idx, l in enumerate(lines):
                                d = np.fromstring(l, sep=",")
                                person[side]["gaze_dirs"].append(d[:3])
                                person[side]["rot_vecs"].append(d[3:6])
                                person[side]["trans_vecs"].append(d[6:9])
                                person[side]["image_paths"].append(
                                    {
                                        "sample_idx": sample_number,
                                        "subsample_idx": idx,
                                        "person_path": p
                                    }
                                )
                    total_num_samples += subsamples_per_samples
                    self.n += subsamples_per_samples
                # person["id"] = sample_number.zfill(3)
            self.n_per_p.append(total_num_samples)
            self.persons.append(person)
            self.np = len(self.persons)
            self.cumulative_n = np.cumsum(self.n_per_p)

    def get_number_of_persons(self):
        return self.np

    def get_persons_ids(self):
        return [p["id"] for p in self.persons]
    
    def get_person_dataset(self, p):
        return UTMultiviewFromSynthDatasetSinglePerson(self.persons[p], "test" if p == (len(self.persons) - 1) and self.eval_as_last_person else self.folder)

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

        
        person = self.persons[p]

        left_eye = person["left"]["trans_vecs"][idx_in_p]
        right_eye = person["right"]["trans_vecs"][idx_in_p]

        face = 0.5*(left_eye + right_eye)
        left_gaze = person["left"]["gaze_dirs"][idx_in_p]
        right_gaze = person["right"]["gaze_dirs"][idx_in_p]
        # left_image = person["left"]["images"][idx_in_p]
        # right_image = person["right"]["images"][idx_in_p]
        left_headpose = person["left"]["rot_vecs"][idx_in_p]
        right_headpose = person["right"]["rot_vecs"][idx_in_p]
        left_head_yaw_pitch = vector_to_yaw_pitch_head(cv2.Rodrigues(person["left"]["rot_vecs"][idx_in_p])[0][:3,2])
        right_head_yaw_pitch = vector_to_yaw_pitch_head(cv2.Rodrigues(person["right"]["rot_vecs"][idx_in_p])[0][:3,2])

        left_image = UTMultiviewFromSynthDataset.get_image(**person["left"]["image_paths"][idx_in_p], folder=self.folder, side="left")
        right_image = UTMultiviewFromSynthDataset.get_image(**person["right"]["image_paths"][idx_in_p], folder=self.folder, side="right")

        out = {
            # "target":target/1000, 
            # "face": person["faces"][idx]/1000,
            "image_l": np.array(left_image.convert("RGB"), copy=False).copy(),
            "image_r": np.array(right_image.convert("RGB"), copy=False).copy(),
            "right_headpose": right_headpose,
            "left_headpose": left_headpose,
            # "hr": person["head_rs"][idx],
            # "ht": person["head_ts"][idx],
            "right_gaze": right_gaze,
            "left_gaze": left_gaze,
            "left_yaw_pitch": vector_to_yaw_pitch_gaze(left_gaze),
            "right_yaw_pitch": vector_to_yaw_pitch_gaze(right_gaze),
            "left_head_yaw_pitch": left_head_yaw_pitch,
            "right_head_yaw_pitch": right_head_yaw_pitch,
        }
        return out

class UTMultiviewFromSynthDatasetSinglePerson(torch.utils.data.Dataset):
    def __init__(self, person, folder):
        # self.eval = parent_dataset.eval
        # self.folder = parent_dataset.folder
        self.folder = folder
        self.person = person
        self.n = len(person["left"]["image_paths"])

    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        person = self.person

        left_eye = person["left"]["trans_vecs"][idx]
        right_eye = person["right"]["trans_vecs"][idx]

        face = 0.5*(left_eye + right_eye)
        left_gaze = person["left"]["gaze_dirs"][idx]
        right_gaze = person["right"]["gaze_dirs"][idx]
        # left_image = person["left"]["images"][idx]
        # right_image = person["right"]["images"][idx]
        left_headpose = person["left"]["rot_vecs"][idx]
        right_headpose = person["right"]["rot_vecs"][idx]
        left_head_yaw_pitch = vector_to_yaw_pitch_head(cv2.Rodrigues(person["left"]["rot_vecs"][idx])[0][:3,2])
        right_head_yaw_pitch = vector_to_yaw_pitch_head(cv2.Rodrigues(person["right"]["rot_vecs"][idx])[0][:3,2])

        left_image = UTMultiviewFromSynthDataset.get_image(**person["left"]["image_paths"][idx], folder=self.folder, side="left")
        right_image = UTMultiviewFromSynthDataset.get_image(**person["right"]["image_paths"][idx], folder=self.folder, side="right")

        out = {
            # "target":target/1000, 
            # "face": person["faces"][idx]/1000,
            "image_l": np.array(left_image.convert("RGB"), copy=False).copy(),
            "image_r": np.array(right_image.convert("RGB"), copy=False).copy(),
            "right_headpose": right_headpose,
            "left_headpose": left_headpose,
            # "hr": person["head_rs"][idx],
            # "ht": person["head_ts"][idx],
            "right_gaze": right_gaze,
            "left_gaze": left_gaze,
            "left_yaw_pitch": vector_to_yaw_pitch_gaze(left_gaze),
            "right_yaw_pitch": vector_to_yaw_pitch_gaze(right_gaze),
            "left_head_yaw_pitch": left_head_yaw_pitch,
            "right_head_yaw_pitch": right_head_yaw_pitch,
        }
        return out

    def get_camera_matrices(self):
        return self.person["M"], np.zeros((4,))