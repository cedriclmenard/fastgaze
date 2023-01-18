import torch
from pathlib import Path
from PIL import Image
import torch
from bisect import bisect_left
import numpy as np
import h5py
import cv2
import torchvision.transforms.functional as TF
import pickle


from ..utilities.general import progress_bar
from ..utilities.geometry import Rectangle, vector_to_yaw_pitch, compute_yaw_pitch, vector_to_yaw_pitch_gaze, vector_to_yaw_pitch_head
from ..face import default_face_model_path
from .helpers import normalize_img, PersonDataset


def compute_face_position_from_supplement(head_pose, face_model_3d_coordinates):
    # Calculate rotation matrix and euler angles
    rvec = head_pose[:3].reshape(3,1)
    tvec = head_pose[3:].reshape(3,1)
    rotate_mat, _ = cv2.Rodrigues(rvec)

    # Take mean face model landmarks and get transformed 3D positions
    landmarks_3d = np.matmul(rotate_mat, face_model_3d_coordinates.T).T
    landmarks_3d += tvec.T

    # Gaze-origin (g_o) and target (g_t)
    g_o = np.mean(landmarks_3d[10:12, :], axis=0)  # between 2 eyes
    g_o = g_o.squeeze()/1000.0
    return g_o, landmarks_3d

# Supplementary file from FAZE: https://github.com/swook/faze_preprocess/blob/5c33caaa1bc271a8d6aad21837e334108f293683/grab_prerequisites.bash
class GazeCaptureDataset(PersonDataset):
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
    def compute_one_sample(person, idx, face_model_3d_coordinates, as_dataloader=False, square=False, square_size=None, more_annotations={}, undistort=False, normalization_pose=None, skip_cropping_eyes=False):
        # try:
        img_path = person["base_path"] + "/" + person["supplementary"]["file_name"][idx].decode("UTF-8")
        # img = np.array(imageio.imread(img_path))
        img = Image.open(img_path)
        
        # Build camera matrix
        fx, fy, cx, cy = person["supplementary"]["camera_parameters"][idx, :]
        M = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        D = np.append(person["supplementary"]["distortion_parameters"][idx,:], 0) # Add one of the optional parameters back to work with MPIIFaceGaze

        if normalization_pose:
            if undistort:
                if not np.array_equal(GazeCaptureDataset.cached_M, M) or not np.array_equal(GazeCaptureDataset.cached_D, D):
                    M_new, roi = cv2.getOptimalNewCameraMatrix(M, D, (img.height, img.width), alpha=0)
                    GazeCaptureDataset.cached_map1, GazeCaptureDataset.cached_map2 = cv2.initUndistortRectifyMap(M, D, None, M_new, (img.width, img.height), cv2.CV_32FC1)
                    M = M_new
                img = Image.fromarray(cv2.remap(np.array(img), GazeCaptureDataset.cached_map1, GazeCaptureDataset.cached_map2, interpolation=cv2.INTER_LINEAR))
            im_path = Path(img_path)
            im_id = str(Path(im_path.parent.parent.name, im_path.parent.name, im_path.name))
            pose = normalization_pose[im_id]
            target = person["supplementary"]["3d_gaze_target"][idx, :]/1000.0

            image_r, right_headpose, right_gaze = normalize_img(np.array(img), pose["right_eye_center"], pose["hR"], target*1000, (64,64), M)
            image_l, left_headpose, left_gaze = normalize_img(np.array(img), pose["left_eye_center"], pose["hR"], target*1000, (64,64), M)
            yaw_pitch_r = vector_to_yaw_pitch_gaze(-right_gaze)
            yaw_pitch_l = vector_to_yaw_pitch_gaze(-left_gaze)
            head_yaw_pitch_r = vector_to_yaw_pitch_head(cv2.Rodrigues(right_headpose)[0][:,2])
            head_yaw_pitch_l = vector_to_yaw_pitch_head(cv2.Rodrigues(left_headpose)[0][:,2])
            yaw_pitch = (yaw_pitch_r + yaw_pitch_l) / 2.0
            head_yaw_pitch = vector_to_yaw_pitch_head(pose["hR"][:,2])

            real_yaw_pitch = vector_to_yaw_pitch_gaze(target - pose["face"]/1000)
            out = {
                    "M": M, 
                    "D": D.squeeze(), 
                    "target":target, 
                    "face": pose["face"]/1000,
                    # "real_face": pose["face"]/1000,
                    "image_path": im_id,
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
                    "right_eye_position": pose["right_eye_center"]/1000,
                    "left_eye_position": pose["left_eye_center"]/1000,
                    "left_head_yaw_pitch": head_yaw_pitch_l,
                    "right_head_yaw_pitch": head_yaw_pitch_r,
                    "head_yaw_pitch": head_yaw_pitch,
                    "hr": pose["hR"]
                }
            if not as_dataloader:
                out["image"] = np.array(img)
            return out
        else:
            if undistort:
                if not np.array_equal(GazeCaptureDataset.cached_M, M) or not np.array_equal(GazeCaptureDataset.cached_D, D):
                    M_new, roi = cv2.getOptimalNewCameraMatrix(M, D, (img.height, img.width), alpha=0)
                    GazeCaptureDataset.cached_map1, GazeCaptureDataset.cached_map2 = cv2.initUndistortRectifyMap(M, D, None, M_new, (img.width, img.height), cv2.CV_32FC1)
                    M = M_new
                # img = Image.fromarray(cv2.undistort(np.array(img), M, D))
                img = Image.fromarray(cv2.remap(np.array(img), GazeCaptureDataset.cached_map1, GazeCaptureDataset.cached_map2, interpolation=cv2.INTER_LINEAR))

            # Compute face location used by FAZE
            face, landmarks_3d = compute_face_position_from_supplement(person["supplementary"]["head_pose"][idx, :], face_model_3d_coordinates)
            landmarks_3d_cam = (M @ landmarks_3d.T).T
            landmarks_2d = landmarks_3d_cam[:,:2] / landmarks_3d_cam[:,2,None]

            yaw_pitch = compute_yaw_pitch(face, person["supplementary"]["3d_gaze_target"][idx, :]/1000.0)
            gaze_direction_mirror = person["supplementary"]["3d_gaze_target"][idx, :]/1000.0 - face
            gaze_direction_mirror[0] = -gaze_direction_mirror[0]
            yaw_pitch_mirror = vector_to_yaw_pitch_gaze(gaze_direction_mirror)
            
            if square:
                image, dx, dy, s = GazeCaptureDataset.crop_to_square(np.array(img), landmarks_2d[9:13, :].mean(axis=0), square_size)
                # adjust camera matrix
                M[0,2] -= dx
                M[1,2] -= dy
                # landmarks_2d[:, 0] -= dx
                # landmarks_2d[:, 1] -= dy
                if s != 1.0:
                    M[:2, :3] *= s
                    # landmarks_2d *= s
                    
            else:
                dx = 0
                dy = 0
                s = 1.0
                image = np.array(img)

            if not skip_cropping_eyes:
                # Compute image_l and image_r
                a = more_annotations.get(str(Path(Path(img_path).parent.parent.name, Path(img_path).parent.name, Path(img_path).name)))
                if a is None:
                    left_eye_width = (landmarks_2d[12,0] - landmarks_2d[11,0])*3
                    right_eye_width = (landmarks_2d[10,0] - landmarks_2d[9,0])*3
                    left_eye_center = (landmarks_2d[12,:] + landmarks_2d[11,:])/2.0
                    right_eye_center = (landmarks_2d[10,:] + landmarks_2d[9,:])/2.0
                    left_eye_lt = left_eye_center - left_eye_width/2
                    right_eye_lt = right_eye_center - right_eye_width/2
                else:
                    left_rect = Rectangle.from_dict(a["left_eye"])
                    right_rect = Rectangle.from_dict(a["right_eye"])
                    left_eye_lt = left_rect.lt_i()
                    right_eye_lt = right_rect.lt_i()
                    left_eye_width = left_rect.width_i()
                    right_eye_width = right_rect.width_i()

                if undistort:
                        left_eye_lt = np.array(left_eye_lt, dtype=np.float)
                        left_eye_lt = cv2.undistortPoints(left_eye_lt, M, D, P=M).squeeze()

                        right_eye_lt = np.array(right_eye_lt, dtype=np.float)
                        right_eye_lt = cv2.undistortPoints(right_eye_lt, M, D, P=M).squeeze()
                
                image_l = TF.resized_crop(img, left_eye_lt[1], left_eye_lt[0], left_eye_width, left_eye_width, (64,64))
                image_r = TF.resized_crop(img, right_eye_lt[1], right_eye_lt[0], right_eye_width, right_eye_width, (64,64))

            

            if as_dataloader:
                sample = {
                    "target": person["supplementary"]["3d_gaze_target"][idx, :]/1000.0, 
                    "M": M, "D": D, 
                    "face": face, 
                    "image_path": img_path,
                    "image_l": np.array(image_l) if not skip_cropping_eyes else None,
                    "image_r": np.array(image_r) if not skip_cropping_eyes else None,
                    "yaw_pitch": yaw_pitch.astype(np.float32),
                    "yaw_pitch_mirror": yaw_pitch_mirror.astype(np.float32)
                }
            else:
                sample = {
                    "image": np.array(image), 
                    "target": person["supplementary"]["3d_gaze_target"][idx, :]/1000.0, 
                    "M": M, "D": D, 
                    "face": face, 
                    "image_path": img_path,
                    "image_l": np.array(image_l) if not skip_cropping_eyes else None,
                    "image_r": np.array(image_r) if not skip_cropping_eyes else None,
                    "yaw_pitch": yaw_pitch.astype(np.float32),
                    "yaw_pitch_mirror": yaw_pitch_mirror.astype(np.float32),
                    "x_offset": dx,
                    "y_offset": dy,
                    "scale_offset": s
                }
            return sample


    def __init__(self, path, as_dataloader=False, square=False, square_size=None, use_more_annotations=True, undistort=False, custom_normalization_path="", skip_cropping_eyes=False):
        self.path = path
        self.as_dataloader = as_dataloader
        self.square = square
        self.square_size = square_size
        self.undistort = undistort
        self.skip_cropping_eyes = skip_cropping_eyes

        # Load supplementary data
        sup_path = Path(self.path + "/GazeCapture_supplementary.h5")
        assert sup_path.is_file(), "Missing supplementary file, download from: https://ait.ethz.ch/projects/2019/faze/downloads/preprocessing/GazeCapture_supplementary.h5"
        # self.supp_data = h5py.File(sup_path, "r")
        self.supp_data = {}
        h5file = h5py.File(sup_path, "r")
        for k in h5file.keys():
            data = {}
            for j in h5file[k].keys():
                data[j] = h5file[k][j][:]
            self.supp_data[k] = data

        # Get more annotations if available
        a_path = Path(Path(__file__).parent.absolute(), "more_annotations", "gazecapture_annotations.pkl")
        if use_more_annotations and a_path.exists():
            self.more_annotations = pickle.load(open(a_path, "rb"))
        else:
            self.more_annotations = {}
        

        # Load supplementary facial landmarks
        self.face_model_3d_coordinates = np.load(self.path + "/sfm_face_coordinates.npy")

        if custom_normalization_path:
            self.face_model = np.empty((468,3), dtype=float)
            with open(default_face_model_path, "r") as f:
                for i in range(468):
                    self.face_model[i, :] = np.array(f.readline().split(" ")[1:], dtype=float)
            self.pose_data = pickle.load(open(custom_normalization_path, "rb"))
        else:
            self.pose_data = None

        # Using supplementary data, not all dirs/persons are available
        self.dirs = [Path(self.path + "/" + str(p)) for p in self.supp_data.keys()]

        self.np = int(len(self.dirs))
        
        self.samples_per_p = []
        self.n_per_p = []
        self.cumulative_n = []
        self.n = 0

        self.persons = []
        for d in self.dirs:
            supp_data_p = self.supp_data[d.stem]
            if self.pose_data:
                gaze_targets = []
                camera_parameters = []
                distortion_parameters = []
                file_names = []
                head_poses = []
                for i, f in enumerate(supp_data_p["file_name"]):
                    if f.decode('UTF-8') in self.pose_data:
                        gaze_targets.append(supp_data_p["3d_gaze_target"][i,:])
                        camera_parameters.append(supp_data_p["camera_parameters"][i,:])
                        distortion_parameters.append(supp_data_p["distortion_parameters"][i,:])
                        file_names.append(f)
                        head_poses.append(supp_data_p["head_pose"][i,:])
                supp_data_p = {
                    "file_name": np.array(file_names),
                    "3d_gaze_target": np.array(gaze_targets),
                    "camera_parameters": np.array(camera_parameters),
                    "distortion_parameters": np.array(distortion_parameters),
                    "head_pose": np.array(head_poses)
                }
            
            person = {"base_path": self.path, "dir": str(d), "supplementary": supp_data_p, "id": Path(d).name}
            self.persons.append(person)
            self.n_per_p.append(len(person["supplementary"]["file_name"]))
            self.n += len(person["supplementary"]["file_name"])
            self.cumulative_n.append(self.n)
            self.samples_per_p.append(person["supplementary"]["file_name"])
            
    def get_number_of_persons(self):
        return self.np

    def get_persons_ids(self):
        return [p["id"] for p in self.persons]
    
    def get_person_dataset(self, p):
        return GazeCaptureDatasetSinglePerson(self.persons[p], self.face_model_3d_coordinates, self.as_dataloader, self.square, self.square_size, self.more_annotations, self.undistort, self.pose_data, self.skip_cropping_eyes)

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
        return GazeCaptureDataset.compute_one_sample(person, idx_in_p, self.face_model_3d_coordinates, self.as_dataloader, self.square, self.square_size, self.more_annotations, self.undistort, self.pose_data, self.skip_cropping_eyes)

    def export_to_hdf5(self, path):
        # sample = {
        #         "target": person["supplementary"]["3d_gaze_target"][idx, :]/1000.0, 
        #         "M": M, "D": D, 
        #         "face": face, 
        #         "image_path": img_path,
        #         "image_l": np.array(image_l),
        #         "image_r": np.array(image_r),
        #         "yaw_pitch": yaw_pitch.astype(np.float32)
        #     }
        f = h5py.File(path, "w")
        ds = f.create_group("gazecapture")
        ds.attrs["face_model_3d_coordinates"] = self.face_model_3d_coordinates # TODO: update dataset with this
        for pidx, p in enumerate(progress_bar(self.persons, prefix="Progress:", suffix="Complete", length=50)):
            g = ds.create_group(Path(p["dir"]).name)
            data = [GazeCaptureDataset.compute_one_sample(p, i, self.face_model_3d_coordinates, as_dataloader=True) for i in range(self.n_per_p[pidx])]
            g.create_dataset("target", data=np.stack([d["target"] for d in data]))
            g.create_dataset("M", data=np.stack([d["M"] for d in data]))
            g.create_dataset("D", data=np.stack([d["D"] for d in data]))
            g.create_dataset("face", data=np.stack([d["face"] for d in data]))
            g.create_dataset("image_path", data=np.array([Path(d["image_path"]).name for d in data], dtype="S"))
            g.create_dataset("image_l", data=np.stack([d["image_l"] for d in data]))
            g.create_dataset("image_r", data=np.stack([d["image_r"] for d in data]))
            g.create_dataset("yaw_pitch", data=np.stack([d["yaw_pitch"] for d in data]))



class GazeCaptureDatasetSinglePerson(torch.utils.data.Dataset):
    def __init__(self, person, face_model_3d_coordinates, as_dataloader=False, square=False, square_size=None, more_annotations={}, undistort=False, normalization_pose=None, skip_cropping_eyes=False):
        self.square = square
        self.square_size = square_size
        self.as_dataloader = as_dataloader
        self.face_model_3d_coordinates = face_model_3d_coordinates
        self.n = len(person["supplementary"]["file_name"])
        self.person = person
        self.more_annotations = more_annotations
        self.undistort = undistort
        self.normalization_pose = normalization_pose
        self.skip_cropping_eyes = skip_cropping_eyes

    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        person = self.person

        return GazeCaptureDataset.compute_one_sample(person, idx, self.face_model_3d_coordinates, self.as_dataloader, self.square, self.square_size, self.more_annotations, self.undistort, self.normalization_pose, self.skip_cropping_eyes)


class GazeCaptureFazeDataset(GazeCaptureDataset):
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
    def normalize_sample(image, supplementary_data, i, face_model_landmarks, side_left):

        normalized_camera = {
            'focal_length': 650,
            'distance': 600,
            'size': (64, 64),
        }

        norm_camera_matrix = np.array(
            [
                [normalized_camera['focal_length'], 0, 0.5*normalized_camera['size'][0]],  # noqa
                [0, normalized_camera['focal_length'], 0.5*normalized_camera['size'][1]],  # noqa
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
         # Form original camera matrix
        fx, fy, cx, cy = supplementary_data['camera_parameters'][i, :]
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                                dtype=np.float64)

        # Calculate rotation matrix and euler angles
        rvec = supplementary_data['head_pose'][i, :3].reshape(3, 1)
        tvec = supplementary_data['head_pose'][i, 3:].reshape(3, 1)
        rotate_mat, _ = cv2.Rodrigues(rvec)

        # Take mean face model landmarks and get transformed 3D positions
        landmarks_3d = np.matmul(rotate_mat, face_model_landmarks.T).T
        landmarks_3d += tvec.T

        # Gaze-origin (g_o) and target (g_t)
        if side_left:
            g_o = np.mean(landmarks_3d[11:13, :], axis=0)
        else:
            g_o = np.mean(landmarks_3d[9:11, :], axis=0)
        g_o = g_o.reshape(3, 1)
        g_t = supplementary_data['3d_gaze_target'][i, :].reshape(3, 1)
        g = g_t - g_o
        g /= np.linalg.norm(g)

        # Code below is an adaptation of code by Xucong Zhang
        # https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/revisiting-data-normalization-for-appearance-based-gaze-estimation/

        # actual distance between gaze origin and original camera
        distance = np.linalg.norm(g_o)
        z_scale = normalized_camera['distance'] / distance
        S = np.eye(3, dtype=np.float64)
        S[2, 2] = z_scale

        hRx = rotate_mat[:, 0]
        forward = (g_o / distance).reshape(3)
        down = np.cross(forward, hRx)
        down /= np.linalg.norm(down)
        right = np.cross(down, forward)
        right /= np.linalg.norm(right)
        R = np.c_[right, down, forward].T  # rotation matrix R

        # transformation matrix
        W = np.dot(np.dot(norm_camera_matrix, S),
                np.dot(R, np.linalg.inv(camera_matrix)))

        ow, oh = normalized_camera['size']
        patch = cv2.warpPerspective(np.array(image), W, (ow, oh))  # image normalization

        R = np.asmatrix(R)

        # Correct head pose
        # h = np.array([np.arcsin(rotate_mat[1, 2]),
        #             np.arctan2(rotate_mat[0, 2], rotate_mat[2, 2])])
        head_mat = R * rotate_mat
        # n_h = np.array([np.arcsin(head_mat[1, 2]),
        #                 np.arctan2(head_mat[0, 2], head_mat[2, 2])])

        # Correct gaze
        n_g = R * g
        n_g /= np.linalg.norm(n_g)

        return patch, cv2.Rodrigues(head_mat)[0].reshape((3,)), np.array(n_g).squeeze(), g_o, g_t, rotate_mat

    
    @staticmethod
    def compute_one_sample(person, idx, face_model_3d_coordinates, as_dataloader=False):
        # try:
        img_path = person["base_path"] + "/" + person["supplementary"]["file_name"][idx].decode("UTF-8")
        # img = np.array(imageio.imread(img_path))
        img = Image.open(img_path)
        
        # Build camera matrix
        fx, fy, cx, cy = person["supplementary"]["camera_parameters"][idx, :]
        M = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        D = np.append(person["supplementary"]["distortion_parameters"][idx,:], 0) # Add one of the optional parameters back to work with MPIIFaceGaze

        if not np.array_equal(GazeCaptureDataset.cached_M, M) or not np.array_equal(GazeCaptureDataset.cached_D, D):
            # M_new, roi = cv2.getOptimalNewCameraMatrix(M, D, (img.height, img.width), alpha=0)
            # GazeCaptureDataset.cached_map1, GazeCaptureDataset.cached_map2 = cv2.initUndistortRectifyMap(M, D, None, M_new, (img.width, img.height), cv2.CV_32FC1)
            # M = M_new
            GazeCaptureDataset.cached_map1, GazeCaptureDataset.cached_map2 = cv2.initUndistortRectifyMap(M, D, None, M, (img.width, img.height), cv2.CV_32FC1)
        img = Image.fromarray(cv2.remap(np.array(img), GazeCaptureDataset.cached_map1, GazeCaptureDataset.cached_map2, interpolation=cv2.INTER_LINEAR))
        im_path = Path(img_path)
        im_id = str(Path(im_path.parent.parent.name, im_path.parent.name, im_path.name))



        image_r, right_headpose, right_gaze, g_o_r, g_t, headpose = GazeCaptureFazeDataset.normalize_sample(img, person["supplementary"], idx, face_model_3d_coordinates, side_left=False)
        image_l, left_headpose, left_gaze, g_o_l, g_t, headpose = GazeCaptureFazeDataset.normalize_sample(img, person["supplementary"], idx, face_model_3d_coordinates, side_left=True)
        yaw_pitch_r = vector_to_yaw_pitch_gaze(right_gaze)
        yaw_pitch_l = vector_to_yaw_pitch_gaze(left_gaze)
        head_yaw_pitch_r = vector_to_yaw_pitch_head(cv2.Rodrigues(right_headpose)[0][:,2])
        head_yaw_pitch_l = vector_to_yaw_pitch_head(cv2.Rodrigues(left_headpose)[0][:,2])
        yaw_pitch = (yaw_pitch_r + yaw_pitch_l) / 2.0

        out = {
                "M": M, 
                "D": D.squeeze(), 
                # "target":target, 
                # "face": pose["face"]/1000,
                # "real_face": pose["face"]/1000,
                "face": (g_o_r + g_o_l)/1000.0,
                "target": g_t/1000.0,
                "hr": headpose,
                "image_path": im_id,
                "image_l": np.array(image_l, copy=False).copy(),
                "image_r": np.array(image_r, copy=False).copy(),
                "yaw_pitch": yaw_pitch.astype(np.float32),
                # "real_yaw_pitch": real_yaw_pitch.astype(np.float32),
                "right_headpose": right_headpose,
                "left_headpose": left_headpose,
                "right_gaze": right_gaze,
                "left_gaze": left_gaze,
                "left_yaw_pitch": vector_to_yaw_pitch_gaze(left_gaze),
                "right_yaw_pitch": vector_to_yaw_pitch_gaze(right_gaze),
                "right_eye_position": g_o_r/1000.0,
                "left_eye_position": g_o_l/1000.0,
                "left_head_yaw_pitch": head_yaw_pitch_l,
                "right_head_yaw_pitch": head_yaw_pitch_r,
                # "head_yaw_pitch": head_yaw_pitch
            }
        if not as_dataloader:
            out["image"] = np.array(img)
        return out
        


    def __init__(self, path, as_dataloader=False):
        self.path = path
        self.as_dataloader = as_dataloader
        # Load supplementary data
        sup_path = Path(self.path + "/GazeCapture_supplementary.h5")
        assert sup_path.is_file(), "Missing supplementary file, download from: https://ait.ethz.ch/projects/2019/faze/downloads/preprocessing/GazeCapture_supplementary.h5"
        # self.supp_data = h5py.File(sup_path, "r")
        self.supp_data = {}
        h5file = h5py.File(sup_path, "r")
        for k in h5file.keys():
            data = {}
            for j in h5file[k].keys():
                data[j] = h5file[k][j][:]
            self.supp_data[k] = data
        

        # Load supplementary facial landmarks
        self.face_model_3d_coordinates = np.load(self.path + "/sfm_face_coordinates.npy")

        # Using supplementary data, not all dirs/persons are available
        self.dirs = [Path(self.path + "/" + str(p)) for p in self.supp_data.keys()]

        self.np = int(len(self.dirs))
        
        self.samples_per_p = []
        self.n_per_p = []
        self.cumulative_n = []
        self.n = 0

        self.persons = []
        for d in self.dirs:
            supp_data_p = self.supp_data[d.stem]
            
            person = {"base_path": self.path, "dir": str(d), "supplementary": supp_data_p, "id": Path(d).name}
            self.persons.append(person)
            self.n_per_p.append(len(person["supplementary"]["file_name"]))
            self.n += len(person["supplementary"]["file_name"])
            self.cumulative_n.append(self.n)
            self.samples_per_p.append(person["supplementary"]["file_name"])
            
    def get_number_of_persons(self):
        return self.np

    def get_persons_ids(self):
        return [p["id"] for p in self.persons]
    
    def get_person_dataset(self, p):
        return GazeCaptureFazeDatasetSinglePerson(self.persons[p], self.face_model_3d_coordinates, self.as_dataloader)

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
        return GazeCaptureFazeDataset.compute_one_sample(person, idx_in_p, self.face_model_3d_coordinates, self.as_dataloader)



class GazeCaptureFazeDatasetSinglePerson(torch.utils.data.Dataset):
    def __init__(self, person, face_model_3d_coordinates, as_dataloader=False):
        self.as_dataloader = as_dataloader
        self.face_model_3d_coordinates = face_model_3d_coordinates
        self.n = len(person["supplementary"]["file_name"])
        self.person = person

    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        person = self.person

        return GazeCaptureFazeDataset.compute_one_sample(person, idx, self.face_model_3d_coordinates, self.as_dataloader)