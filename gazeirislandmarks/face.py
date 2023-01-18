import numpy as np
from pathlib import Path
import os.path

import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import transformations as tr
import cv2

from .models import BlazeFace, FaceMesh
from .utilities.geometry import Rectangle, cosine_similarity_angle
from .utilities.image import get_width_height, square_center_crop_resize
from .utilities.procrustes import Procrustes

default_face_model_path = str(Path(os.path.dirname(__file__), "..", "data", "canonical_face_model.obj").resolve())

def compute_roll_from_blazeface(blazeface_output):
    right_eye = blazeface_output[4:6]
    left_eye = blazeface_output[6:8]
    right_ear = blazeface_output[12:14]
    left_ear = blazeface_output[14:16]

    # get roll angle from eyes and ears and use average
    eye_v = left_eye - right_eye
    ear_v = left_ear - right_ear

    eye_angle = -np.sign(eye_v[1]) * cosine_similarity_angle(eye_v, np.array([1,0]))[0]
    ear_angle = -np.sign(eye_v[1]) * cosine_similarity_angle(ear_v, np.array([1,0]))[0]
    roll = (ear_angle + eye_angle)/2.0
    return roll

class FaceNotDetectedError(Exception):
    pass

class FaceDetectAndAlign():
    default_face_model_path = default_face_model_path

    right_eye_lower_indices = [263, 249, 390, 373, 374, 380, 381, 382, 362]
    right_eye_upper_indices = [466, 388, 387, 386, 385, 384, 398]
    right_eye_indices = [7, 33, 133,144,145,153,154,155,157,158,159,160,161,163,173,246]
    left_eye_lower_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133]
    left_eye_upper_indices = [246, 161, 160, 159, 158, 157, 173]
    left_eye_indices = [249,263,362,373,374,380,381,382,384,385,386,387,388,390,398,466]
    mouth_center_indices = [0,11,12,13,14,15,16,17]
    upper_lip_top_indices = [57,185,40,39,37,267,269,270,409,287]
    lower_lip_bottom_indices = [57,146,91,181,84,17,314,405,321,375,287]

    def _crop_eye_regions(self, image, fl):
        left_eye_center = np.mean(fl[self.left_eye_indices, :], axis=0)
        right_eye_center = np.mean(fl[self.right_eye_indices, :], axis=0)
        left_eye_width = np.max(fl[self.left_eye_indices, 0]) - np.min(fl[self.left_eye_indices, 0]) * 1.5
        right_eye_width = np.max(fl[self.right_eye_indices, 0]) - np.min(fl[self.right_eye_indices, 0]) * 1.5
        
        left_im = TF.resized_crop(image, top=left_eye_center[1] - left_eye_width/2, left=left_eye_center[0] - left_eye_width/2, height=left_eye_width, width=left_eye_width, size=(64,64))
        right_im = TF.resized_crop(image, top=right_eye_center[1] - right_eye_width/2, left=right_eye_center[0] - right_eye_width/2, height=right_eye_width, width=right_eye_width, size=(64,64))

        return {
            "left_eye": {
                "image": left_im,
                "scale": left_eye_width/64.0,
                "top": left_eye_center[1] - left_eye_width/2,
                "left": left_eye_center[0] - left_eye_width/2
            },
            "right_eye": {
                "image": right_im,
                "scale": right_eye_width/64.0,
                "top": right_eye_center[1] - right_eye_width/2,
                "left": right_eye_center[0] - right_eye_width/2
            }
        }



    def __init__(self, device="cuda", face_detection_threshold=0.1, face_mesh_threshold=0.1):
        self.face_mesh_threshold = face_mesh_threshold
        self.face_detector = BlazeFace(min_score=face_detection_threshold).to(device)
        self.face_detector.load_weights(BlazeFace.default_weights_path)
        self.face_detector.load_anchors(BlazeFace.default_anchors_path)
        self.face_detector.eval()

        self.face_aligner = FaceMesh().to(device)
        self.face_aligner.load_weights(FaceMesh.default_weights_path)
        self.face_aligner.eval()

        self.device=device
        
        self.procrustes = None
        self.camera_matrix = None
        self.dist_coeff = np.zeros((4, 1))
        self.face_model = np.empty((468,3), dtype=float)
        with open(FaceDetectAndAlign.default_face_model_path, "r") as f:
            for i in range(468):
                self.face_model[i, :] = np.array(f.readline().split(" ")[1:], dtype=float)

        # self.face_detector = torch.jit.script(self.face_detector)
        # self.face_detector.to_jit_script()
        # self.face_detector = self.face_detector.half()
        # self.face_aligner = torch.jit.script(self.face_aligner)
        # self.face_aligner = self.face_aligner.half()


    def _detect_face(self, image):
        w, h = get_width_height(image)
        img_small, scale, x_offset, y_offset = square_center_crop_resize(image, square=128)


        face_detection = self.face_detector.predict_on_image(img_small).cpu().numpy()
        if face_detection.size == 0:
            return None
        face_detection[:][::2] = face_detection[:][::2]*128
        face_detection[:][1::2] = face_detection[:][1::2]*128

        h = int(face_detection[0][2] - face_detection[0][0])
        w = int(face_detection[0][3] - face_detection[0][1])

        ymin = int(face_detection[0][0])
        ymax = ymin + h
        xmin = int(face_detection[0][1])
        xmax = xmin + w
        r = Rectangle(ymin, xmin, ymax, xmax)
        r = r.scale(scale).translate(-x_offset, -y_offset)
        roll = compute_roll_from_blazeface(face_detection[0])
        return r, roll

    def _align_face(self, image, face_rectangle: Rectangle, roll=0.0):
        # r = face_rectangle.expand(0.5) # TODO: investigate why the mouth is misplaced if using the recommended 0.25
        r = face_rectangle.expand(0.35)
        w, h = get_width_height(image)
        rot_roll_full = tr.rotation_matrix(roll, [0,0,1])
        rot_roll = rot_roll_full[:3,:3]
        center = np.array([r.hcenter(), r.vcenter()]) - np.array([w/2, h/2])
        center = (rot_roll[:2,:2] @ center.T).T

        trans = tr.concatenate_matrices(
            tr.scale_matrix(192/r.width(), direction=np.array([0,0,1])),
            tr.scale_matrix(192/r.width(), direction=np.array([1,0,0])),
            tr.scale_matrix(192/r.height(), direction=np.array([0,1,0])),
            tr.translation_matrix(np.array([r.width()/2, r.height()/2,0])),
            rot_roll_full,
            tr.translation_matrix(np.array([-r.hcenter(), -r.vcenter(), 0])),
        )
        trans_1 = tr.inverse_matrix(trans)

        trans_torch = tr.concatenate_matrices(
            tr.translation_matrix((-1,-1,0)),
            tr.scale_matrix(2/192, direction=np.array([1,0,0])),
            tr.scale_matrix(2/192, direction=np.array([0,1,0])),
            trans,
            tr.scale_matrix(w/2, direction=np.array([1,0,0])),
            tr.scale_matrix(h/2, direction=np.array([0,1,0])),
            tr.translation_matrix((1,1,0))
        )
        trans_torch_1 = tr.inverse_matrix(trans_torch)
        [a,b],c,[d,e],f = trans_torch_1[0,:2], trans_torch_1[0,3], trans_torch_1[1,:2], trans_torch_1[1,3]
        affine_mat = torch.tensor([[[a,b,c],[d,e,f]]], dtype=torch.float32, device=self.device)
        grid = F.affine_grid(affine_mat, (1,3,192,192))
        imt = F.grid_sample(image.unsqueeze(0), grid, mode="bilinear", align_corners=False).squeeze()

        fl_device, confidence_device = self.face_aligner.predict_on_image(imt, output_confidence=True)
        fl = fl_device.cpu().numpy()
        confidence = confidence_device.cpu().numpy()
        if confidence < self.face_mesh_threshold:
            return (None,)*6
        uncorrected_fl = fl.copy()

        tmp = np.ones((uncorrected_fl.shape[0], 4))
        tmp[:, :3] = uncorrected_fl
        fl = (trans_1 @ tmp.T).T[:,:3]

        # return fl, im, uncorrected_fl, trans, trans_1
        return fl, imt, uncorrected_fl, trans, trans_1, confidence

    
    def detect(self, image, camera_matrix, dist_coeff=np.zeros((4, 1))):
        
        with torch.no_grad():
            if not torch.is_tensor(image):
                image = TF.to_tensor(image).to(self.device)
            w, h = get_width_height(image)
            if self.procrustes is None or self.camera_matrix is None or not np.allclose(camera_matrix, self.camera_matrix) or not np.allclose(dist_coeff, dist_coeff):
                self.camera_matrix = camera_matrix
                self.dist_coeff = dist_coeff
                self.procrustes = Procrustes(self.face_model, camera_matrix, dist_coeff, w, h)
            
            face_rects, rolls = self._detect_face(image)
            
            if face_rects.width() < 1.0 or face_rects.height() < 1.0:
                raise FaceNotDetectedError("No face detected")
            
            # face_mesh_corrected, im_192, face_mesh, trans, trans_1 = self._align_face(image, face_rects, rolls)
            face_mesh_corrected, im_192, face_mesh, trans, trans_1, confidence = self._align_face(image, face_rects, 0.0)
            if face_mesh_corrected is None:
                raise FaceNotDetectedError("Face could not be aligned")

            headpose_r, headpose_t = self.procrustes(face_mesh_corrected)

            
            return face_mesh_corrected, headpose_r, headpose_t, confidence

    def get_landmarks(self, image, camera_matrix, dist_coeff=np.zeros((4, 1))):
        with torch.no_grad():
            if not torch.is_tensor(image):
                image = TF.to_tensor(image).to(self.device)
            w, h = get_width_height(image)
            if self.procrustes is None or self.camera_matrix is None or not np.allclose(camera_matrix, self.camera_matrix) or not np.allclose(dist_coeff, dist_coeff):
                self.camera_matrix = camera_matrix
                self.dist_coeff = dist_coeff
                self.procrustes = Procrustes(self.face_model, camera_matrix, dist_coeff, w, h)
            
            face_rects, rolls = self._detect_face(image)
            
            if face_rects.width() < 1.0 or face_rects.height() < 1.0:
                raise FaceNotDetectedError("No face detected")
            
            face_mesh_corrected, im_192, face_mesh, trans, trans_1, confidence = self._align_face(image, face_rects, rolls)
            if face_mesh_corrected is None:
                raise FaceNotDetectedError("Face could not be aligned")

            # headpose_r, headpose_t = self.procrustes(face_mesh_corrected)
            face_mesh_metric, headpose_r, headpose_t = self.procrustes.compute_metric_landmarks(face_mesh_corrected)
            transform = np.eye(4)
            transform[:3,:3] = cv2.Rodrigues(headpose_r)[0]
            transform[:3,3] = headpose_t

            # hr = cv2.Rodrigues(headpose_r)[0]
            flip_y = np.eye(4)
            flip_y[1,1] = -1
            flip_z = np.eye(4)
            flip_z[2,2] = -1
            transform = transform @ (flip_z @ flip_y)
            face_mesh_metric = ((flip_z[:3,:3] @ flip_y[:3,:3]) @ face_mesh_metric.T).T

            
            return face_mesh_corrected, face_mesh_metric, transform

    @staticmethod
    def right_eye_center_from_facemesh(mesh):
        return np.mean(mesh[FaceDetectAndAlign.right_eye_indices, :], axis=0)

    @staticmethod
    def left_eye_center_from_facemesh(mesh):
        return np.mean(mesh[FaceDetectAndAlign.left_eye_indices, :], axis=0)

    def world_right_eye_center_from_canonical(self, headpose_r, headpose_t):
        hr, _ = cv2.Rodrigues(headpose_r)
        return np.mean((hr @ self.face_model[FaceDetectAndAlign.right_eye_indices,:].T).T + headpose_t.squeeze(), axis=0)

    def world_left_eye_center_from_canonical(self, headpose_r, headpose_t):
        hr, _ = cv2.Rodrigues(headpose_r)
        return np.mean((hr @ self.face_model[FaceDetectAndAlign.left_eye_indices,:].T).T + headpose_t.squeeze(), axis=0)
    
    def world_mouth_center_from_canonical(self, headpose_r, headpose_t):
        hr, _ = cv2.Rodrigues(headpose_r)
        return np.mean((hr @ self.face_model[FaceDetectAndAlign.mouth_center_indices,:].T).T + headpose_t.squeeze(), axis=0)

    def get_head_pose(self, image, camera_matrix, dist_coeff=np.zeros((4, 1))):

        with torch.no_grad():
            if not torch.is_tensor(image):
                image = TF.to_tensor(image).to(self.device)
            
            face_mesh_corrected, headpose_r, headpose_t, align_confidence = self.detect(image, camera_matrix, dist_coeff)

            right_eye = self.world_right_eye_center_from_canonical(headpose_r, headpose_t)
            left_eye = self.world_left_eye_center_from_canonical(headpose_r, headpose_t)
            mouth_center = self.world_mouth_center_from_canonical(headpose_r, headpose_t)

            hr = cv2.Rodrigues(headpose_r)[0]
            flip_y = np.eye(3)
            flip_y[1,1] = -1
            flip_z = np.eye(3)
            flip_z[2,2] = -1
            hr = hr @ (flip_z @ flip_y)

            
        
            pose_data = {
                "headpose_r": headpose_r.squeeze(),
                "headpose_t": headpose_t.squeeze(),
                "hR": hr,
                "right_eye_center": right_eye / 100, # in meters, from cm
                "left_eye_center": left_eye / 100,
                "face": 0.5 * (right_eye + left_eye) / 100,
                "mouth_center": mouth_center / 100,
                "align_confidence": align_confidence
            }
            return pose_data, face_mesh_corrected

    @staticmethod
    def show_landmarks_on_image(image: np.ndarray, face_mesh_corrected: np.ndarray):
        image = image.copy()
        for p in face_mesh_corrected:
                cv2.circle(image, p[:2].astype(int), 2, (255,0,0), -1)
        return image
