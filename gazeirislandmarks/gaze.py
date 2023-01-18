import os.path
import cv2
import numpy as np
import torch

from typing import (
    List,
)

import torchvision.transforms.functional as TF
import torchgeometry as tgm

from .models.gazeirislandmarks import load_model
from .face import FaceDetectAndAlign
from .utilities.geometry import yaw_pitch_to_vector, vector_to_yaw_pitch_gaze, vector_to_yaw_pitch_head
from .datasets.helpers import denormalize_gaze, denormalize_gaze_new

def normalize_image(image, gaze_origin, head_rotation, roi_size, cam_matrix, focal_new=650, distance_new=600, device="cuda"):
    distance = np.linalg.norm(gaze_origin)
    z_scale = distance_new / distance
    cam_new = np.array([[focal_new, 0, roi_size[0] / 2],
                        [0.0, focal_new, roi_size[1] / 2],
                        [0, 0, 1.0]])
    scale_mat = np.array([[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, z_scale]])
    h_rx = head_rotation[:, 0]
    forward = (gaze_origin / distance)
    down = np.cross(forward, h_rx)
    down = down / np.linalg.norm(down)
    right = np.cross(down, forward)
    right = right / np.linalg.norm(right)

    rot_mat = np.c_[right, down, forward].T
    warp_mat = (cam_new @ scale_mat) @ (rot_mat @ np.linalg.inv(cam_matrix))
    image_warped = tgm.warp_perspective(image.unsqueeze(0), torch.from_numpy(warp_mat).float().to(device), roi_size).squeeze()

    n_head_rotation = rot_mat @ head_rotation # head pose in the new normalized camera reference frame

    return image_warped, cv2.Rodrigues(n_head_rotation)[0].squeeze()

def denormalize_points(points, head_rot_mat, eye_position_3d_in_mm, roi_size, cam_matrix, focal_new=650, distance_new=600):
    distance = np.linalg.norm(eye_position_3d_in_mm)
    z_scale = distance_new / distance
    cam_new = np.array([[focal_new, 0, roi_size[0] / 2],
                        [0.0, focal_new, roi_size[1] / 2],
                        [0, 0, 1.0]])
    scale_mat = np.array([[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, z_scale]])
    h_rx = head_rot_mat[:, 0]
    forward = (eye_position_3d_in_mm / distance)
    down = np.cross(forward, h_rx)
    down = down / np.linalg.norm(down)
    right = np.cross(down, forward)
    right = right / np.linalg.norm(right)
    rot_mat = np.array([right.T, down.T, forward.T])

    warp_mat = (cam_new @ scale_mat) @ (rot_mat @ np.linalg.inv(cam_matrix))
    return cv2.perspectiveTransform(np.expand_dims(points, 1), np.linalg.inv(warp_mat)).squeeze()

class GazeDetector:
    default_model_path = os.path.join(os.path.dirname(__file__), "gazeirislandmarks.pth")

    def __init__(self, model_path=None, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), face_detection_threshold=0.1, face_mesh_threshold=0.1, halfprecision=False):
        self.device = device
        self.halfprecision = halfprecision
        if model_path is not None:
            if halfprecision:
                self.gaze_estimator = load_model(model_path, device).half()
            else:
                self.gaze_estimator = load_model(model_path, device)
            self.gaze_estimator.eval()
        else:
            if os.path.exists(GazeDetector.default_model_path):
                self.gaze_estimator = load_model(GazeDetector.default_model_path, device)
                self.gaze_estimator.eval()
            else:
                raise FileNotFoundError("No model path specified and default is not found: " + GazeDetector.default_model_path)
        
        self.face_detector = FaceDetectAndAlign(device, face_detection_threshold, face_mesh_threshold)
        self.gaze_estimator_config = self.gaze_estimator.config
        self.new_norm = self.gaze_estimator_config["new_norm"]
        

        if "utmv" in self.gaze_estimator_config["hdf5_name"]: # hacky but works for now since there isn't a separate flag for utmv
            print("Using UTMV offset")
            self.head_yaw_pitch_offset = np.radians(np.array([-0.0495545, 5.71939336]))
        else:
            self.head_yaw_pitch_offset = np.radians(np.array([0.0, 0.0]))
        
        pitch = self.head_yaw_pitch_offset[1]
        self.offset_head_rot = np.array([[1, 0, 0],
                                         [0, np.cos(pitch), -np.sin(pitch)], 
                                         [0, np.sin(pitch), np.cos(pitch)]])
        yaw = self.head_yaw_pitch_offset[0]
        self.offset_head_rot = self.offset_head_rot @ np.array([[np.cos(yaw), 0, np.sin(yaw)],
                                                                [0, 1, 0], 
                                                                [-np.sin(yaw), 0, np.cos(yaw)]])

        self.normalized_focal_length, self.normalized_distance = self.gaze_estimator.get_normalization_values()

        if True:
            torch.autocast("cuda" if torch.cuda.is_available() else "cpu")
            torch.backends.cudnn.benchmark = True
            self.gaze_estimator = torch.jit.script(self.gaze_estimator)

    def __call__(self, image, camera_matrix, dist_coeff=np.zeros((4, 1)), visualize_directions: bool = False, visualize_landmarks: bool = False):
        with torch.no_grad():
            if not torch.is_tensor(image):
                image = TF.to_tensor(image).to(self.device)

            pose_data, face_mesh_corrected = self.face_detector.get_head_pose(image, camera_matrix, dist_coeff)
            right_eye = pose_data["right_eye_center"]
            right_eye_mm = right_eye * 1000
            left_eye = pose_data["left_eye_center"]
            left_eye_mm = left_eye * 1000
            hr = pose_data["hR"]
            # hr = self.offset_head_rot @ pose_data["hR"] # seems to be the right one as it is working better this way
            headpose_r = pose_data["headpose_r"]
            headpose_t = pose_data["headpose_t"]

            right_eye_image, right_headpose = normalize_image(image, right_eye_mm, hr, (64,64), camera_matrix, device=self.device, focal_new=self.normalized_focal_length, distance_new=self.normalized_distance)
            head_yaw_pitch_r = vector_to_yaw_pitch_head(cv2.Rodrigues(right_headpose)[0][:,2]) + self.head_yaw_pitch_offset
            # head_yaw_pitch_r = vector_to_yaw_pitch_head(cv2.Rodrigues(right_headpose)[0][:,2])

            left_eye_image, left_headpose = normalize_image(image, left_eye_mm, hr, (64,64), camera_matrix, device=self.device, focal_new=self.normalized_focal_length, distance_new=self.normalized_distance)
            head_yaw_pitch_l = vector_to_yaw_pitch_head(cv2.Rodrigues(left_headpose)[0][:,2]) + self.head_yaw_pitch_offset
            # head_yaw_pitch_l = vector_to_yaw_pitch_head(cv2.Rodrigues(left_headpose)[0][:,2])

            if "face_distance" in self.gaze_estimator_config:
                head_yaw_pitch_r = np.append(head_yaw_pitch_r, np.linalg.norm(right_eye_mm)/1000)
                head_yaw_pitch_l = np.append(head_yaw_pitch_l, np.linalg.norm(left_eye_mm)/1000)

            outputs = self.gaze_estimator.forward(
                {
                    "right": right_eye_image.unsqueeze(0), 
                    "left": TF.hflip(left_eye_image).unsqueeze(0)
                }, 
                {
                    "left": torch.from_numpy(head_yaw_pitch_l).unsqueeze(0).float().to(self.device), 
                    "right": torch.from_numpy(head_yaw_pitch_r).unsqueeze(0).float().to(self.device)
                }
            )
            g = outputs[2].squeeze().cpu().numpy()
            if len(g) > 2:
                variance = g[2]
                g = g[:2]
                if self.gaze_estimator_config["type"] == "uncertainty_loss":
                    variance = np.exp(variance)
            else:
                variance = None
            if outputs[0] is not None:
                i = {k: v.squeeze().cpu().numpy() for k, v in outputs[1].items()}
                e = {k: v.squeeze().cpu().numpy() for k, v in outputs[0].items()}

            # convert the gaze vector to world
            gv = -yaw_pitch_to_vector(g)

            face = 0.5 * (right_eye_mm + left_eye_mm) / 10 # in cm
            # denorm_gv_l = denormalize_gaze(gv, hr, left_eye_mm)
            # denorm_gv_r = denormalize_gaze(gv, hr, right_eye_mm)
            # denorm_gv_l = cv2.Rodrigues(left_headpose)[0].T @ gv
            # denorm_gv_r = cv2.Rodrigues(right_headpose)[0].T @ gv
            # denorm_gv = 0.5 * (denorm_gv_l + denorm_gv_r)
            if self.new_norm:
                denorm_gv = denormalize_gaze_new(gv, 0.5 * (right_eye_mm + left_eye_mm), hr)
            else:
                denorm_gv = denormalize_gaze(gv, 0.5 * (right_eye_mm + left_eye_mm), hr, distance_new=self.normalized_distance)
            
            im_draw = None
            if visualize_directions:
                # Draw face direction
                line_begining = np.zeros((3,))
                line_end = np.array([0,0,5.0])
                line_begining = (camera_matrix @ ((hr @ line_begining.T) + headpose_t.T).T).T
                line_begining /= line_begining[...,2]
                line_end = (camera_matrix @ ((hr @ line_end.T) + headpose_t.T).T).T
                line_end /= line_end[...,2]

                im_draw = np.array(TF.to_pil_image(image))
                cv2.line(im_draw, line_begining.squeeze().astype(int)[:2], line_end.squeeze().astype(int)[:2], (0, 0, 255), 3)
                
                # Draw right eye direction
                line_start = right_eye_mm/1000
                line_end = line_start + 0.05*denorm_gv
                line_start = (camera_matrix @ line_start.T).T
                line_start /= line_start[...,2]
                line_end = (camera_matrix @ line_end.T).T
                line_end /= line_end[...,2]

                cv2.line(im_draw, line_start.squeeze().astype(int)[:2], line_end.squeeze().astype(int)[:2], (0, 255, 0), 3)

                # Draw left eye direction
                line_start = left_eye_mm/1000
                line_end = line_start + 0.05*denorm_gv
                line_start = (camera_matrix @ line_start.T).T
                line_start /= line_start[2,None]
                line_end = (camera_matrix @ line_end.T).T
                line_end /= line_end[2,None]

                cv2.line(im_draw, line_start.squeeze().astype(int)[:2], line_end.squeeze().astype(int)[:2], (0, 255, 0), 3)
            if visualize_landmarks:
                for p in face_mesh_corrected:
                    cv2.circle(im_draw, p[:2].astype(int), 1, (255,0,0), 1)

                if outputs[0] is not None:
                    i_l = denormalize_points(i["left"].reshape((5,-1))[:,:2], hr, left_eye_mm, (64,64), camera_matrix)
                    i_r = denormalize_points(i["right"].reshape((5,-1))[:,:2], hr, right_eye_mm, (64,64), camera_matrix)
                    for p in i_l:
                        cv2.circle(im_draw, p[:2].astype(int), 1, (255,255,0), 1)
                    for p in i_r:
                        cv2.circle(im_draw, p[:2].astype(int), 1, (255,255,0), 1)

            return {
                "gaze": denorm_gv,
                "face_position_in_m": face/100.0,
                "hr": hr,
                "visualization": im_draw,
                "extra_pose_data": pose_data,
                "variance": variance,
                "align_confidence": pose_data["align_confidence"]
            }

    def detect(self, *argv):
        return self.__call__(*argv)

    def compute_pose_data(self, image, camera_matrix, dist_coeff=np.zeros((4, 1))):
        with torch.no_grad():
            if not torch.is_tensor(image):
                image = TF.to_tensor(image).to(self.device)
                pose_data, face_mesh_corrected = self.face_detector.get_head_pose(image, camera_matrix, dist_coeff)
        return pose_data