import numpy as np
import torch
from bisect import bisect_left


import trimesh
import trimesh.visual
from PIL import Image
import pyrender

from ..face import FaceDetectAndAlign, default_face_model_path

from ..utilities.geometry import yaw_pitch_to_vector

from ..face import default_face_model_path

from .helpers import PersonDataset

def transform_vec3(trans, vec, last_val=1.0):
    return trans.dot(np.append(vec, last_val))[:3]

class RenderDataset(PersonDataset):
    # @staticmethod
    # def process_one_sample(self)
    @staticmethod
    def _get_val(lst, v):
        if lst[-1] < v:
            return None
        idx = bisect_left(lst, v, hi=len(lst) - 1)
        if lst[idx] == v:
            return idx + 1
        return idx
    
    def __init__(self, 
            dataset: PersonDataset, 
            camera_yaw_pitch = np.zeros((2,)), 
            normalized_distance=600, 
            normalized_focal_length=650, 
            image_width=64, 
            image_height=64, 
            camera_origin_transform=lambda o: o, 
            get_head_pose_transform=lambda: np.eye(4), 
            sample_duplicates=[1,0],
            face_align_confidence_threshold=10.,
            **kw):
        self.camera_matrix = np.array([[normalized_focal_length, 0, image_width / 2],
                            [0.0, normalized_focal_length, image_height / 2],
                            [0, 0, 1.0]])
        self.camera_yaw_pitch = camera_yaw_pitch
        self.normalized_distance = normalized_distance
        self.normalized_focal_length = normalized_focal_length
        self.image_width = image_width
        self.image_height = image_height
        self.camera_origin_transform = camera_origin_transform
        self.get_head_pose_transform = get_head_pose_transform
        self.sample_duplicates = sample_duplicates

        self.dataset = dataset

        self.set_camera_properties(self.camera_matrix, [image_height, image_width])

        self.detector = FaceDetectAndAlign(face_mesh_threshold=face_align_confidence_threshold)
        # annotations = {}
        self.face_model_mesh = trimesh.exchange.obj.load_obj(open(default_face_model_path))

    def set_camera_properties(self, camera_matrix, image_size=(64,64)):
        self.camera_matrix = camera_matrix
        self.image_size = image_size
        self.renderer = pyrender.OffscreenRenderer(viewport_width=image_size[1], viewport_height=image_size[0])
        # self.renderer = None

        fy = camera_matrix[1,1]
        # self.camera_fovy = 2 * np.arctan(2*fy/image_size[0])
        self.camera_fovy = 2 * np.arctan(image_size[0]/2/fy)
        self.camera_aspect_ratio = camera_matrix[1,1] / camera_matrix[0,0]

    @staticmethod
    def _get_camera_pose_normalized(eye, camera_origin, headpose):
        h_rx = headpose[:3, 0]
        forward = eye - camera_origin
        forward /= np.linalg.norm(forward)
        down = np.cross(forward, h_rx)
        down = down / np.linalg.norm(down)
        right = np.cross(down, forward)
        right = right / np.linalg.norm(right)

        rot_mat = np.c_[right, down, forward]

        camera_pose = np.eye(4)
        camera_pose[:3,:3] = rot_mat
        camera_pose[:3,3] = camera_origin
        return camera_pose

    @staticmethod
    def render_image(renderer, mesh, camera_pose, camera_fovy, camera_aspect_ratio):
        scene = pyrender.Scene(ambient_light=[0.5,0.5,0.5], bg_color=[0.485, 0.456, 0.406])
        ogl_camera_pose = np.eye(4)
        ogl_camera_pose[1,1] *= -1.0
        ogl_camera_pose[2,2] *= -1.0
        full_camera_pose = camera_pose.dot(ogl_camera_pose)
        # full_camera_pose = camera_pose
        # full_camera_pose[:3,:3] = camera_pose[:3,:3].dot(ogl_camera_pose)
        camera = pyrender.PerspectiveCamera(camera_fovy, aspectRatio=camera_aspect_ratio)
        scene.add(camera, "camera", pose=full_camera_pose)

        # scene.add(mesh, "mesh", pose=pose)
        mesh_pyrender = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        mesh_pyrender.primitives[0].material.baseColorFactor=np.array([1., 1., 1., 1.], dtype=np.float32)
        scene.add(mesh_pyrender, "mesh", pose=np.eye(4))

        # pyrender.Viewer(scene)

        color, _ = renderer.render(scene)
        return color

    @staticmethod
    def process_one_sample(sample, detector, face_model_mesh, camera_yaw_pitch, camera_fovy, camera_aspect_ratio, camera_matrix, renderer, normalized_distance=600, camera_origin_transform=lambda o: o, get_head_pose_transform=lambda: np.eye(4)):

        mesh, left_eye, right_eye, pose = RenderDataset.get_mesh(sample, detector, face_model_mesh)

        target = sample["target"]
        face = (left_eye + right_eye) / 2.0
        # pose = to_new_world.dot(person["poses"][idx])
        # pose = np.eye(4)

        head_pose_transform = get_head_pose_transform()

        prev_pose_inv = np.linalg.inv(pose)
        # new_pose = pose @ head_pose_transform
        new_pose = head_pose_transform

        target = transform_vec3(prev_pose_inv, target)
        target = transform_vec3(new_pose, target)
        face = transform_vec3(prev_pose_inv, face)
        face = transform_vec3(new_pose, face)
        right_eye = transform_vec3(prev_pose_inv, right_eye)
        right_eye = transform_vec3(new_pose, right_eye)
        left_eye = transform_vec3(prev_pose_inv, left_eye)
        left_eye = transform_vec3(new_pose, left_eye)
        mesh = mesh.apply_transform(prev_pose_inv)
        mesh = mesh.apply_transform(new_pose)

        pose = new_pose

        z_vec = yaw_pitch_to_vector(camera_yaw_pitch)
        z_vec *= normalized_distance/1000 / np.linalg.norm(z_vec)
        camera_origin = face - z_vec
        camera_origin = camera_origin_transform(camera_origin)

        # normalization stuff
        camera_pose_left = RenderDataset._get_camera_pose_normalized(left_eye, camera_origin, pose)
        camera_pose_right = RenderDataset._get_camera_pose_normalized(right_eye, camera_origin, pose)
        camera_pose_face = RenderDataset._get_camera_pose_normalized(face, camera_origin, pose)

        target_cam = np.linalg.inv(camera_pose_face).dot(np.append(target,1))[:3]
        face_cam = np.linalg.inv(camera_pose_face).dot(np.append(face,1))[:3]
        right_eye_cam = np.linalg.inv(camera_pose_face).dot(np.append(right_eye,1))[:3]
        left_eye_cam = np.linalg.inv(camera_pose_face).dot(np.append(left_eye,1))[:3]
        hr = np.linalg.inv(camera_pose_face).dot(pose)

        image_l = RenderDataset.render_image(renderer, mesh, camera_pose_left, camera_fovy, camera_aspect_ratio)
        image_r = RenderDataset.render_image(renderer, mesh, camera_pose_right, camera_fovy, camera_aspect_ratio)

        # Image.fromarray(image_l).save("tmp_figures/image_l.png")
        # Image.fromarray(image_r).save("tmp_figures/image_r.png")

        sample = {
            "M": camera_matrix, 
            "D": np.zeros((4,)),
            "target":target_cam, 
            "face": face_cam,
            # "image_path": person["path"][idx],
            "image_l": np.array(image_l, copy=False).copy(),
            "image_r": np.array(image_r, copy=False).copy(),
            "hr": hr,
            "right_eye_position": right_eye_cam,
            "left_eye_position": left_eye_cam,
        }
        return sample

    @staticmethod
    def get_mesh(sample, detector: FaceDetectAndAlign, face_model_mesh):
        vertices_pixels, vertices_metric, transform = detector.get_landmarks(sample["image"], sample["M"], sample["D"])
        # Image.fromarray(FaceDetectAndAlign.show_landmarks_on_image(np.array(sample["image"]), vertices_pixels)).save("tmp_figures/landmarks.png")

        # left_eye = transform_vec3(transform, FaceDetectAndAlign.left_eye_center_from_facemesh(vertices_metric)) / 100.0
        # right_eye = transform_vec3(transform, FaceDetectAndAlign.right_eye_center_from_facemesh(vertices_metric)) / 100.0

        left_eye = FaceDetectAndAlign.left_eye_center_from_facemesh(vertices_metric) / 100.0
        right_eye = FaceDetectAndAlign.right_eye_center_from_facemesh(vertices_metric) / 100.0
        

        uv = np.c_[vertices_pixels[:,0]/sample["image"].shape[1], (sample["image"].shape[0] - vertices_pixels[:,1])/sample["image"].shape[0]]

        texture = trimesh.visual.TextureVisuals(uv=uv, image=Image.fromarray(sample["image"]))

        # mesh = trimesh.Trimesh(vertices=vertices_metric/100, faces=face_model_mesh["faces"], visual=texture).apply_transform(np.linalg.inv(transform))
        mesh = trimesh.Trimesh(vertices=vertices_metric/100, faces=face_model_mesh["faces"], visual=texture)
        # mesh.apply_transform(transform)

        transform[:3,3] *= 0.01 # cm to m
        
        # mesh.export(open("tmp_figures/figure.glb", "wb"), "glb")
        
        return mesh, left_eye, right_eye, transform

    def get_number_of_persons(self):
        return self.dataset.get_number_of_persons()

    def get_persons_ids(self):
        return self.dataset.get_persons_ids()
    
    def get_person_dataset(self, p):
        return RenderDatasetSinglePerson(self.dataset.get_person_dataset(p), self)

    def __len__(self):
        return len(self.dataset) * (self.sample_duplicates[0] + self.sample_duplicates[1])
    
    def __getitem__(self, idx):
        total_duplicates = self.sample_duplicates[0] + self.sample_duplicates[1]
        real_idx = idx // total_duplicates
        duplicate_idx = idx % total_duplicates

        sample = self.dataset[real_idx]

        if duplicate_idx < self.sample_duplicates[0]:
            sample = RenderDataset.process_one_sample(
                sample,
                self.detector,
                self.face_model_mesh,
                self.camera_yaw_pitch, 
                self.camera_fovy, 
                self.camera_aspect_ratio, 
                self.camera_matrix,
                self.renderer,
                self.normalized_distance
            )
        else:
            sample = RenderDataset.process_one_sample(
                sample,
                self.detector,
                self.face_model_mesh,
                self.camera_yaw_pitch, 
                self.camera_fovy, 
                self.camera_aspect_ratio, 
                self.camera_matrix,
                self.renderer,
                self.normalized_distance,
                self.camera_origin_transform,
                self.get_head_pose_transform
            )
        return sample
    
class RenderDatasetSinglePerson(torch.utils.data.Dataset):
    def __init__(self, dataset, parent):
        self.dataset = dataset
        self.camera_matrix = parent.camera_matrix
        self.camera_yaw_pitch = parent.camera_yaw_pitch
        self.normalized_distance = parent.normalized_distance
        self.normalized_focal_length = parent.normalized_focal_length
        self.camera_fovy = parent.camera_fovy
        self.camera_aspect_ratio = parent.camera_aspect_ratio
        self.renderer = parent.renderer
        self.image_height = parent.image_height
        self.image_width = parent.image_width
        self.camera_origin_transform = parent.camera_origin_transform
        self.get_head_pose_transform = parent.get_head_pose_transform
        self.sample_duplicates = parent.sample_duplicates
        self.detector = parent.detector
        self.face_model_mesh = parent.face_model_mesh

    def __len__(self):
        return len(self.dataset) * (self.sample_duplicates[0] + self.sample_duplicates[1])
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # sample = UTMultiviewRenderDataset.process_one_sample(
        #     idx, 
        #     self.person, 
        #     self.camera_yaw_pitch, 
        #     self.camera_fovy, 
        #     self.camera_aspect_ratio, 
        #     self.camera_matrix,
        #     self.renderer,
        #     self.normalized_distance,
        #     self.camera_origin_transform,
        #     self.get_head_pose_transform,
        # )
        # return sample

        total_duplicates = self.sample_duplicates[0] + self.sample_duplicates[1]
        real_idx = idx // total_duplicates
        duplicate_idx = idx % total_duplicates
        sample = self.dataset[real_idx]

        if duplicate_idx < self.sample_duplicates[0]:
            sample = RenderDataset.process_one_sample(
                sample,
                self.detector,
                self.face_model_mesh,
                self.camera_yaw_pitch, 
                self.camera_fovy, 
                self.camera_aspect_ratio, 
                self.camera_matrix,
                self.renderer,
                self.normalized_distance
            )
        else:
            sample = RenderDataset.process_one_sample(
                sample,
                self.detector,
                self.face_model_mesh,
                self.camera_yaw_pitch, 
                self.camera_fovy, 
                self.camera_aspect_ratio, 
                self.camera_matrix,
                self.renderer,
                self.normalized_distance,
                self.camera_origin_transform,
                self.get_head_pose_transform
            )
        return sample
    

    def get_camera_matrices(self):
        return self.camera_matrix, np.zeros((4,))