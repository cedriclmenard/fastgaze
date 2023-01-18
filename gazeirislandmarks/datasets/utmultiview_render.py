from bisect import bisect_left
from glob import glob
import numpy as np
import pyrender
import trimesh
import trimesh.exchange.obj
from trimesh.resolvers import FilePathResolver
import os.path
import torch
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

from .utmultiview import UTMultiviewDataset, PersonDataset
from ..utilities.geometry import yaw_pitch_to_vector, calculate_rotation_matrix

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


class UTMultiviewRenderDataset(PersonDataset):
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
            path, 
            camera_yaw_pitch = np.zeros((2,)), 
            normalized_distance=600, 
            normalized_focal_length=650, 
            image_width=64, 
            image_height=64, 
            camera_origin_transform=lambda o: o, 
            get_head_pose_transform=lambda: np.eye(4), 
            sample_duplicates=[1,0],
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

        self.path = path

        self.set_camera_properties(self.camera_matrix, [image_height, image_width])

        # List person directories
        persons_path = sorted(glob(path + "/*/"))
        
        self.n_per_p = []
        self.n = 0
        self.cumulative_n = []
        self.persons = []
        self.features_samples = []
        self.headpose_samples = []
        for p in persons_path:
            poses = []
            right_eye_centers = []
            left_eye_centers = []
            face_centers = []
            gaze_targets = []
            sample_paths = []
            meshes = []
            monitor_rs = []
            monitor_ts = []
            
            with open(os.path.join(p, "raw", "monitor.txt"), "r") as f:
                lines = f.readlines()
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

                # get 3d model
                mesh_trimesh = trimesh.load(os.path.join(sample_dir, "models", "face_mesh.obj"))
                scale = np.eye(4)
                scale[:3,:3] *= 1e-3
                mesh_trimesh.apply_transform(scale)

                # mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(**obj_data))
                # mesh = pyrender.Mesh.from_trimesh(mesh_trimesh, smooth=False)
                mesh = mesh_trimesh
                # mesh.primitives[0].material.baseColorFactor=np.array([1., 1., 1., 1.], dtype=np.float32)

                pose = np.eye(4)
                pose[:3,:3] = head_r
                pose[:3,3] = head_t * 1e-3

                left_eye = (features[2,:] + features[3,:])/2.0 / 1000.0
                right_eye = (features[0,:] + features[1,:])/2.0 / 1000.0

                poses.append(pose)
                gaze_targets.append(g_w/1000.0)
                left_eye_centers.append(left_eye)
                right_eye_centers.append(right_eye)
                face_centers.append((left_eye + right_eye)/2.0)
                sample_paths.append(sample_dir)
                meshes.append(mesh)

                monitor_rs.append(monitor_r)
                monitor_ts.append(monitor_t)

            identifier = os.path.basename(os.path.normpath(p))

            person = {
                "id": identifier,
                "path": p,
                "n": len(sample_paths),
                "targets": np.array(gaze_targets, dtype=np.float32),
                "faces": np.array(face_centers, dtype=np.float32),
                "poses": poses,
                "right_eye_centers": right_eye_centers,
                "left_eye_centers": left_eye_centers,
                "meshes": meshes,
                "monitor_rs": monitor_rs,
                "monitor_ts": monitor_ts
            }

            # expand with number of duplicate samples
            n = len(sample_paths) * (self.sample_duplicates[0] + self.sample_duplicates[1])
            self.n_per_p.append(n)
            self.n += n
            self.cumulative_n.append(self.n)
            self.persons.append(person)
        
        self.np = len(self.persons)

    def set_camera_properties(self, camera_matrix, image_size=(64,64)):
        self.camera_matrix = camera_matrix
        self.image_size = image_size
        self.renderer = pyrender.OffscreenRenderer(viewport_width=image_size[1], viewport_height=image_size[0])

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

        color, _ = renderer.render(scene)
        return color

    @staticmethod
    def process_one_sample(idx, person, camera_yaw_pitch, camera_fovy, camera_aspect_ratio, camera_matrix, renderer, normalized_distance=600, camera_origin_transform=lambda o: o, get_head_pose_transform=lambda: np.eye(4)):

        # find the center of the screen as the new "world" coordinates
        monitor_pose = np.eye(4)
        monitor_pose[:3,:3] = person["monitor_rs"][idx]
        monitor_pose[:3,3] = person["monitor_ts"][idx]/1000

        inv_monitor_pose = np.linalg.inv(monitor_pose)
        new_world_center_mon = inv_monitor_pose[:3,3].squeeze()
        new_world_center_mon[0] /= 2.0 # horizontal center of monitor
        new_world_center_mon[2] = 0.0 # flat with monitor, not in front of it
        new_world_mon = np.eye(4)
        # flip x and z
        new_world_mon[0,0] *= -1.0
        new_world_mon[2,2] *= -1.0
        new_world_mon[:3,3] = new_world_center_mon

        # to original world coordinates
        new_world_cam = monitor_pose.dot(new_world_mon)
        to_new_world = np.linalg.inv(new_world_cam)

        def transform_vec3(trans, vec, last_val=1.0):
            return trans.dot(np.append(vec, last_val))[:3]

        

        target = transform_vec3(to_new_world, person["targets"][idx, ...])
        face = transform_vec3(to_new_world, person["faces"][idx, ...])
        pose = to_new_world.dot(person["poses"][idx])
        right_eye = transform_vec3(to_new_world, person["right_eye_centers"][idx])
        left_eye = transform_vec3(to_new_world, person["left_eye_centers"][idx])
        mesh = person["meshes"][idx].copy().apply_transform(to_new_world)

        head_pose_transform = get_head_pose_transform()

        prev_pose_inv = np.linalg.inv(pose)
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
        camera_pose_left = UTMultiviewRenderDataset._get_camera_pose_normalized(left_eye, camera_origin, pose)
        camera_pose_right = UTMultiviewRenderDataset._get_camera_pose_normalized(right_eye, camera_origin, pose)
        camera_pose_face = UTMultiviewRenderDataset._get_camera_pose_normalized(face, camera_origin, pose)

        # gaze_vector = face - target
        # gaze_vector_cam = np.linalg.inv(camera_pose_face).dot(np.append(gaze_vector,0))[:3]

        target_cam = np.linalg.inv(camera_pose_face).dot(np.append(target,1))[:3]
        face_cam = np.linalg.inv(camera_pose_face).dot(np.append(face,1))[:3]
        right_eye_cam = np.linalg.inv(camera_pose_face).dot(np.append(right_eye,1))[:3]
        left_eye_cam = np.linalg.inv(camera_pose_face).dot(np.append(left_eye,1))[:3]
        hr = np.linalg.inv(camera_pose_face).dot(pose)

        image_l = UTMultiviewRenderDataset.render_image(renderer, mesh, camera_pose_left, camera_fovy, camera_aspect_ratio)
        image_r = UTMultiviewRenderDataset.render_image(renderer, mesh, camera_pose_right, camera_fovy, camera_aspect_ratio)

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

    def get_number_of_persons(self):
        return self.np

    def get_persons_ids(self):
        return [p["id"] for p in self.persons]
    
    def get_person_dataset(self, p):
        return UTMultiviewRenderDatasetSinglePerson(self.persons[p], self)

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

        total_duplicates = self.sample_duplicates[0] + self.sample_duplicates[1]
        real_idx_in_p = idx_in_p // total_duplicates
        duplicate_idx = idx_in_p % total_duplicates

        if duplicate_idx < self.sample_duplicates[0]:
            sample = UTMultiviewRenderDataset.process_one_sample(
                real_idx_in_p, 
                self.persons[p], 
                self.camera_yaw_pitch, 
                self.camera_fovy, 
                self.camera_aspect_ratio, 
                self.camera_matrix,
                self.renderer,
                self.normalized_distance
            )
        else:
            sample = UTMultiviewRenderDataset.process_one_sample(
                real_idx_in_p, 
                self.persons[p], 
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
    
class UTMultiviewRenderDatasetSinglePerson(torch.utils.data.Dataset):
    def __init__(self, person, parent):
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

        self.n = len(person["meshes"]) * (self.sample_duplicates[0] + self.sample_duplicates[1])
        self.person = person

    def __len__(self):
        return self.n
    
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

        if duplicate_idx < self.sample_duplicates[0]:
            sample = UTMultiviewRenderDataset.process_one_sample(
                real_idx, 
                self.person, 
                self.camera_yaw_pitch, 
                self.camera_fovy, 
                self.camera_aspect_ratio, 
                self.camera_matrix,
                self.renderer,
                self.normalized_distance
            )
        else:
            sample = UTMultiviewRenderDataset.process_one_sample(
                real_idx, 
                self.person, 
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