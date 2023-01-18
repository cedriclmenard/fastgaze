import numpy as np
import cv2


class Procrustes():
    procrustes_landmark_basis = [
        (4, 0.070909939706326),
        (6, 0.032100144773722),
        (10, 0.008446550928056),
        (33, 0.058724168688059),
        (54, 0.007667080033571),
        (67, 0.009078059345484),
        (117, 0.009791937656701),
        (119, 0.014565368182957),
        (121, 0.018591361120343),
        (127, 0.005197994410992),
        (129, 0.120625205338001),
        (132, 0.005560018587857),
        (133, 0.05328618362546),
        (136, 0.066890455782413),
        (143, 0.014816547743976),
        (147, 0.014262833632529),
        (198, 0.025462191551924),
        (205, 0.047252278774977),
        (263, 0.058724168688059),
        (284, 0.007667080033571),
        (297, 0.009078059345484),
        (346, 0.009791937656701),
        (348, 0.014565368182957),
        (350, 0.018591361120343),
        (356, 0.005197994410992),
        (358, 0.120625205338001),
        (361, 0.005560018587857),
        (362, 0.05328618362546),
        (365, 0.066890455782413),
        (372, 0.014816547743976),
        (376, 0.014262833632529),
        (420, 0.025462191551924),
        (425, 0.047252278774977),
    ]

    
    def __init__(self, model_landmarks, camera_matrix, dist_coeff, frame_width, frame_height, near_clipping_plane=1, far_clipping_plane=10000):
        self.dist_coeff = dist_coeff
        self.near = near_clipping_plane
        self.far = far_clipping_plane
        self.camera_matrix = camera_matrix
        self.model_landmarks = model_landmarks
        self.landmark_weights = np.zeros((self.model_landmarks.shape[0],))
        for idx, weight in self.procrustes_landmark_basis:
            self.landmark_weights[idx] = weight
        self.points_idx = [33,263,61,291,199]
        self.points_idx = self.points_idx + [key for (key,val) in self.procrustes_landmark_basis]
        self.points_idx = list(set(self.points_idx))
        self.points_idx.sort()
        self.frame_width = frame_width
        self.frame_height = frame_height

        # near plane properties
        fy = camera_matrix[1,1]
        fovy = 2 * np.arctan(frame_height / (2 * fy))
        height_at_near = 2 * self.near * np.tan(0.5 * fovy)
        width_at_near = frame_width * height_at_near / frame_height

        self.left_near = -0.5 * width_at_near
        self.right_near = 0.5 * width_at_near
        self.bottom_near = -0.5 * height_at_near
        self.top_near = 0.5 * height_at_near

    def __call__(self, face_landmarks):
        metric_landmarks, pose_transform = self._convert_to_metric(face_landmarks.copy())
        model_points = metric_landmarks[self.points_idx,:]
        image_points = face_landmarks[self.points_idx,:2]

        success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, self.camera_matrix, self.dist_coeff, flags=cv2.SOLVEPNP_ITERATIVE)
        return rotation_vector.squeeze(), translation_vector.squeeze()

    def compute_metric_landmarks(self, face_landmarks):
        metric_landmarks, pose_transform = self._convert_to_metric(face_landmarks.copy())

        model_points = metric_landmarks[self.points_idx,:]
        image_points = face_landmarks[self.points_idx,:2]
        success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, self.camera_matrix, self.dist_coeff, flags=cv2.SOLVEPNP_ITERATIVE)
        return (pose_transform[:3,:3] @ metric_landmarks.T).T + pose_transform[:3,3], rotation_vector.squeeze(), translation_vector.squeeze()

    def update_camera(self, camera_matrix, dist_coeff, frame_width, frame_height):
        self.camera_matrix = camera_matrix
        self.dist_coeff = dist_coeff
        self.frame_width = frame_width
        self.frame_height = frame_height

        # near plane properties
        fy = camera_matrix[1,1]
        fovy = 2 * np.arctan(frame_height / (2 * fy))
        height_at_near = 2 * self.near * np.tan(0.5 * fovy)
        width_at_near = frame_width * height_at_near / frame_height

        self.left_near = -0.5 * width_at_near
        self.right_near = 0.5 * width_at_near
        self.bottom_near = -0.5 * height_at_near
        self.top_near = 0.5 * height_at_near


    def _normalize(self, points):
        points[:, 0] /= self.frame_width
        points[:, 1] /= self.frame_height
        points[:, 2] /= self.frame_width
        return points

    def _project_near(self, points):
        x_scale = self.right_near - self.left_near
        y_scale = self.top_near - self.bottom_near
        x_translation = self.left_near
        y_translation = self.bottom_near
        
        points[:,1] = 1.0 - points[:,1]
        
        points = points * np.array([[x_scale,y_scale,x_scale]])
        points = points + np.array([[x_translation,y_translation,0]])
        
        return points

    def _unproject_near(self, points):
        points[:,0] = points[:,0] * points[:,2] / self.near
        points[:,1] = points[:,1] * points[:,2] / self.near
        
        return points

    def _estimate_scale(self, landmarks):
        transform = Procrustes._orthogonal_problem_solve(self.model_landmarks, landmarks, self.landmark_weights)
        return np.linalg.norm(transform[:,0])

    def _convert_to_metric(self, face_landmarks):

        screen_landmarks = self._project_near(self._normalize(face_landmarks))

        ## FIRST PASS
        depth_offset = np.mean(screen_landmarks[:,2])

        # flip handedness NOTE: check if required
        tmp_landmarks = screen_landmarks.copy()
        tmp_landmarks[:,2] *= -1.0

        # first pass scale estimation
        first_scale_estimate = self._estimate_scale(tmp_landmarks)

        ## SECOND PASS
        tmp_landmarks = screen_landmarks.copy()
        # move and rescale z
        tmp_landmarks[:,2] = (tmp_landmarks[:,2] - depth_offset + self.near) / first_scale_estimate

        tmp_landmarks = self._unproject_near(tmp_landmarks)
        
        # flip handedness
        tmp_landmarks[:,2] *= -1.0

        second_scale_estimate = self._estimate_scale(tmp_landmarks)


        ## FINAL PASS
        metric_landmarks = screen_landmarks.copy()
        total_scale = first_scale_estimate * second_scale_estimate
        # move and rescale z
        metric_landmarks[:,2] = (metric_landmarks[:,2] - depth_offset + self.near) / total_scale
        metric_landmarks = self._unproject_near(metric_landmarks)
        # flip handedness
        metric_landmarks[:,2] *= -1.0

        pose_transform = Procrustes._orthogonal_problem_solve(self.model_landmarks, metric_landmarks, self.landmark_weights)

        inv_pose_transform = np.linalg.inv(pose_transform)
        inv_pose_rotation = inv_pose_transform[:3, :3]
        inv_pose_translation = inv_pose_transform[:3, 3]

        metric_landmarks = (inv_pose_rotation @ metric_landmarks.T + inv_pose_translation[:, None]).T

        return metric_landmarks, pose_transform


    @staticmethod
    def _compute_optimal_rotation(design_matrix):
        if np.linalg.norm(design_matrix) < 1e-9: # Norm is too small
            raise ValueError("Design matrix has a too small norm")

        u, _, vh = np.linalg.svd(design_matrix)
        postrotation = u
        prerotation = vh

        if np.linalg.det(postrotation) * np.linalg.det(prerotation) < 0:
            postrotation[:,2] = -1 * postrotation[:,2]
        
        rotation = postrotation @ prerotation
        return rotation
    
    @staticmethod
    def _compute_optimal_scale(w_source_centered, w_source, w_dest, rotation):
        r_w_source_centered = (rotation @ w_source_centered.T).T
        numerator = np.sum(r_w_source_centered.T * w_dest.T)
        denominator = np.sum(w_source_centered.T * w_source.T)
        scale = numerator / denominator
        if denominator < 1e-9:
            raise ValueError("Denominator is too small")
        if scale < 1e-9:
            raise ValueError("Scale is too small")
        return scale



    @staticmethod
    def _orthogonal_problem_solve(source, dest, weights):
        w = np.sqrt(weights)
        weight_total = np.sum(weights)
        w_source = source * w[:,None]
        ww_source = w_source * w[:,None]
        w_dest = dest * w[:,None]

        source_weighted_center = np.sum(ww_source, axis=0)/weight_total
        w_source_centered = w_source - (source_weighted_center[:, None] @ w[None,:]).T

        design_matrix = w_dest.T @ w_source_centered

        rotation = Procrustes._compute_optimal_rotation(design_matrix)

        scale = Procrustes._compute_optimal_scale(w_source_centered, w_source, w_dest, rotation)

        rotation_scale = scale * rotation

        pointwise_diffs = w_dest - (rotation_scale @ w_source.T).T
            
        w_pointwise_diffs = pointwise_diffs * w[:,None]
        
        translation = np.sum(w_pointwise_diffs, axis=0) / weight_total
        
        transform = np.eye(4)
        transform[:3,:3] = rotation_scale
        transform[:3, 3] = translation
        return transform