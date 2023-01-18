import numpy as np

# def yaw_pitch_to_vector(yaw_pitch):
#     if np.ndim(yaw_pitch) == 1:
#         yaw_pitchs = np.expand_dims(yaw_pitch, 0)
#     else:
#         yaw_pitchs = yaw_pitch
#     n = yaw_pitchs.shape[0]
#     sin = np.sin(yaw_pitchs)
#     cos = np.cos(yaw_pitchs)
#     out = np.empty((n, 3))
#     out[:, 0] = np.multiply(cos[:, 1], sin[:, 0])
#     out[:, 1] = sin[:, 1]
#     out[:, 2] = np.multiply(cos[:, 1], cos[:, 0])
#     if np.ndim(yaw_pitch) == 1:
#         return out.squeeze() / np.linalg.norm(out.squeeze())
#     return out / np.linalg.norm(out)

# def vector_to_yaw_pitch(vector):
#     if np.ndim(vector) == 1:
#         vectors = np.expand_dims(vector, 0)
#     else:
#         vectors = vector
#     n = vectors.shape[0]
#     out = np.empty((n, 2))
#     vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
#     out[:, 1] = np.arcsin(vectors[:, 1])  # theta
#     out[:, 0] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
#     # out[out[:,0] < 0, 0] += 2*np.pi
#     if np.ndim(vector) == 1:
#         return out.squeeze()
#     return out

# def vector_to_yaw_pitch_gaze(vector):
#     out = vector_to_yaw_pitch(vector)
#     angle_force_positive(out[...,0])
#     # angle_force_half_circle(out[...,1])
#     # out[...,0] += np.pi
#     # out[...,1] *= -1.0
#     # print(np.allclose(vector/np.linalg.norm(vector), yaw_pitch_to_vector(out)))
#     return out

# def vector_to_yaw_pitch_head(vector):
#     out = vector_to_yaw_pitch(vector)
#     out[out >= np.pi] -= 2*np.pi
#     # angle_force_half_circle(out[...,0])
#     # angle_force_half_circle(out[...,1])
#     return out

# def angle_force_positive(angles, in_place=True):
#     a = angles if in_place else angles.copy()
#     a[a < 0] += 2*np.pi
#     return a

# def angle_force_half_circle(angles, in_place=True):
#     a = angles if in_place else angles.copy()
#     a[a > np.pi] -= 2*np.pi
#     a[a < -np.pi] += 2*np.pi

def vector_to_yaw_pitch_head(vector):
    return vector_to_yaw_pitch(vector)

def vector_to_yaw_pitch_gaze(vector):
    return vector_to_yaw_pitch(vector)

def vector_to_yaw_pitch(vector):
    if np.ndim(vector) == 1:
        vectors = np.expand_dims(vector, 0)
    else:
        vectors = vector
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 1] = np.arcsin(vectors[:, 1])  # theta
    out[:, 0] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out.squeeze()

def yaw_pitch_to_vector(yaw_pitch):
    if np.ndim(yaw_pitch) == 1:
        yaw_pitchs = np.expand_dims(yaw_pitch, 0)
    else:
        yaw_pitchs = yaw_pitch
    n = yaw_pitchs.shape[0]
    sin = np.sin(yaw_pitchs)
    cos = np.cos(yaw_pitchs)
    out = np.empty((n, 3))
    out[:, 0] = np.multiply(cos[:, 1], sin[:, 0])
    out[:, 1] = sin[:, 1]
    out[:, 2] = np.multiply(cos[:, 1], cos[:, 0])
    return out.squeeze()

def _R_x(theta):
    sin_ = np.sin(theta)
    cos_ = np.cos(theta)
    return np.array([
        [1., 0., 0.],
        [0., cos_, -sin_],
        [0., sin_, cos_]
    ]). astype(np.float32)

def _R_y(phi):
    sin_ = np.sin(phi)
    cos_ = np.cos(phi)
    return np.array([
        [cos_, 0., sin_],
        [0., 1., 0.],
        [-sin_, 0., cos_]
    ]). astype(np.float32)

def calculate_rotation_matrix(yaw_pitch):
    return np.matmul(_R_y(yaw_pitch[0]), _R_x(yaw_pitch[1]))

# def matrix_to_yaw_pitch(rot_mat):
#     return np.array([np.arctan2(rot_mat[0,2], rot_mat[2,2]), np.arcsin(rot_mat[1,2])])


def cosine_similarity(v1, v2):
    if np.ndim(v1) == 1:
        v1 = np.expand_dims(v1, 0)
    if np.ndim(v2) == 1:
        v2 = np.expand_dims(v2, 0)
    return np.einsum('ij,ij->i', v1, v2)/np.linalg.norm(v1, axis=1)/np.linalg.norm(v2, axis=1)

def cosine_similarity_angle(v1, v2):
    sim = cosine_similarity(v1, v2)
    sim = np.clip(sim, -1.0 + 1e-6, 1.0 - 1e-6)
    return np.arccos(sim)

def compute_yaw_pitch(face_position, gaze_target):
    # For compatibility
    return vector_to_yaw_pitch(gaze_target - face_position)


class Rectangle():
    def __init__(self, top: float, left: float, bottom: float, right: float) -> None:
        self.t = top
        self.l = left
        self.b = bottom
        self.r = right
    
    def scale(self, scale: float):
        return Rectangle(self.t*scale,self.l*scale,self.b*scale,self.r*scale)

    def expand(self, scale: float):
        w = self.width()
        h = self.height()
        return Rectangle(self.t - h*scale, self.l - w*scale, self.b + h*scale, self.r + w*scale)
    
    def translate(self, x: float, y: float):
        t = self.t + y
        l = self.l + x
        b = self.b + y
        r = self.r + x
        return Rectangle(t,l,b,r)

    def tlbr(self):
        return self.t, self.l, self.b, self.r

    def tl(self):
        return self.t, self.l
    
    def lt(self):
        return self.l, self.t
    
    def br(self):
        return self.b, self.r

    def width(self):
        return self.r - self.l
    
    def height(self):
        return self.b - self.t

    def hcenter(self):
        return (self.l + self.r)/2.0
    
    def vcenter(self):
        return (self.t + self.b)/2.0
    
    def tlbr_i(self):
        return int(self.t), int(self.l), int(self.b), int(self.r)

    def tl_i(self):
        return int(self.t), int(self.l)

    def lt_i(self):
        return int(self.l), int(self.t)
    
    def br_i(self):
        return int(self.b), int(self.r)

    def width_i(self):
        return int(self.r - self.l)
    
    def height_i(self):
        return int(self.b - self.t)

    def hcenter_i(self):
        return int((self.l + self.r)/2.0)
    
    def vcenter_i(self):
        return int((self.t + self.b)/2.0)

    def to_dict(self):
        d = {
            "top": self.t,
            "left": self.l,
            "bottom": self.b,
            "right": self.r
        }
        return d
    
    def to_dict_i(self):
        d = {
            "top": int(self.t),
            "left": int(self.l),
            "bottom": int(self.b),
            "right": int(self.r)
        }
        return d

    @staticmethod
    def from_dict(d):
        return Rectangle(d["top"], d["left"], d["bottom"], d["right"])

    @staticmethod
    def from_center(vcenter: float, hcenter: float, height: float, width: float):
        top = vcenter - height/2.0
        left = hcenter - width/2.0
        return Rectangle(top, left, top+height, left+width)
    
    def is_null(self):
        return (self.t - self.b == 0) or (self.r - self.l) == 0
