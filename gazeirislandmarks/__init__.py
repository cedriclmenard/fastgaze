# Constants and options
use_static_denormalization = True
# print(f"WARNING: Static normalization usage is: {use_static_denormalization}")

from .gaze import GazeDetector
from .face import FaceNotDetectedError
from .utilities.geometry import cosine_similarity_angle
