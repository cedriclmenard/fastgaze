import numpy as np
import cv2
import acapture
from torch._C import Value

class FaceCamera():
    def __init__(self, idx, frame_width=None, frame_height=None, fovx=None, focal_length=None, aperture=None):
        self.cap = acapture.open(idx)
        if frame_width is None or frame_height is None:
            frame_height, frame_width = self.cap.read()[1].shape[:2]
        if focal_length is None and aperture is None and not fovx is None:
            self.get_simple_calibration_values(fovx=fovx, frame_height=frame_height, frame_width=frame_width)
        elif fovx is None and not focal_length is None and not aperture is None:
            self.get_simple_calibration_values(focal_length=focal_length, aperture=aperture, frame_height=frame_height, frame_width=frame_width)
        else:
            raise ValueError('Either the FOV or the focal length AND aperture must be provided.')
        
    
    def read(self):
        return self.cap.read()

    def calibration_procedure(self, numberFrames, squareSize=0.023, m=9, n=6):
        raise NotImplementedError

    def save_current_calibration_values(self, filename):
        fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
        fs.write(name="M", val=self.M)
        fs.write(name="D", val=self.D)
        # fs.write(name="rv", val=self.rvecs)
        # fs.write(name="tv", val=self.tvecs)
        fs.release

    def load_calibration_values(self, filename):
        fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        self.M = fs.getNode("M").mat()
        self.D = fs.getNode("D").mat()
        # self.rvecs = fs.read(name="rv")
        # self.tvecs = fs.read(name="tv")
        fs.release

    def evaluate_depth_dx(self, dxp, dx):
        fx = self.M[0,0]
        return fx*dx/dxp
    
    def evaluate_depth_dy(self, dyp, dy):
        fy = self.M[1,1]
        return fy*dy/dyp

    def evaluate_position(self, xp, yp, z):
        cx = self.M[0, 2]
        cy = self.M[1, 2]
        fx = self.M[0, 0]
        fy = self.M[1, 1]
        
        return [(xp - cx)*z/fx, (yp - cy)*z/fy, z]
    
    def get_simple_calibration_values(self, frame_height: int, frame_width: int, fovx: int=None, focal_length: float=None, aperture: float=None):
        """[summary]

        Args:
            frame_height (int): Height in pixels of the image frame.
            frame_width (int): Width in pixels of the image frame
            fovx (int): Horizontal field of view of the camera.
            focal_length (float): Focal length in the same real world units as the aperture.
            aperture (float): Aperture in the same real world units as the focal_length. General given as a ratio of the focal length e.g., f/2.2.
        """
        center = (frame_width/2, frame_height/2)
        if not fovx is None:
            fx = 0.5 * frame_width / np.tan(0.5 * fovx * np.pi / 180.0)
            fy = 0.5 * frame_height / np.tan(0.5 * fovx * frame_height / frame_width * np.pi / 180.0)
        elif not focal_length is None and not aperture is None:
            fx = focal_length * frame_width / aperture
            fy = focal_length * frame_height / aperture
        else:
            raise ValueError('fovx or focal length and aperture should be defined')
        camera_matrix = np.array(
                                [[fx, 0, center[0]],
                                [0, fy, center[1]],
                                [0, 0, 1]], dtype = "double"
                                )

        dist_coeff = np.zeros((4, 1))
        self.M = camera_matrix
        self.D = dist_coeff


class FaceCameraPSEye(FaceCamera):
    def __init__(self, idx, fovx=56):
        self.cap = acapture.open(idx)
        self.get_simple_calibration_values(fovx)
    
    def read(self):
        return self.cap.read()
    
    def calibration_procedure(self, numberFrames, squareSize=0.023, m=9, n=6):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((m*n,3), np.float32)
        objp[:,:2] = np.mgrid[0:m,0:n].T.reshape(-1,2)*squareSize
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        gray = None

        i = 0
        while True:
            if i > numberFrames:
                break
            check, img = self.read()
            if not check:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (m,n), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (m,n), corners2, ret)
                cv2.imshow('img', img)
                key = -1
                while key < 0:
                    key = cv2.waitKey()
                if key == 13: # Enter
                    i += 1
                    imgpoints.append(corners2)
                    objpoints.append(objp)
                    print("Appended this image")
            else:
                cv2.imshow('img', img)
                cv2.waitKey(1)
                # if key >= 0:
                #     print("key pressed: " + str(key))
            
        if i != 0:
            # Run calibration
            ret, self.M, self.D, self.rvecs, self.tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    def save_current_calibration_values(self, filename):
        fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
        fs.write(name="M", val=self.M)
        fs.write(name="D", val=self.D)
        # fs.write(name="rv", val=self.rvecs)
        # fs.write(name="tv", val=self.tvecs)
        fs.release()

    def load_calibration_values(self, filename):
        fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        self.M = fs.getNode("M").mat()
        self.D = fs.getNode("D").mat()
        # self.rvecs = fs.read(name="rv")
        # self.tvecs = fs.read(name="tv")
        fs.release()

    def get_simple_calibration_values(self, fovx=56):
        """[summary]

        Args:
            fovx (int, optional): Should be 56 or 76 depending on selected mechanical setting. Defaults to 56.
        """
        frame_height, frame_width, channels = (480, 640, 3)
        center = (frame_width/2, frame_height/2)
        # fovx = 56 # 76 or 56
        fx = 0.5 * frame_width / np.tan(0.5 * fovx * np.pi / 180.0)
        fy = 0.5 * frame_height / np.tan(0.5 * fovx * frame_height / frame_width * np.pi / 180.0)
        camera_matrix = np.array(
                                 [[fx, 0, center[0]],
                                 [0, fy, center[1]],
                                 [0, 0, 1]], dtype = "double"
                                 )

        dist_coeff = np.zeros((4, 1))
        self.M = camera_matrix
        self.D = dist_coeff

    def evaluate_depth_dx(self, dxp, dx):
        fx = self.M[0,0]
        return fx*dx/dxp
    
    def evaluate_depth_dy(self, dyp, dy):
        fy = self.M[1,1]
        return fy*dy/dyp

    def evaluate_position(self, xp, yp, z):
        cx = self.M[0, 2]
        cy = self.M[1, 2]
        fx = self.M[0, 0]
        fy = self.M[1, 1]
        
        return [(xp - cx)*z/fx, (yp - cy)*z/fy, z]

    # TODO: add undistort procedure, maybe only for points


def load_calibration_values(filename):
        fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        M = fs.getNode("M").mat()
        D = fs.getNode("D").mat()
        # self.rvecs = fs.read(name="rv")
        # self.tvecs = fs.read(name="tv")
        fs.release
        return M, D

def evaluate_depth_dx(dxp, dx, M):
        fx = M[0,0]
        return fx*dx/dxp
    
def evaluate_depth_dy(dyp, dy, M):
    fy = M[1,1]
    return fy*dy/dyp

def evaluate_position(xp, yp, z, M):
    cx = M[0, 2]
    cy = M[1, 2]
    fx = M[0, 0]
    fy = M[1, 1]
    
    return [(xp - cx)*z/fx, (yp - cy)*z/fy, z]