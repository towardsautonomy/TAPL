import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import yaml

class CameraModel:

    def __init__(self, data_path=None):
        self.N_X_CORNER_POINTS = 4
        self.N_Y_CORNER_POINTS = 4
        if data_path is None:
            self.CALIB_DATA_PATH = '../data/calib/'
        else:
            self.CALIB_DATA_PATH = data_path

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((self.N_Y_CORNER_POINTS*self.N_X_CORNER_POINTS,3), np.float32)
        self.objp[:,:2] = np.mgrid[0:self.N_X_CORNER_POINTS, 0:self.N_Y_CORNER_POINTS].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d points in real world space
        self.imgpoints = [] # 2d points in image plane.

        # Make a list of calibration images
        self.images = glob.glob(self.CALIB_DATA_PATH + '*.JPEG')

    def calibrate(self):
        img_size = None
        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(self.images):
            img = cv2.imread(fname)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if img_size is None:
                    img_size = gray.shape

                # Find the chessboard corners
                ret, corners = cv2.findChessboardCorners(gray, (self.N_X_CORNER_POINTS, self.N_Y_CORNER_POINTS), None)

                # If found, add object points, image points
                if ret == True:
                    self.objpoints.append(self.objp)
                    self.imgpoints.append(corners)

                    # Draw and display the corners
                    cv2.drawChessboardCorners(img, (self.N_X_CORNER_POINTS, self.N_Y_CORNER_POINTS), corners, ret)
                    cv2.imshow('img', img)
                    cv2.waitKey(1)

        cv2.destroyAllWindows()

        # Do camera calibration given object points and image points
        print('calibrating...')
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, img_size, None, None)
        
        return mtx, dist

if __name__ == '__main__':
    # Camera Calibration
    camModel = CameraModel()
    # mtx, dist = camModel.calibrate()
    # # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    # # cam_model = {"camera_matrix": mtx, "dist_coeff": dist}

    # print("Camera Matrix:")
    # print(mtx)
    # print("Distortion Matrix:")
    # print(dist)
    
    # # # Dump the Camera Model and Distortion Matrix into a yaml file
    fname = "../data/calib/camera_model.yaml"
    # cam_model = cv2.FileStorage(fname, cv2.FILE_STORAGE_WRITE)
    # cam_model.write("camera_matrix", mtx)
    # cam_model.write("dist_coeff", dist)
    # cam_model.release()
        
    # Test undistortion on an image
    cam_model = cv2.FileStorage(fname, cv2.FILE_STORAGE_READ)
    mtx = cam_model.getNode("camera_matrix").mat()
    dist = cam_model.getNode("dist_coeff").mat()
    cam_model.release()

    img = cv2.cvtColor(cv2.imread('../data/calib/IMG_1450.JPEG'), cv2.COLOR_BGR2RGB)
    undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
        
    # Visualize undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(undistorted_img)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.show()
