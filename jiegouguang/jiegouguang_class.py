import cv2
import numpy as np
import copy
import open3d as o3d
from manual_feature.ManualFeature import ManualFeatureJieGouGuang

from sgbm.sgbm import SGBM

from FoundationStereo.stereo_inference import StereoInference

class JieGouGuang:
    def __init__(self,img1_path,img2_path):
        self.img1_path = img1_path
        self.img2_path = img2_path
        self.img1 = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2GRAY)
        self.img2 = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2GRAY)

        self.max_dis = 2000
        self.min_dis = 100
        # 最近最远深度

        self.binary_image_threshold = [19,-4]
        # 自适应二值化的两个参数
        self.contourArea_threshold = 30
        # 过滤光斑的参数
        self.lightfilter_threshold = 0
        # 按照亮度过滤的参数

    def import_biaodin(self,extri_path,intri_path):
        extri = cv2.FileStorage(extri_path, cv2.FILE_STORAGE_READ)
        intri = cv2.FileStorage(intri_path, cv2.FILE_STORAGE_READ)

        M1 = intri.getNode('M1').mat()
        M2 = intri.getNode('M2').mat()
        D1 = intri.getNode('D1').mat()
        D2 = intri.getNode('D2').mat()

        R = extri.getNode('R').mat()
        t = extri.getNode('T').mat()
        t_cross = np.array([[0, -t[2][0], t[1][0]],
                            [t[2][0], 0, -t[0][0]],
                            [-t[1][0], t[0][0], 0]])

        F = np.dot(np.dot(np.transpose(np.linalg.inv(M2)), np.dot(t_cross, R)), np.linalg.inv(M1))


        h, w = self.img1.shape[:2]
        new_M1, roi1 = cv2.getOptimalNewCameraMatrix(M1, D1, (w, h), 1, (w, h))
        new_M2, roi2 = cv2.getOptimalNewCameraMatrix(M2, D2, (w, h), 1, (w, h))

        self.K1 = new_M1
        self.K2 = new_M2

        self.cam_R = R
        self.cam_t = t

        # 计算极线变换矩阵
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(new_M1, D1, new_M2, D2, (w, h), R, t, flags=cv2.CALIB_ZERO_TANGENT_DIST)

        self.P1 = P1
        self.P2 = P2

        self.Q = Q


        # 生成映射
        left_map_x, left_map_y = cv2.initUndistortRectifyMap(new_M1, D1, R1, P1, (w, h), cv2.CV_32FC1)
        right_map_x, right_map_y = cv2.initUndistortRectifyMap(new_M2, D2, R2, P2, (w, h), cv2.CV_32FC1)

        # 进行矫正
        left_rectified = cv2.remap(self.img1, left_map_x, left_map_y, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(self.img2, right_map_x, right_map_y, cv2.INTER_LINEAR)

        self.img1_rectify = left_rectified
        self.img2_rectify = right_rectified

        f = float(self.K1[0,0])
        B = abs(float(self.cam_t[0][0]))
        self.max_disp = f * B / self.min_dis  # for closest points
        self.min_disp = f * B / self.max_dis  # for farthest points

    def manual_feature_extracting(self):
        processor = ManualFeatureJieGouGuang(base=self)

        img1_with_center,img2_with_center = processor.extract_circle()
        processor.feature_extracting()
        processor.feature_matching()
        pcd = processor.pointcloud_from_disparity()

        return pcd
    
    def sgbm(self):
        processor = SGBM(base=self)

        pcd = processor.sgbm()

        return pcd
    

    def foundation_stereo(self):
        processor = StereoInference("jiegouguang/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth")

        results = processor.infer(self.img1_rectify, self.img2_rectify, self.K1, abs(float(self.cam_t[0])))
        pcd_np = results['points_cam1'].astype(np.float64).reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_np)

        return pcd