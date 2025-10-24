import cv2
import numpy as np
import copy
import open3d as o3d
from manual_feature.ManualFeature import ManualFeatureJieGouGuang
from NeuralFeature import NeuralFeatureJieGouGuang

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

    def neural_feature_extracting(self):
        """
        使用神经网络（SuperPoint + LightGlue）进行特征提取和匹配

        Returns:
            pcd: Open3D 点云对象
        """
        processor = NeuralFeatureJieGouGuang(base=self)

        img1_with_center, img2_with_center = processor.extract_circle()
        processor.feature_extracting()
        processor.feature_matching()
        pcd = processor.pointcloud_from_disparity()

        return pcd

    def extract_circle_1(self, img, use_subpixel=False):
        """
        从图像中提取圆心位置（用于神经网络特征提取）

        Args:
            img: 灰度图像
            use_subpixel: 是否使用亚像素精度检测（默认False）

        Returns:
            center_extracted: 圆心坐标数组 (N, 2)
            img_with_center: 标记了圆心的图像
        """
        H, W = img.shape
        binary_image = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            self.binary_image_threshold[0], self.binary_image_threshold[1]
        )

        # 只保留小面积连通域
        contours_filtered = []
        for contour in cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            if cv2.contourArea(contour) < self.contourArea_threshold:
                contours_filtered.append(contour)
        contours = contours_filtered

        # 重新绘制二值化图
        binary_image_filtered = np.zeros_like(binary_image)
        cv2.drawContours(binary_image_filtered, contours, -1, 255, -1)
        binary_image = binary_image_filtered

        img_with_center = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        center_extracted = []

        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # 选择光斑周围的区域
            bbox_size = 5 if use_subpixel else 3  # 亚像素需要更大窗口
            y_start = max(0, cY - bbox_size)
            y_end = min(H, cY + bbox_size)
            x_start = max(0, cX - bbox_size)
            x_end = min(W, cX + bbox_size)
            local_patch = img[y_start:y_end, x_start:x_end]

            # 计算最亮点坐标
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(local_patch)

            # 亮度过滤（如果设置了阈值）
            if self.lightfilter_threshold > 0 and max_val < self.lightfilter_threshold:
                continue

            if use_subpixel:
                # 亚像素精度 - 高斯拟合法
                from scipy.optimize import curve_fit
                try:
                    patch_float = local_patch.astype(np.float64)
                    patch_h, patch_w = patch_float.shape

                    # 2D高斯函数
                    def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, offset):
                        x, y = xy
                        xo = float(xo)
                        yo = float(yo)
                        g = offset + amplitude * np.exp(
                            -(((x - xo) ** 2) / (2 * sigma_x ** 2) +
                              ((y - yo) ** 2) / (2 * sigma_y ** 2))
                        )
                        return g.ravel()

                    # 创建网格
                    x = np.arange(0, patch_w)
                    y = np.arange(0, patch_h)
                    x, y = np.meshgrid(x, y)

                    # 初始参数估计
                    initial_guess = (
                        patch_float.max() - patch_float.min(),
                        patch_w / 2,
                        patch_h / 2,
                        1.0, 1.0,
                        patch_float.min()
                    )

                    # 高斯拟合
                    popt, _ = curve_fit(
                        gaussian_2d, (x, y), patch_float.ravel(),
                        p0=initial_guess, maxfev=5000
                    )

                    # 亚像素中心
                    center_x = x_start + popt[1]
                    center_y = y_start + popt[2]
                except:
                    # 拟合失败，回退到最亮点法
                    center_x = x_start + max_loc[0]
                    center_y = y_start + max_loc[1]
            else:
                # 像素级精度 - 最亮点法
                center_x = x_start + max_loc[0]
                center_y = y_start + max_loc[1]

            cv2.circle(img_with_center, (int(center_x), int(center_y)), 1, (0, 0, 255), -1)
            center_extracted.append([center_x, center_y])

        return np.array(center_extracted), img_with_center