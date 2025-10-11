import cv2
import numpy as np
import copy
from scipy.optimize import curve_fit

class JieGouGuang:
    def __init__(self,img1_path,img2_path):
        self.img1_path = img1_path
        self.img2_path = img2_path
        self.img1 = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2GRAY)
        self.img2 = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2GRAY)

    def extract_circle(self):
        centers_img1,img1_with_center = self.extract_circle_1(self.img1)
        centers_img2,img2_with_center = self.extract_circle_1(self.img2)

        self.centers_img1 = centers_img1
        self.centers_img2 = centers_img2
        return img1_with_center,img2_with_center

    def extract_circle_1(self,img):

        H,W = img.shape
        mean_light = img.mean()
        binary_image = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, -4
        )
        # 只保留少于30个像素的连通域
        contours_filtered = []
        for contour in cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            if cv2.contourArea(contour) < 20:
                contours_filtered.append(contour)
        contours = contours_filtered

        # 重新绘制二值化图，只保留筛选后的连通域
        binary_image_filtered = np.zeros_like(binary_image)
        cv2.drawContours(binary_image_filtered, contours, -1, 255, -1)
        binary_image = binary_image_filtered

        # _, binary_image = cv2.threshold(img, mean_light * 2, 255, cv2.THRESH_BINARY)
        # contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_with_center = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        center_extracted = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # 选择光斑周围的区域进行插值
            bbox_size = 3
            y_start = max(0, cY-bbox_size)
            y_end = min(H, cY+bbox_size)
            x_start = max(0, cX-bbox_size)
            x_end = min(W, cX+bbox_size)
            local_patch = img[y_start:y_end, x_start:x_end]

            # 将local_patch放大5倍
            beishu = 20
            patch_resized = cv2.resize(local_patch, (local_patch.shape[1]*beishu, local_patch.shape[0]*beishu))
            # 计算放大后中心点坐标
            # 先计算local_patch中最亮点的坐标
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(local_patch)
            if max_val < 150:
                continue
            # 放大坐标
            max_loc_resized = (max_loc[0] * beishu, max_loc[1] * beishu)
            # 在放大后的patch上绘制中心
            cv2.circle(patch_resized, max_loc_resized, 3, (0, 0, 255), -1)
            # cv2.imwrite("test.png", patch_resized)
            # cv2.imwrite("local_patch.png", local_patch)
            center_x = x_start + max_loc[0] // 5
            center_y = y_start + max_loc[1] // 5
            # 在放大后的patch上绘制中心
            cv2.circle(patch_resized, (max_loc[0], max_loc[1]), 3, (0, 0, 255), -1)
            # 保存或显示patch_resized可选
            # cv2.imwrite("patch_resized.png", patch_resized)

            # 直接取局部区域最亮的点作为中心
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(local_patch)
            if max_val<150:
                continue
            center_x = x_start + max_loc[0]
            center_y = y_start + max_loc[1]
            patch_resized = cv2.cvtColor(local_patch, cv2.COLOR_GRAY2BGR)
            cx, cy = center_x, center_y
            # cv2.circle(patch_resized, (int(max_loc[0]), int(max_loc[1])), 1, (0, 0, 255), -1)
            # cv2.imwrite("test.png", patch_resized)
            cv2.circle(img_with_center, (cx, cy), 1, (0,0,255), -1)
            # cv2.circle(img_with_center, (int(subpixel_center[0]), int(subpixel_center[1])), 3, (0,0,255), -1)

            center_extracted.append([cx,cy])
        return np.array(center_extracted), img_with_center

    def extract_circle_subpixel(self, img, method='gaussian', visualize=True, save_path='subpixel_test'):
        """
        亚像素精度圆心检测方法 - 保留两种方法

        Args:
            img: 输入灰度图像
            method: 亚像素方法 ('gaussian', 'centroid', 'all')
            visualize: 是否生成20倍放大可视化
            save_path: 可视化保存路径

        Returns:
            centers: 亚像素精度的圆心坐标 [(x, y), ...]
            img_with_center: 标记了圆心的图像
            comparison_images: 各方法对比的放大图像字典
        """
        import os
        from scipy.optimize import curve_fit

        H, W = img.shape
        mean_light = img.mean()

        # 自适应阈值二值化
        binary_image = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, -4
        )

        # 连通域过滤
        contours_filtered = []
        for contour in cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            if cv2.contourArea(contour) < 20:
                contours_filtered.append(contour)
        contours = contours_filtered

        # 重新绘制二值化图
        binary_image_filtered = np.zeros_like(binary_image)
        cv2.drawContours(binary_image_filtered, contours, -1, 255, -1)
        binary_image = binary_image_filtered

        img_with_center = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        center_extracted = []
        comparison_images = {}

        os.makedirs(save_path, exist_ok=True)

        spot_idx = 0
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue

            # 粗略中心
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # 提取局部patch
            bbox_size = 5  # 更大的窗口用于高斯拟合
            y_start = max(0, cY - bbox_size)
            y_end = min(H, cY + bbox_size)
            x_start = max(0, cX - bbox_size)
            x_end = min(W, cX + bbox_size)
            local_patch = img[y_start:y_end, x_start:x_end].astype(np.float64)

            # 亮度过滤
            if local_patch.max() < 150:
                continue

            patch_h, patch_w = local_patch.shape

            # ========== 方法1: 二维高斯拟合 (Gaussian Fitting) ==========
            if method in ['gaussian', 'all']:
                try:
                    # 定义2D高斯函数
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
                        local_patch.max() - local_patch.min(),  # amplitude
                        patch_w / 2,  # xo
                        patch_h / 2,  # yo
                        1.0,  # sigma_x
                        1.0,  # sigma_y
                        local_patch.min()  # offset
                    )

                    # 拟合
                    popt, _ = curve_fit(
                        gaussian_2d,
                        (x, y),
                        local_patch.ravel(),
                        p0=initial_guess,
                        maxfev=5000
                    )

                    # 提取亚像素中心
                    subpixel_x_gaussian = x_start + popt[1]
                    subpixel_y_gaussian = y_start + popt[2]

                    if method == 'gaussian':
                        cx, cy = subpixel_x_gaussian, subpixel_y_gaussian
                        method_name = 'Gaussian'
                except:
                    # 拟合失败，回退到最亮点法
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(local_patch)
                    subpixel_x_gaussian = x_start + max_loc[0]
                    subpixel_y_gaussian = y_start + max_loc[1]
                    if method == 'gaussian':
                        cx, cy = subpixel_x_gaussian, subpixel_y_gaussian
                        method_name = 'Gaussian (fallback)'

            # ========== 方法2: 亮度加权质心法 (Weighted Centroid) ==========
            if method in ['centroid', 'all']:
                # 归一化亮度作为权重
                weights = local_patch - local_patch.min()
                weights = weights / weights.sum()

                # 计算加权质心
                x_indices = np.arange(patch_w)
                y_indices = np.arange(patch_h)
                x_grid, y_grid = np.meshgrid(x_indices, y_indices)

                subpixel_x_centroid = x_start + np.sum(x_grid * weights)
                subpixel_y_centroid = y_start + np.sum(y_grid * weights)

                if method == 'centroid':
                    cx, cy = subpixel_x_centroid, subpixel_y_centroid
                    method_name = 'Weighted Centroid'

            # ========== 方法对比模式 ==========
            if method == 'all':
                # 使用高斯拟合结果
                cx, cy = subpixel_x_gaussian, subpixel_y_gaussian
                method_name = 'All Methods'

            # 绘制圆心
            cv2.circle(img_with_center, (int(cx), int(cy)), 1, (0, 0, 255), -1)
            center_extracted.append([cx, cy])

            # ========== 生成20倍放大对比图 ==========
            if visualize and spot_idx < 5:  # 只保存前5个光斑
                beishu = 20
                patch_resized = cv2.resize(local_patch,
                                          (local_patch.shape[1] * beishu,
                                           local_patch.shape[0] * beishu),
                                          interpolation=cv2.INTER_CUBIC)

                # 归一化显示
                patch_vis = cv2.normalize(patch_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                patch_vis = cv2.cvtColor(patch_vis, cv2.COLOR_GRAY2BGR)

                # 在放大图上标记不同方法的结果
                if method == 'all':
                    # 高斯拟合 - 红色
                    local_x_g = (subpixel_x_gaussian - x_start) * beishu
                    local_y_g = (subpixel_y_gaussian - y_start) * beishu
                    cv2.circle(patch_vis, (int(local_x_g), int(local_y_g)), 3, (0, 0, 255), -1)
                    cv2.putText(patch_vis, 'G', (int(local_x_g) + 5, int(local_y_g)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    # 质心法 - 绿色
                    local_x_c = (subpixel_x_centroid - x_start) * beishu
                    local_y_c = (subpixel_y_centroid - y_start) * beishu
                    cv2.circle(patch_vis, (int(local_x_c), int(local_y_c)), 3, (0, 255, 0), -1)
                    cv2.putText(patch_vis, 'C', (int(local_x_c) + 5, int(local_y_c)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # 添加图例
                    legend_y = 20
                    cv2.putText(patch_vis, 'G:Gaussian C:Centroid',
                               (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                else:
                    # 单一方法 - 红色标记
                    local_x = (cx - x_start) * beishu
                    local_y = (cy - y_start) * beishu
                    cv2.circle(patch_vis, (int(local_x), int(local_y)), 3, (0, 0, 255), -1)
                    cv2.putText(patch_vis, method_name, (10, 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # 保存放大图
                save_file = os.path.join(save_path, f'spot_{spot_idx:03d}_{method}_20x.png')
                cv2.imwrite(save_file, patch_vis)
                comparison_images[f'spot_{spot_idx}'] = patch_vis

                spot_idx += 1

        return np.array(center_extracted), img_with_center, comparison_images

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