import cv2
import numpy as np
import copy
import open3d as o3d
from scipy.optimize import curve_fit
from sklearn.neighbors import NearestNeighbors

from scipy.optimize import curve_fit

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

    def extract_circle(self):
        # centers_img1,img1_with_center = self.extract_circle_1(self.img1)
        # centers_img2,img2_with_center = self.extract_circle_1(self.img2)

        centers_img1,img1_with_center, _ = self.extract_circle_subpixel(self.img1,method='gaussian')
        centers_img2,img2_with_center, _ = self.extract_circle_subpixel(self.img2,method='gaussian')
        self.centers_img1 = centers_img1
        self.centers_img2 = centers_img2
        return img1_with_center,img2_with_center

    def extract_circle_1(self,img):

        H,W = img.shape
        mean_light = img.mean()
        binary_image = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.binary_image_threshold[0], self.binary_image_threshold[1]
        )
        
        contours_filtered = []
        for contour in cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            if cv2.contourArea(contour) < self.contourArea_threshold:
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
            if max_val < self.lightfilter_threshold:
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
    
    def extract_circle_subpixel(self, img, method='gaussian', visualize=False, save_path='subpixel_test'):
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
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.binary_image_threshold[0], self.binary_image_threshold[1]
        )

        # 连通域过滤
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
        comparison_images = {}
        if visualize:
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
            if local_patch.max() < self.lightfilter_threshold:
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


    def feature_extracting_1_dis(self,centers_img,n_neighbors = 5):
        centers_img = np.array(centers_img)
        # 如果没有点或点数量小于2，直接返回空数组
        if centers_img.size == 0 or centers_img.shape[0] < 2:
            return np.zeros((0, 0, 2))

        nbrs = NearestNeighbors(n_neighbors=min(n_neighbors, centers_img.shape[0]-1) + 1, algorithm='auto').fit(centers_img)
        distances, indices = nbrs.kneighbors(centers_img)
        # 排除自身（第一列）
        neighbors_idx = indices[:, 1:]
        neighbors_xy = centers_img[neighbors_idx]

        # 计算每个邻近点相对于当前点的坐标差：neighbor - point
        # centers_img 形状 (N,2) -> 扩展为 (N,1,2) 与 neighbors_xy (N,k,2) 广播
        diffs = neighbors_xy - centers_img[:, None, :]
        return diffs

    def feature_extracting_1(self, centers_img, img, patch_size=32):
        """为每个中心点计算 SIFT orb 描述子。

        参数:
            centers_img: (N,2) 数组或可转为 numpy 的中心点列表
            img: 灰度图像，若为 None 使用 self.img1_rectify
            patch_size: 用于判断边界的半边长

        返回:
            descriptors: (N,128) numpy 数组，无法计算的位置填 0
            valid_mask: (N,) 布尔数组，表示该点是否成功计算到描述子
        """
        centers = np.array(centers_img)

        # 确保为灰度单通道
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img


        # extracter = cv2.SIFT_create()
        extracter = cv2.ORB_create()

        keypoints = []
        valid_mask = np.ones((centers.shape[0],), dtype=bool)
        h, w = img_gray.shape[:2]
        half = int(patch_size // 2)
        for i, (x, y) in enumerate(centers):
            xi = int(round(x)); yi = int(round(y))
            # 边界检查：若关键点太靠近边界，标为无效（也可选择裁剪后计算）
            if xi - half < 0 or yi - half < 0 or xi + half >= w or yi + half >= h:
                valid_mask[i] = False
                # still create a keypoint to keep indexing consistent, but will have no descriptor
                keypoints.append(cv2.KeyPoint(float(xi), float(yi), float(patch_size)))
            else:
                # 切出 patch 并保存为图片
                # patch = img_gray[yi - half:yi + half, xi - half:xi + half]
                # save_path = f"test.png"
                # cv2.imwrite(save_path, patch)
                keypoints.append(cv2.KeyPoint(float(xi), float(yi), float(patch_size)))

        # 计算描述子
        kp, des = extracter.compute(img_gray, keypoints)

        # 如果没有返回描述子，返回空
        if des is None or len(kp) == 0:
            return np.zeros((0, 128), dtype=np.float32), np.zeros((0,2), dtype=np.float32)

        # 因为输入 keypoints 顺序通常与返回 kp 顺序一致，优先尝试直接一一对应
        coords_out = []
        descs_out = []
        if len(kp) == len(keypoints):
            # 直接对应
            for i_k, kp_i in enumerate(kp):
                xpt, ypt = kp_i.pt
                coords_out.append([xpt, ypt])
                descs_out.append(des[i_k])
        else:
            # 回退：基于坐标最近邻分配
            pts_out = np.array([kp_i.pt for kp_i in kp])
            for i_k, pt in enumerate(pts_out):
                dists = (centers[:, 0] - pt[0]) ** 2 + (centers[:, 1] - pt[1]) ** 2
                idx = int(np.argmin(dists))
                coords_out.append([centers[idx,0], centers[idx,1]])
                descs_out.append(des[i_k])

        coords_arr = np.array(coords_out, dtype=np.float32)
        descs_arr = np.array(descs_out, dtype=np.float32)
        return coords_arr, descs_arr



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

    def feature_extracting(self):
        centers_img1 = self.centers_img1
        centers_img2 = self.centers_img2

        coords_arr_img1, descs_arr_img1 = self.feature_extracting_1(centers_img1, self.img1_rectify,patch_size=32)
        coords_arr_img2, descs_arr_img2 = self.feature_extracting_1(centers_img2, self.img2_rectify,patch_size=32)
        # features_img1 = self.feature_extracting_1(centers_img1,n_neighbors = 10)
        # features_img2 = self.feature_extracting_1(centers_img2,n_neighbors = 10)


        self.feature_coords_img1 = coords_arr_img1
        self.feature_coords_img2 = coords_arr_img2
        self.features_img1 = descs_arr_img1
        self.features_img2 = descs_arr_img2


    def feature_matching_dis(self):

        y_tol = 3.0
        max_spatial_dist = 200

        features_img1 = getattr(self, 'features_img1', None)
        features_img2 = getattr(self, 'features_img2', None)
        centers_img1 = getattr(self, 'centers_img1', None)
        centers_img2 = getattr(self, 'centers_img2', None)

        if features_img1 is None or features_img2 is None:
            raise ValueError('请先运行 feature_extracting() 以获得 features_img1/2')

        # 扁平化描述子 (N, k*2)
        desc1 = features_img1.reshape(features_img1.shape[0], -1)
        desc2 = features_img2.reshape(features_img2.shape[0], -1)

        # 仅在 y 坐标接近的候选点中进行匹配（双目系统常在同一扫描线匹配）

        mutual_check = True

        matches = []

        dis_desc = []
        dis_xiangsu = []

        pts1 = np.array(centers_img1)
        pts2 = np.array(centers_img2)

        # 逐点查找候选（基于 y），再在候选集上做描述子最近邻
        # 新增限制：pts2 一定在 pts1 左边（x2 < x1）
        for i in range(desc1.shape[0]):
            x1 = pts1[i, 0]
            y1 = pts1[i, 1]
            # 候选索引
            candidates = (np.where((np.abs(pts2[:, 1] - y1) <= y_tol)) and np.where((pts2[:, 0] < x1)))[0]
            if candidates.size == 0:
                continue

            # 描述子维度可能不一致（如果邻居数或点数不同），对 desc2 子集进行相同 reshape
            desc1_i = desc1[i:i+1]
            desc2_c = desc2[candidates]

            # 用最近邻在候选集合上匹配
            nbrs_c = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(desc2_c)
            d_c, idx_c = nbrs_c.kneighbors(desc1_i)
            j_rel = int(idx_c[0, 0])
            j = int(candidates[j_rel])
            d_desc = float(d_c[0, 0])

            # 互检：检查 pts1 中 j 的最佳匹配是否为 i（在同样的候选策略下）
            if mutual_check:
                # 生成 j 在 pts2 对应的候选（基于 y）——这里用 pts1 中与 pts2[j] y 接近的集合
                y2 = pts2[j, 1]
                candidates_rev = np.where(np.abs(pts1[:, 1] - y2) <= y_tol)[0]
                if candidates_rev.size == 0:
                    continue
                desc2_j = desc2[j:j+1]
                desc1_rev = desc1[candidates_rev]
                nbrs_rev = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(desc1_rev)
                d_rev, idx_rev = nbrs_rev.kneighbors(desc2_j)
                i_rel = int(idx_rev[0, 0])
                i_back = int(candidates_rev[i_rel])
                if i_back != i:
                    continue

            # 空间距离过滤
            p1 = pts1[i]
            p2 = pts2[j]
            spatial_dist = float(np.linalg.norm(p1 - p2))
            if spatial_dist > max_spatial_dist:
                continue

            matches.append((i, j))
            dis_desc.append(d_desc)
            dis_xiangsu.append(spatial_dist)

            # 序号1 序号2 描述子距离 空间距离

        matches = np.array(matches)
        dis_desc = np.array(dis_desc)
        dis_xiangsu = np.array(dis_xiangsu)
        self.matches = matches
        self.dis_desc = dis_desc
        self.dis_xiangsu = dis_xiangsu




        idx1 = matches[:, 0].astype(int)
        idx2 = matches[:, 1].astype(int)

        kp1 = pts1[idx1]
        kp2 = pts2[idx2]

        # 保存为 float32，形状为 (N,2)。后续 triangulate_points 中使用 self.kp1_np.T 得到 (2,N)
        self.kp1_np = kp1.astype(np.float32)
        self.kp2_np = kp2.astype(np.float32)

        # 绘制匹配涂鸦并保存到 self.match_img
        img1_vis = getattr(self, 'img1_rectify', self.img1_rectify)
        img2_vis = getattr(self, 'img2_rectify', self.img2_rectify)

        # # 确保为 BGR 彩色图像
        # def _to_bgr(img):
        #     if img is None:
        #         return None
        #     if len(img.shape) == 2 or img.shape[2] == 1:
        #         return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #     return img.copy()

        img1_bgr = cv2.cvtColor(img1_vis, cv2.COLOR_GRAY2BGR)
        img2_bgr = cv2.cvtColor(img2_vis, cv2.COLOR_GRAY2BGR)

        # 横向拼接两图
        h_img = cv2.hconcat([img1_bgr, img2_bgr])
        offset_x = img1_bgr.shape[1]

        rng = np.random.RandomState(42)
        for (p1, p2) in zip(self.kp1_np, self.kp2_np):
            color = tuple(map(int, rng.randint(0, 256, 3)))
            pt1 = (int(round(p1[0])), int(round(p1[1])))
            pt2 = (int(round(p2[0])) + offset_x, int(round(p2[1])))
            cv2.circle(h_img, pt1, 3, color, -1)
            cv2.circle(h_img, pt2, 3, color, -1)
            cv2.line(h_img, pt1, pt2, color, 1)

        self.match_img = h_img

        return self.kp1_np, self.kp2_np

    def feature_matching(self, y_tol=1.0, mutual_check=True, ratio=1.0):
        """使用 SIFT ORB 描述子在两幅矫正图上进行受限匹配（仅在近似同一行匹配）。

        返回: self.kp1_np, self.kp2_np（匹配点坐标，float32）并保存可视化到 self.match_img
        """

        img1 = self.img1_rectify
        img2 = self.img2_rectify


        coords1 = self.feature_coords_img1
        coords2 = self.feature_coords_img2
        desc1 = self.features_img1
        desc2 = self.features_img2

        pts1 = np.array(coords1)
        pts2 = np.array(coords2)

        # 计算视差范围：disp = f * B / Z
        # max_disp for min_Z (closest), min_disp for max_Z (farthest)
        if hasattr(self, 'K1') and self.K1 is not None:
            f = float(self.K1[0,0])
        elif hasattr(self, 'P1') and self.P1 is not None:
            f = float(self.P1[0,0])
        else:
            raise ValueError('无法获取焦距 f')

        if hasattr(self, 'cam_t') and self.cam_t is not None:
            try:
                B = abs(float(self.cam_t[0][0]))
            except Exception:
                B = abs(float(np.array(self.cam_t).ravel()[0]))
        else:
            raise ValueError('无法获取基线 B')

        min_Z = getattr(self, 'min_dis', 100)
        max_Z = getattr(self, 'max_dis', 1500)
        max_disp = f * B / min_Z  # for closest points
        min_disp = f * B / max_Z  # for farthest points

        matches = []
        dis_desc = []
        dis_spatial = []

        # 逐点受限匹配（增加 ratio test）
        for i in range(desc1.shape[0]):
            x1, y1 = pts1[i]
            # 选取 y 接近且位于左图右方的候选（x2 < x1）
            candidates = np.where((np.abs(pts2[:,1] - y1) <= y_tol) & (pts2[:,0] < x1))[0]
            if candidates.size == 0:
                continue

            # 计算描述子距离并选最优/次优（用于 ratio test）
            if candidates.size == 1:
                # # 只有一个候选，无法做 ratio test：直接取该项但仍可通过 mutual_check
                # dists = np.linalg.norm(desc2[candidates] - desc1[i], axis=1)
                # j_rel = 0
                # d1 = float(dists[0])
                # d2 = np.inf
                continue
            else:
                dists = np.linalg.norm(desc2[candidates] - desc1[i], axis=1)
                # 找到最小与次小
                sorted_idx = np.argsort(dists)
                j_rel = int(sorted_idx[0])
                d1 = float(dists[sorted_idx[0]])
                d2 = float(dists[sorted_idx[1]])

            # ratio test
            if d1 >= ratio * d2:
                # 不满足 ratio test，跳过
                continue

            j = int(candidates[j_rel])
            d_desc = float(d1)

            # 互检：检查 j 在右图是否也把 i 作为最佳匹配
            if mutual_check:
                y2 = pts2[j,1]
                candidates_rev = np.where(np.abs(pts1[:,1] - y2) <= y_tol)[0]
                if candidates_rev.size == 0:
                    continue
                dists_rev = np.linalg.norm(desc1[candidates_rev] - desc2[j], axis=1)
                i_rel = int(np.argmin(dists_rev))
                i_back = int(candidates_rev[i_rel])
                if i_back != i:
                    continue

            # 计算视差并过滤范围
            disp = float(x1 - pts2[j,0])
            if disp < min_disp or disp > max_disp:
                continue

            # 空间距离过滤（可选，保留像素距离）
            spatial_dist = disp

            matches.append((i, j))
            dis_desc.append(d_desc)
            dis_spatial.append(spatial_dist)


        matches = np.array(matches)
        dis_desc = np.array(dis_desc)
        dis_spatial = np.array(dis_spatial)
        self.sift_matches = matches
        self.sift_dis_desc = dis_desc
        self.sift_dis_spatial = dis_spatial

        idx1 = matches[:,0].astype(int)
        idx2 = matches[:,1].astype(int)

        kp1 = pts1[idx1]
        kp2 = pts2[idx2]

        self.kp1_np = kp1.astype(np.float32)
        self.kp2_np = kp2.astype(np.float32)

        # 可视化匹配
        img1_bgr = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img2_bgr = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        h_img = cv2.hconcat([img1_bgr, img2_bgr])
        offset_x = img1_bgr.shape[1]
        rng = np.random.RandomState(42)
        for (p1, p2) in zip(self.kp1_np, self.kp2_np):
            color = tuple(map(int, rng.randint(0,256,3)))
            pt1 = (int(round(p1[0])), int(round(p1[1])))
            pt2 = (int(round(p2[0])) + offset_x, int(round(p2[1])))
            cv2.circle(h_img, pt1, 3, color, -1)
            cv2.circle(h_img, pt2, 3, color, -1)
            cv2.line(h_img, pt1, pt2, color, 1)

        self.match_img = h_img
        return self.kp1_np, self.kp2_np



    def draw_chess_board(self, img1, img2):
        # Find chessboard corners

        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        pattern_size = (11, 8)  # Adjust based on your chessboard size
        
        ret1, corners1 = cv2.findChessboardCorners(img1, pattern_size)
        ret2, corners2 = cv2.findChessboardCorners(img2, pattern_size)
        
        if ret1 and ret2:
            # Refine corner positions
            corners1 = cv2.cornerSubPix(img1, corners1, (11,11), (-1,-1), criteria)
            corners2 = cv2.cornerSubPix(img2, corners2, (11,11), (-1,-1), criteria)
            
            # Create combined image


            h_img = cv2.hconcat([img1, img2])
            h_img = cv2.cvtColor(h_img, cv2.COLOR_GRAY2BGR)
            
            # Draw corners and lines
            w = img1.shape[1]
            for i in range(len(corners1)):
                pt1 = (int(corners1[i][0][0]), int(corners1[i][0][1]))
                pt2 = (int(corners2[i][0][0]) + w, int(corners2[i][0][1]))
                cv2.circle(h_img, pt1, 3, (0,255,0), -1)
                cv2.circle(h_img, pt2, 3, (0,255,0), -1)
                cv2.line(h_img, pt1, pt2, (0,0,255), 1)
                
            return h_img
        
        return None



    def pointcloud_from_disparity(self, output_ply='test.ply', f=None, B=None, z_threshold=100, neighbor_radius=20):
        """通过视差计算深度并生成点云。

        Z = f * B / disparity

        参数:
            output_ply: 输出 ply 文件名
            f: 焦距（像素），如果为 None 则从 self.K1 或 self.P1 中读取
            B: 基线（米或与相机标定一致的单位），如果为 None 则从 self.cam_t 读取 t[0]
        要求: 已运行 feature_matching() 生成 self.kp1_np/self.kp2_np，并已运行 import_biaodin() 获得 self.K1/self.cam_t
        """


        if f is None:
            if hasattr(self, 'K1') and self.K1 is not None:
                f = float(self.K1[0, 0])
            elif hasattr(self, 'P1') and self.P1 is not None:
                f = float(self.P1[0, 0])
            else:
                raise ValueError('无法获取焦距 f，请传入 f 或先运行 import_biaodin()')

        if B is None:
            if hasattr(self, 'cam_t') and self.cam_t is not None:
                try:
                    B = abs(float(self.cam_t[0][0]))
                except Exception:
                    # 尝试展平
                    B = abs(float(np.array(self.cam_t).ravel()[0]))
            else:
                raise ValueError('无法获取基线 B，请传入 B 或先运行 import_biaodin()')

        # principal point

        cx = float(self.K1[0, 2])
        cy = float(self.K1[1, 2])

        kp1 = np.array(self.kp1_np)
        kp2 = np.array(self.kp2_np)


        pts_3d = []
        colors = []

        # First pass: compute Z for all matches and keep intermediate data for neighborhood check
        match_pixels = []  # list of (x1, y1)
        match_Zs = []
        match_colors = []

        for (p1, p2) in zip(kp1, kp2):
            x1, y1 = float(p1[0]), float(p1[1])
            x2, y2 = float(p2[0]), float(p2[1])
            disp = x1 - x2

            Z = f * B / disp
            if Z > self.max_dis or Z < self.min_dis:
                continue

            # color sampling (grayscale normalized)
            color_val = [1.0, 1.0, 1.0]

            match_pixels.append((x1, y1))
            match_Zs.append(float(Z))
            match_colors.append(color_val)

        # Neighborhood filtering: for each valid Z, compute median Z of neighbors within neighbor_radius pixels
        # 手动过滤的 可以去掉 主要是为了去掉深度快速变化的点
        N = len(match_Zs)
        for idx in range(N):
            Z = match_Zs[idx]
            if Z is None:
                continue
            x1, y1 = match_pixels[idx]
            # collect neighbor Zs (including self) within pixel radius
            neighbor_Zs = []
            for j in range(N):
                Zj = match_Zs[j]
                xj, yj = match_pixels[j]
                if abs(xj - x1) <= neighbor_radius and abs(yj - y1) <= neighbor_radius:
                    neighbor_Zs.append(Zj)

            if len(neighbor_Zs) == 0:
                # no neighbors with valid Z — treat as outlier and skip
                continue

            median_Z = float(np.median(np.array(neighbor_Zs)))
            if abs(Z - median_Z) > z_threshold:
                # depth differs too much from neighborhood — skip
                continue

            # keep point
            X = (x1 - cx) * Z / f
            Y = (y1 - cy) * Z / f
            pts_3d.append([X, Y, Z])
            colors.append(match_colors[idx])

        if len(pts_3d) == 0:
            print('no valid 3d points from disparity')
            return None


        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(pts_3d))
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

        o3d.io.write_point_cloud(output_ply, pcd)
        print(f'Wrote {len(pts_3d)} points to', output_ply)
        return pcd
