import cv2
import numpy as np
import copy
import open3d as o3d
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


    def feature_matching(self):
        stereo = cv2.StereoBM_create(numDisparities=16*6, blockSize=15)
        disparity = stereo.compute(self.img1_rectify, self.img2_rectify)
        points_3d = cv2.reprojectImageTo3D(disparity, self.Q)

        # 获取有效视差点
        mask = disparity < 300
        points = points_3d[mask]
        colors = cv2.cvtColor(self.img1_rectify, cv2.COLOR_GRAY2BGR)[mask] / 255.0

        # 创建Open3D点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # 保存为.ply文件
        o3d.io.write_point_cloud("test.ply", pcd)
        # cv2.imwrite('test.png', img_matches)
        # Convert keypoints to numpy arrays


        return disparity
    

    def triangulate_points(self):
        
        pts1 = self.kp1_np.T
        pts2 = self.kp2_np.T
        # points_4d_hom = cv2.triangulatePoints(P1, P2, pts1, pts2)

        pass

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

