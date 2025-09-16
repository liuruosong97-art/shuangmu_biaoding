import cv2
import numpy as np

class JieGouGuang:
    def __init__(self,img1_path,img2_path):
        self.img1_path = img1_path
        self.img2_path = img2_path
        self.img1 = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2GRAY)
        self.img2 = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2GRAY)

    def extract_circle(self):
        centers_img1,img1_with_center = self.extract_circle_1(self.img1)
        centers_img2,img2_with_center = self.extract_circle_1(self.img2)
        return centers_img1,centers_img2,img1_with_center,img2_with_center

    def extract_circle_1(self,img):
        
        H,W = img.shape
        mean_light = img.mean()
        _, binary_image = cv2.threshold(img, mean_light * 2, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_with_center = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        center_extracted = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # 选择光斑周围的区域进行插值
            bbox_size = 5
            y_start = max(0, cY-bbox_size)
            y_end = min(H, cY+bbox_size)
            x_start = max(0, cX-bbox_size)
            x_end = min(W, cX+bbox_size)
            local_patch = img[y_start:y_end, x_start:x_end]
            # subpixel_x = np.arange(x_start-cX, x_end-cX)
            # subpixel_y = np.arange(y_start-cY, y_end-cY)
            # X, Y = np.meshgrid(subpixel_x, subpixel_y)

            max_pos = np.unravel_index(np.argmax(local_patch), local_patch.shape)
            subpixel_center = (cX - bbox_size + max_pos[1], cY - bbox_size + max_pos[0])
            # subpixel_center = (cX, cY)
            # 在img上绘制subpixel_center点
            
            cv2.circle(img_with_center, (int(subpixel_center[0]), int(subpixel_center[1])), 1, (0,0,255), -1)
            center_extracted.append([int(subpixel_center[0]), int(subpixel_center[1])])
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

        # 计算极线变换矩阵
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(new_M1, D1, new_M2, D2, (w, h), R, t, flags=cv2.CALIB_ZERO_TANGENT_DIST)

        # 生成映射
        left_map_x, left_map_y = cv2.initUndistortRectifyMap(new_M1, D1, R1, P1, (w, h), cv2.CV_32FC1)
        right_map_x, right_map_y = cv2.initUndistortRectifyMap(new_M2, D2, R2, P2, (w, h), cv2.CV_32FC1)

        # 进行矫正
        left_rectified = cv2.remap(self.img1, left_map_x, left_map_y, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(self.img2, right_map_x, right_map_y, cv2.INTER_LINEAR)

        self.img1_rectify = left_rectified
        self.img2_rectify = right_rectified

