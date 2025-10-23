import cv2
import numpy as np
import glob
import os

class StereoResult:
    def __init__(self):
        self.m1 = None
        self.m2 = None
        self.d1 = None
        self.d2 = None
        self.R = None
        self.T = None

stereo = StereoResult()

class StereoCalibration:
    def __init__(self, left_dir='camL', right_dir='camR', pattern=(11,8),
                 square_size=25.0, use_circles=False):
        """
        left_dir, right_dir: folders containing left/right images
        pattern: (cols, rows) = 内角点数量，比如 11x8
        square_size: 物理方格/圆心间距（任意单位）
        use_circles: 如果你的标定板是圆点阵，设置 True，否则使用棋盘格
        """
        self.left_dir = left_dir
        self.right_dir = right_dir
        self.pattern = pattern
        self.square_size = square_size
        self.use_circles = use_circles

        self.imagesL = self.read_images(self.left_dir)
        self.imagesR = self.read_images(self.right_dir)

        print("Left images:", len(self.imagesL))
        print("Right images:", len(self.imagesR))

    def read_images(self, cal_path):
        # 支持多种后缀
        patterns = ['/*.png', '/*.jpg', '/*.jpeg', '/*.bmp', '/*.PNG', '/*.JPG']
        files = []
        for p in patterns:
            files += glob.glob(cal_path + p)
        files.sort()
        return files

    def calibration_photo(self):
        x_nums, y_nums = self.pattern

        # world points (one board)
        objp = np.zeros((x_nums * y_nums, 3), np.float32)
        objp[:, :2] = np.mgrid[0:x_nums, 0:y_nums].T.reshape(-1, 2) * self.square_size

        objpoints = []
        imgpointsL = []
        imgpointsR = []

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        n_pairs = min(len(self.imagesL), len(self.imagesR))
        if n_pairs == 0:
            raise RuntimeError("❌ 未找到左右图像，请检查路径和扩展名。")

        print(f"Processing {n_pairs} pairs (will skip pairs where detection fails)...")

        for ii in range(n_pairs):
            left_path = self.imagesL[ii]
            right_path = self.imagesR[ii]

            imgL = cv2.imread(left_path)
            imgR = cv2.imread(right_path)
            if imgL is None or imgR is None:
                print(f"❌ 读取失败: {left_path} or {right_path}, 跳过")
                continue

            grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

            if self.use_circles:
                # 对圆点阵，可能需要 SimpleBlobDetector 参数，但先尝试默认
                flags = cv2.CALIB_CB_SYMMETRIC_GRID
                okL, cornersL = cv2.findCirclesGrid(grayL, (x_nums, y_nums), None, flags=flags)
                okR, cornersR = cv2.findCirclesGrid(grayR, (x_nums, y_nums), None, flags=flags)
            else:
                # 棋盘格
                flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
                okL, cornersL = cv2.findChessboardCorners(grayL, (x_nums, y_nums), flags)
                okR, cornersR = cv2.findChessboardCorners(grayR, (x_nums, y_nums), flags)

            print(f"[{ii}] {os.path.basename(left_path)} | {os.path.basename(right_path)} -> L:{okL} R:{okR}")

            if okL and okR:
                # refine to subpixel
                cornersL2 = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
                cornersR2 = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)

                objpoints.append(objp)
                imgpointsL.append(cornersL2)
                imgpointsR.append(cornersR2)

                # 可视化保存（可选）
                dispL = imgL.copy()
                dispR = imgR.copy()
                if self.use_circles:
                    cv2.drawChessboardCorners(dispL, (x_nums, y_nums), cornersL2, okL)
                    cv2.drawChessboardCorners(dispR, (x_nums, y_nums), cornersR2, okR)
                else:
                    cv2.drawChessboardCorners(dispL, (x_nums, y_nums), cornersL2, okL)
                    cv2.drawChessboardCorners(dispR, (x_nums, y_nums), cornersR2, okR)

                cv2.imwrite(f"corner_L_{ii+1}.png", dispL)
                cv2.imwrite(f"corner_R_{ii+1}.png", dispR)

        print("Valid pairs found:", len(objpoints))
        if len(objpoints) < 3:
            raise RuntimeError("❌ 有效的角点对太少，无法进行标定（至少 3 对，推荐 10+ 对）。")

        image_shape = grayL.shape[::-1]

        # 单目标定（可选择跳过单目标定，直接使用 stereoCalibrate 同时优化）
        retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, image_shape, None, None)
        retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, image_shape, None, None)

        print("Left intrinsic:\n", mtxL)
        print("Right intrinsic:\n", mtxR)

        # stereoCalibrate：如果想固定内参，使用 CALIB_FIX_INTRINSIC，否则用 flags=0 来联合优化
        flags = 0  # 推荐先联合优化
        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        retS, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpointsL, imgpointsR,
            mtxL, distL, mtxR, distR,
            image_shape,
            criteria=stereocalib_criteria, flags=flags
        )

        print("Stereo R:\n", R)
        print("Stereo T:\n", T)

        # 保存结果到全局对象
        stereo.m1 = CM1; stereo.d1 = dist1
        stereo.m2 = CM2; stereo.d2 = dist2
        stereo.R = R; stereo.T = T

        # 立体校正并保存 YAML（OpenCV FileStorage）
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(CM1, dist1, CM2, dist2, image_shape, R, T, alpha=0)
        fs = cv2.FileStorage("stereo_result.yaml", cv2.FILE_STORAGE_WRITE)
        fs.write("R", R)
        fs.write("T", T)
        fs.write("Rw", np.eye(3))
        fs.write("Tw", np.zeros((3,1)))
        fs.write("R1", R1)
        fs.write("R2", R2)
        fs.write("P1", P1)
        fs.write("P2", P2)
        fs.write("Q", Q)
        fs.write("F", F)
        fs.release()

        print("✅ Done. Results saved to stereo_result.yaml")
        return

if __name__ == "__main__":
    # 根据你实际情况设置 left_dir/right_dir, pattern 和 use_circles
    # 如果你的图片在 /home/lrs/xiangji/d455_biaoding/left/*.png 等，请写绝对路径
    cal = StereoCalibration(
        left_dir='/home/lrs/biaoding/shuangmu_biaoding/biaoding/left',
        right_dir='/home/lrs/biaoding/shuangmu_biaoding/biaoding/right',
        pattern=(11,8),          # 11x8 内角点（你说 12x9 格 -> 11x8）
        square_size=25.0,
        use_circles=False       # 若是真正的圆点阵改为 True
    )
    cal.calibration_photo()