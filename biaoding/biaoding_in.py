import cv2
import numpy as np
import glob
import os

# 设置亚像素角点精度
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 棋盘格内角点规格（12x9格 -> 11x8角点）
pattern_size = (11, 8)

# 生成棋盘格三维点坐标（Z=0）
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

def calibrate_camera(image_path_pattern, save_prefix):
    """标定单个相机，并返回 mtx, dist, image_size"""
    obj_points = []
    img_points = []

    images = glob.glob(image_path_pattern)
    if len(images) == 0:
        raise FileNotFoundError(f"未找到图片: {image_path_pattern}")

    print(f"\n正在处理路径: {image_path_pattern} 共 {len(images)} 张图片")

    # 用于记录图像尺寸（所有图片应相同）
    img_size = None

    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        if img is None:
            print(f"  读取失败: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = gray.shape[::-1]  # (width, height)

        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        print(f"  {os.path.basename(fname)} -> ret={ret}")

        if ret:
            obj_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            img_points.append(corners2)
            cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
            # cv2.imwrite(f"{save_prefix}_corner_{i+1}.jpg", img)

    if len(img_points) == 0:
        raise RuntimeError(f"{save_prefix} 未检测到任何棋盘格角点。")

    # 标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)

    print(f"\n{save_prefix} 标定完成")
    print("ret:", ret)
    print("mtx:\n", mtx)
    print("dist:\n", dist)
    print("-----------------------------------------------------")

    # 去畸变示例
    img = cv2.imread(images[0])
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst1 = dst[y:y+h, x:x+w]
    # cv2.imwrite(f'{save_prefix}_undistort.jpg', dst1)
    print(f"{save_prefix} 去畸变结果保存为: {save_prefix}_undistort.jpg")
    print("-----------------------------------------------------\n")

    return mtx, dist, img_size


# ========================
# 分别标定左右相机
# ========================

mtx_right, dist_right, size_right = calibrate_camera(
    "d455_biaoding/right/*.png", "right"
)

mtx_left, dist_left, size_left = calibrate_camera(
    "d455_biaoding/left/*.png", "left"
)

print("两组相机标定结果：")
print("右相机内参矩阵:\n", mtx_right)
print("右相机畸变系数:\n", dist_right)
print("左相机内参矩阵:\n", mtx_left)
print("左相机畸变系数:\n", dist_left)


# ========================
# 将内参写入 YAML 文件
# ========================

yaml_file = "intrinsics.yaml"
fs = cv2.FileStorage(yaml_file, cv2.FILE_STORAGE_WRITE)

# 图像尺寸（确保左右一致）
assert size_left == size_right, "左右相机图像尺寸不一致！"
fs.write("image_width",  np.array([size_left[0]], dtype=np.int32))
fs.write("image_height", np.array([size_left[1]], dtype=np.int32))

# 左相机
fs.write("camera_matrix_left",  mtx_left)
fs.write("distortion_left",     dist_left)

# 右相机
fs.write("camera_matrix_right", mtx_right)
fs.write("distortion_right",    dist_right)

fs.release()

print(f"\n内参已成功写入: {os.path.abspath(yaml_file)}")