import cv2
import numpy as np
import glob
import os

# ======================
# 棋盘格与角点检测参数
# ======================
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
pattern_size = (11, 8)

# 生成棋盘格三维点坐标（Z=0）
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)


def calibrate_camera(image_path_pattern, save_prefix):
    """标定单个相机，并返回 M, D, image_size"""
    obj_points = []
    img_points = []

    images = glob.glob(image_path_pattern)
    if len(images) == 0:
        raise FileNotFoundError(f"未找到图片: {image_path_pattern}")

    print(f"\n📷 正在处理路径: {image_path_pattern} 共 {len(images)} 张图片")

    img_size = None

    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        if img is None:
            print(f"❌ 读取失败: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        print(f"  {os.path.basename(fname)} -> ret={ret}")

        if ret:
            obj_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            img_points.append(corners2)
            cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
            cv2.imwrite(f"{save_prefix}_corner_{i+1}.jpg", img)

    if len(img_points) == 0:
        raise RuntimeError(f"❌ {save_prefix} 未检测到任何棋盘格角点。")

    # 相机标定
    ret, M, D, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)

    print(f"\n✅ {save_prefix} 标定完成")
    print("ret:", ret)
    print("M:\n", M)
    print("D:\n", D)
    print("-----------------------------------------------------")

    # 去畸变示例
    img = cv2.imread(images[0])
    h, w = img.shape[:2]
    newM, roi = cv2.getOptimalNewCameraMatrix(M, D, (w, h), 1, (w, h))
    dst = cv2.undistort(img, M, D, None, newM)
    x, y, w, h = roi
    dst1 = dst[y:y+h, x:x+w]
    cv2.imwrite(f"{save_prefix}_undistort.jpg", dst1)
    print(f"✅ {save_prefix} 去畸变结果保存为: {save_prefix}_undistort.jpg")
    print("-----------------------------------------------------\n")

    return M, D, img_size


# ======================
# 分别标定左右相机
# ======================

M2, D2, size_right = calibrate_camera("/home/lrs/biaoding/shuangmu_biaoding/biaoding/right/*.png", "right")  # 右相机
M1, D1, size_left  = calibrate_camera("/home/lrs/biaoding/shuangmu_biaoding/biaoding/left/*.png",  "left")   # 左相机

print("🎯 两组相机标定结果：")
print("M1 (左相机内参矩阵):\n", M1)
print("D1 (左相机畸变系数):\n", D1)
print("M2 (右相机内参矩阵):\n", M2)
print("D2 (右相机畸变系数):\n", D2)


# ======================
# 写入 YAML 文件
# ======================

yaml_file = "intrinsics.yaml"
fs = cv2.FileStorage(yaml_file, cv2.FILE_STORAGE_WRITE)

assert size_left == size_right, "左右相机图像尺寸不一致！"
fs.write("image_width",  np.array([size_left[0]], dtype=np.int32))
fs.write("image_height", np.array([size_left[1]], dtype=np.int32))

# 按 OpenCV Stereo 格式写入
fs.write("M1", M1)
fs.write("D1", D1)
fs.write("M2", M2)
fs.write("D2", D2)

fs.release()

print(f"\n✅ 内参已成功写入: {os.path.abspath(yaml_file)}")
