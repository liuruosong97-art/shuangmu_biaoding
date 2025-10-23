#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import glob
import os
import re

# ==================== 参数 ====================
PATTERN_SIZE = (11, 8)
SQUARE_SIZE  = 25.0
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

LEFT_GLOB  = "/home/lrs/biaoding/shuangmu_biaoding/biaoding/left/*.png"
RIGHT_GLOB = "/home/lrs/biaoding/shuangmu_biaoding/biaoding/*.png"
YAML_OUT   = "stereo_calib.yaml"

# ==================== 提取编号函数 ====================
def extract_number(filename):
    """从文件名中提取数字编号（如 left_001.png → 1）"""
    basename = os.path.basename(filename)
    match = re.search(r'(\d+)', basename)
    return int(match.group(1)) if match else None

# ==================== 单目标定 ====================
def calibrate_camera(image_glob, prefix):
    images = sorted(glob.glob(image_glob))
    if not images:
        raise FileNotFoundError(f"未找到图片: {image_glob}")

    print(f"\n=== {prefix.upper()} 相机标定 ===")
    print(f"  共 {len(images)} 张图片")

    objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

    objpoints = []
    imgpoints = []
    img_size  = None
    good_files = []   # (number, path, objp, corners)

    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        if img is None:
            print(f"  读取失败: {fname}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(
            gray, PATTERN_SIZE,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

        num = extract_number(fname)
        status = 'OK' if ret and num is not None else 'FAIL'
        print(f"  {os.path.basename(fname):30} -> {status} (num={num})")

        if ret and num is not None:
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), SUBPIX_CRITERIA)
            good_files.append((num, fname, objp.copy(), corners2))

            vis = img.copy()
            cv2.drawChessboardCorners(vis, PATTERN_SIZE, corners2, ret)
            cv2.imwrite(f"{prefix}_corner_{idx:03d}.jpg", vis)

    if len(good_files) == 0:
        raise RuntimeError(f"{prefix} 未检测到任何角点")

    # 按编号排序
    good_files.sort(key=lambda x: x[0])

    # 解包
    numbers     = [x[0] for x in good_files]
    objpoints   = [x[2] for x in good_files]
    imgpoints   = [x[3] for x in good_files]
    file_paths  = [x[1] for x in good_files]

    # ---------- 标定 ----------
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None)

    # ---------- 重投影误差 ----------
    mean_err = 0
    for i in range(len(objpoints)):
        pts2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        err = cv2.norm(imgpoints[i], pts2, cv2.NORM_L2) / len(pts2)
        mean_err += err
    mean_err /= len(objpoints)

    print(f"\n{prefix.upper()} 标定结果")
    print(f"  重投影误差 : {mean_err:.4f} 像素")
    print(f"  内参矩阵 mtx :\n{mtx}")
    print(f"  畸变系数 dist:\n{dist}")

    # ---------- 去畸变示例 ----------
    sample = cv2.imread(images[0])
    h, w = sample.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undist = cv2.undistort(sample, mtx, dist, None, newcameramtx)
    x, y, rw, rh = roi
    undist = undist[y:y+rh, x:x+rw]
    cv2.imwrite(f"{prefix}_undistort.jpg", undist)
    print(f"  去畸变示例已保存 -> {prefix}_undistort.jpg\n")

    return mtx, dist, rvecs, tvecs, img_size, objpoints, imgpoints, numbers, file_paths


# ==================== 分别标定 ====================
mtxR, distR, _, _, img_size, objR, imgR, numR, pathR = calibrate_camera(RIGHT_GLOB, "right")
mtxL, distL, _, _, _,       objL, imgL, numL, pathL = calibrate_camera(LEFT_GLOB,  "left")

# ==================== 按编号取交集 ====================
common_nums = sorted(set(numL) & set(numR))
print(f"\n交集图片数量: {len(common_nums)} (左: {len(numL)}, 右: {len(numR)})")

if len(common_nums) < 10:
    print("可用的共同编号:", common_nums)
    raise RuntimeError("交集图片太少 (<10)，请检查左右图片编号是否一致。")

# 构建配对数据
objpoints  = []
imgpointsL = []
imgpointsR = []

for n in common_nums:
    l_idx = numL.index(n)
    r_idx = numR.index(n)
    objpoints.append(objL[l_idx])
    imgpointsL.append(imgL[l_idx])
    imgpointsR.append(imgR[r_idx])

# ==================== 立体标定 ====================
print("\n=== 立体标定 ===")
flags = cv2.CALIB_FIX_INTRINSIC
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR,
    mtxL, distL, mtxR, distR,
    img_size, criteria=criteria, flags=flags)

print(f"立体标定返回值 ret = {ret:.6f}")
print(f"R (旋转矩阵):\n{R}")
print(f"T (平移向量):\n{T}")

# ==================== 立体校正 ====================
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    mtxL, distL, mtxR, distR, img_size, R, T, alpha=0)

# ==================== 写入 YAML ====================
fs = cv2.FileStorage(YAML_OUT, cv2.FILE_STORAGE_WRITE)
fs.write("image_width",  np.array([img_size[0]], dtype=np.int32))
fs.write("image_height", np.array([img_size[1]], dtype=np.int32))
fs.write("camera_matrix_left",  mtxL)
fs.write("distortion_left",     distL)
fs.write("camera_matrix_right", mtxR)
fs.write("distortion_right",    distR)
fs.write("R", R)
fs.write("T", T)
fs.write("R1", R1)
fs.write("R2", R2)
fs.write("P1", P1)
fs.write("P2", P2)
fs.write("Q",  Q)
fs.write("E", E)
fs.write("F", F)
fs.release()

print(f"\n所有标定参数已写入: {os.path.abspath(YAML_OUT)}")