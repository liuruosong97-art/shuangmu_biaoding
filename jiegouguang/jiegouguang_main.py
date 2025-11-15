import cv2
import os
import numpy as np

from jiegouguang_class import JieGouGuang
import open3d as o3d
import time

img = cv2.imread('d455_jiegouguang_save/better/left2.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
H,W = img.shape

jiegouguang_class = JieGouGuang('d455_jiegouguang_save/20251103/left3.png','d455_jiegouguang_save/20251103/right3.png') # 一个杯子
# jiegouguang_class = JieGouGuang('d455_jiegouguang_save/better/left2.png','d455_jiegouguang_save/better/right2.png') # 平面场景
# jiegouguang_class = JieGouGuang('d455_jiegouguang_save/left2.png','d455_jiegouguang_save/right2.png') # 复杂场景 zyb桌子

jiegouguang_class.import_biaodin('biaoding/extrinsics_d455_20250915.yml','biaoding/intrinsics_d455_20250915.yml')

# jiegouguang_class.method = 'sgbm'
# jiegouguang_class.method = 'foundation_stereo'
jiegouguang_class.method = 'bridgedepth'
jiegouguang_class.init_model()

start = time.time()
disparity_raw = jiegouguang_class.forward_disparity()
print(f"\033[31mCosting time (s): {time.time() - start}\033[0m")


# 下面这两种是稀疏方法 暂时弃用
# pcd = jiegouguang_class.manual_feature_extracting()
# pcd = jiegouguang_class.lg_feature_extracting()


depth = (float(jiegouguang_class.K1[0, 0]) * abs(float(jiegouguang_class.cam_t[0]))) / disparity_raw
depth = np.clip(depth, jiegouguang_class.min_dis, jiegouguang_class.max_dis)
pcd = jiegouguang_class.depth2pointcloud(depth)


# jiegouguang_class.cal_error(pcd)


# pcd =  pcd.voxel_down_sample(voxel_size=10)
pcd.colors = o3d.utility.Vector3dVector(np.repeat([[1.0, 0.0, 0.0]], len(pcd.points), axis=0))
o3d.io.write_point_cloud("test.ply", pcd)

# print(f"Pointcloud shape: {len(pcd.points)}")





# cv2.imwrite("test2.png", img2_with_center)
print("end")

