import cv2
import os
import numpy as np

from jiegouguang_class import JieGouGuang
import open3d as o3d

img = cv2.imread('d455_jiegouguang_save/better/left2.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
H,W = img.shape

# jiegouguang_class = JieGouGuang('d455_jiegouguang_save/20251103/left3.png','d455_jiegouguang_save/20251103/right3.png') # 一个杯子
jiegouguang_class = JieGouGuang('d455_jiegouguang_save/better/left3.png','d455_jiegouguang_save/better/right3.png') # 平面场景

# jiegouguang_class = JieGouGuang('d455_jiegouguang_save/left2.png','d455_jiegouguang_save/right2.png') # 复杂场景 zyb桌子

jiegouguang_class.import_biaodin('biaoding/extrinsics_d455_20250915.yml','biaoding/intrinsics_d455_20250915.yml')


# pcd = jiegouguang_class.sgbm()
# pcd = jiegouguang_class.manual_feature_extracting()
# pcd = jiegouguang_class.foundation_stereo()
# pcd = jiegouguang_class.lg_feature_extracting()
pcd = jiegouguang_class.bridgedepth_stereo_matching()

# jiegouguang_class.cal_error(pcd)


# pcd =  pcd.voxel_down_sample(voxel_size=10)
pcd.colors = o3d.utility.Vector3dVector(np.repeat([[1.0, 0.0, 0.0]], len(pcd.points), axis=0))
o3d.io.write_point_cloud("test.ply", pcd)

print(f"Pointcloud shape: {len(pcd.points)}")





# cv2.imwrite("test2.png", img2_with_center)
print("end")

