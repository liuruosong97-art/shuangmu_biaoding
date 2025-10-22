import cv2
import os
import numpy as np

from jiegouguang_class import JieGouGuang
import open3d as o3d

img = cv2.imread('d455_jiegouguang_save/better/left2.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
H,W = img.shape

jiegouguang_class = JieGouGuang('d455_jiegouguang_save/better/left3.png','d455_jiegouguang_save/better/right3.png')
# jiegouguang_class = ManualFeatureJieGouGuang('d455_jiegouguang_save/better/left3.png','d455_jiegouguang_save/better/right3.png')
# jiegouguang_class = ManualFeatureJieGouGuang('d455_jiegouguang_save/left2.png','d455_jiegouguang_save/right2.png')

jiegouguang_class.import_biaodin('biaoding/extrinsics_d455_20250915.yml','biaoding/intrinsics_d455_20250915.yml')


pcd = jiegouguang_class.sgbm()
# pcd = jiegouguang_class.manual_feature_extracting()







pcd =  pcd.voxel_down_sample(voxel_size=10)
o3d.io.write_point_cloud("test.ply", pcd)
print(f"Pointcloud shape: {len(pcd.points)}")


# cv2.imwrite("test2.png", img2_with_center)
print("end")

