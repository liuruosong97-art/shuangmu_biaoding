import cv2
import os
import numpy as np

from jiegouguang_class import JieGouGuang

img = cv2.imread('d455_jiegouguang_save/better/left2.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
H,W = img.shape

jiegouguang_class = JieGouGuang('d455_jiegouguang_save/better/left3.png','d455_jiegouguang_save/better/right3.png')
# jiegouguang_class = JieGouGuang('whx_biaoding/L/left_0041.png','whx_biaoding/R/right_0041.png')

jiegouguang_class.import_biaodin('biaoding/extrinsics_d435_20250915.yml','biaoding/intrinsics_d435_20250915.yml')
# jiegouguang_class.import_biaodin('biaoding/extrinsics_whx_zyb.yml','biaoding/intrinsics_whx_zyb.yml')


# biaoding_img = jiegouguang_class.draw_chess_board(jiegouguang_class.img1,jiegouguang_class.img2)
# biaoding_img_rectify = jiegouguang_class.draw_chess_board(jiegouguang_class.img1_rectify,jiegouguang_class.img2_rectify)



img1_with_center,img2_with_center = jiegouguang_class.extract_circle()

jiegouguang_class.feature_extracting()

jiegouguang_class.feature_matching_sift()

jiegouguang_class.pointcloud_from_disparity()

# jiegouguang_class.triangulate_points()


# img_out = np.hstack((img1_with_center, img2_with_center))


cv2.imwrite("test.png", jiegouguang_class.match_img)
# cv2.imwrite("test2.png", img2_with_center)
print("end")

