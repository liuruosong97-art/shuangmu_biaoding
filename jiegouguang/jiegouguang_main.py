import cv2
import os
import numpy as np
from jiegouguang import JieGouGuang 

img = cv2.imread('d455_jiegouguang_save/left2.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
H,W = img.shape

jiegouguang_class = JieGouGuang('d455_jiegouguang_save/left2.png','d455_jiegouguang_save/right2.png')
jiegouguang_class.import_biaodin('biaoding/extrinsics_d435_20250915.yml','biaoding/intrinsics_d435_20250915.yml')
centers_img1,centers_img2,img1_with_center,img2_with_center = jiegouguang_class.extract_circle()



img_out = np.hstack((img1_with_center, img2_with_center))


cv2.imwrite("test.png", img_out)
print("end")

