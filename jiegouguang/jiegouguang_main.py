import cv2
import os
import numpy as np

from jiegouguang_class import JieGouGuang
import open3d as o3d
import time

from tqdm import tqdm

jiegouguang_class = JieGouGuang('d455_jiegouguang_save/20251116_videos/floor/left_20251116_130809.mp4','d455_jiegouguang_save/20251116_videos/floor/right_20251116_130809.mp4') # 地板视频
# jiegouguang_class = JieGouGuang('d455_jiegouguang_save/20251116_videos/zyb_desk/left_20251116_122816.mp4','d455_jiegouguang_save/20251116_videos/zyb_desk/right_20251116_122816.mp4') # zyb桌子视频
# jiegouguang_class = JieGouGuang('d455_jiegouguang_save/20251103/left3.png','d455_jiegouguang_save/20251103/right3.png') # 一个杯子
# jiegouguang_class = JieGouGuang('d455_jiegouguang_save/better/left2.png','d455_jiegouguang_save/better/right2.png') # 平面场景
# jiegouguang_class = JieGouGuang('d455_jiegouguang_save/left2.png','d455_jiegouguang_save/right2.png') # 复杂场景 zyb桌子

# 下面这两种是稀疏方法 暂时弃用
# pcd = jiegouguang_class.manual_feature_extracting()
# pcd = jiegouguang_class.lg_feature_extracting()


# jiegouguang_class.method = 'sgbm'
# jiegouguang_class.method = 'foundation_stereo'
jiegouguang_class.method = 'bridgedepth'
jiegouguang_class.init_model()

depth_list = []
depth_dir = 'depth_outputs'
os.makedirs(depth_dir, exist_ok=True)

video_writer = None

for idx in tqdm(range(len(jiegouguang_class.img1_list))):
    jiegouguang_class.import_biaodin('biaoding/extrinsics_d455_20250915.yml','biaoding/intrinsics_d455_20250915.yml',idx=idx)



    # start = time.time()
    disparity_raw = jiegouguang_class.forward_disparity()
    # print(f"\033[31mCosting time (s): {time.time() - start}\033[0m")




    depth = (float(jiegouguang_class.K1[0, 0]) * abs(float(jiegouguang_class.cam_t[0]))) / disparity_raw
    depth = np.clip(depth, jiegouguang_class.min_dis, jiegouguang_class.max_dis)
    depth_list.append(depth)
    
    depth_vis = np.clip(depth * 0.2, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(depth_dir, f'depth_{idx:03d}.png'), depth_vis)
    video_path = os.path.join(depth_dir, 'adepth_sequence.mp4')
    
    if video_writer is None:
        height, width = depth_vis.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, 30, (width, height), False)
        if not video_writer.isOpened():
            raise RuntimeError(f'failed to open video writer for {video_path}')
    video_writer.write(depth_vis)
    pcd = jiegouguang_class.depth2pointcloud(depth)
    # pcd =  pcd.voxel_down_sample(voxel_size=10)
    pcd.colors = o3d.utility.Vector3dVector(np.repeat([[1.0, 0.0, 0.0]], len(pcd.points), axis=0))
    o3d.io.write_point_cloud(os.path.join(depth_dir, f'cloud_{idx:03d}.ply'), pcd)

    

    # pcd = jiegouguang_class.depth2pointcloud(depth)
    # # pcd =  pcd.voxel_down_sample(voxel_size=10)
    # pcd.colors = o3d.utility.Vector3dVector(np.repeat([[1.0, 0.0, 0.0]], len(pcd.points), axis=0))
    # o3d.io.write_point_cloud("test.ply", pcd)


    # jiegouguang_class.cal_error(pcd) # 计算平面误差

if video_writer is not None:
    video_writer.release()
    





