import cv2
import numpy as np
import open3d as o3d



class SGBM:
    def __init__(self, base=None, *args, **kwargs):
        """可接受父类实例以继承其全部属性。"""
        if base is not None:
            self.__dict__ = base.__dict__.copy()
        else:
            super().__init__(*args, **kwargs)

    
    def sgbm(self, block_size=3,
             invalidate_nonpositive=True, depth_clip=True,
             build_pointcloud=True, return_all=False):
        """运行 SGBM 获取视差与深度，并在内部由深度图生成点云。"""

        img1_rectified = self.img1_rectify
        img2_rectified = self.img2_rectify

        img_channels = 1
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=int(self.max_disp//16*16),
            blockSize=block_size,
            # 惩罚项 P1，当视差变化为 1 时的罚分；常取 8 * 通道数 * block_size^2
            P1=8 * img_channels * block_size * block_size,
            # 惩罚项 P2，当视差变化大于 1 时的罚分；常取 32 * 通道数 * block_size^2
            P2=32 * img_channels * block_size * block_size,
            # 左右一致性检查允许的视差差异，-1 表示禁用
            disp12MaxDiff=-1,
            # 预滤波上限，用于归一化图像亮度
            preFilterCap=1,
            # 代价唯一性比率，越大越严格
            uniquenessRatio=10,
            # 去散斑的窗口尺寸
            speckleWindowSize=100,
            # 去散斑的视差差异阈值
            speckleRange=100,
            # 使用较稳定的 HH 模式
            mode=cv2.STEREO_SGBM_MODE_HH
        )

        disparity_raw = stereo.compute(img1_rectified, img2_rectified).astype(np.float32) / 16.0

        if invalidate_nonpositive:
            disparity_raw[disparity_raw <= 0] = np.nan

        f = float(self.K1[0, 0])
        baseline = float(np.linalg.norm(np.array(self.cam_t, dtype=np.float32).ravel()))
        if baseline == 0:
            baseline = abs(float(np.array(self.cam_t).ravel()[0]))

        depth_map = np.full_like(disparity_raw, np.nan, dtype=np.float32)
        valid_mask = ~np.isnan(disparity_raw)
        depth_map[valid_mask] = (f * baseline) / disparity_raw[valid_mask]

        if depth_clip:
            depth_min = getattr(self, 'min_dis', None)
            depth_max = getattr(self, 'max_dis', None)
            if depth_min is not None:
                depth_map[depth_map < depth_min] = np.nan
            if depth_max is not None:
                depth_map[depth_map > depth_max] = np.nan

        disparity_vis = cv2.normalize(disparity_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        pcd = None
        if build_pointcloud:
            mask = np.isfinite(depth_map)
            if np.any(mask):
                h, w = depth_map.shape
                ys, xs = np.indices((h, w))
                cx = float(self.K1[0, 2])
                cy = float(self.K1[1, 2])

                depth_valid = depth_map[mask]
                xs_valid = xs[mask].astype(np.float32)
                ys_valid = ys[mask].astype(np.float32)

                X = (xs_valid - cx) * depth_valid / f
                Y = (ys_valid - cy) * depth_valid / f
                Z = depth_valid
                pts = np.stack([X, Y, Z], axis=-1)

                colors_src = img1_rectified if img1_rectified.ndim == 2 else cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
                colors = colors_src[mask].astype(np.float32) / 255.0
                colors = np.stack([colors, colors, colors], axis=-1)

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd.colors = o3d.utility.Vector3dVector(colors)

        self.disparity_map = disparity_raw
        self.depth_map = depth_map
        self.pcd_sgbm = pcd
        self.disparity_vis = disparity_vis

        if return_all:
            return disparity_raw, depth_map, pcd, disparity_vis
        return pcd
