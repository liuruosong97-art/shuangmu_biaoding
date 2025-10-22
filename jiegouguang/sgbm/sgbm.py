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

    
    def sgbm(self, min_disparity=0, num_disparities=64, block_size=3,
             invalidate_nonpositive=True, depth_clip=True,
             build_pointcloud=True, return_all=False):
        """运行 SGBM 获取视差与深度，并在内部由深度图生成点云。"""

        img1_rectified = self.img1_rectify
        img2_rectified = self.img2_rectify

        img_channels = 1
        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * img_channels * block_size * block_size,
            P2=32 * img_channels * block_size * block_size,
            disp12MaxDiff=-1,
            preFilterCap=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=100,
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
