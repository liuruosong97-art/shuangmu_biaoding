"""
==================================================================================================
神经网络特征提取和匹配模块 (NeuralFeature)
Neural Network Feature Extraction and Matching Module
==================================================================================================

【模块功能 / Module Function】
使用 SuperPoint + LightGlue 进行结构光圆心的特征提取和匹配
Uses SuperPoint + LightGlue for structured light circle center feature extraction and matching

【核心创新 / Core Innovation】
1. 自定义关键点 + SuperPoint 描述子: 在圆心位置双线性插值采样神经网络描述子
   Custom keypoints + SuperPoint descriptors: Bilinear interpolation sampling at circle centers

2. 100% 圆心保留: 不依赖 SuperPoint 自动检测，保证无信息损失
   100% circle retention: Independent of SuperPoint auto-detection, no information loss

3. LightGlue 鲁棒匹配: Transformer 注意力机制 + 自适应几何约束
   LightGlue robust matching: Transformer attention + adaptive geometric constraints

【性能提升 / Performance Improvement】
- 匹配率: 4.9% → 29.5% (6倍提升)
  Match rate: 4.9% → 29.5% (6x improvement)
- 内点数: 114 → 688 (504% 提升)
  Inliers: 114 → 688 (504% improvement)

【类似于 / Similar to】
ManualFeature.py - 继承 JieGouGuang base 的所有属性，独立处理器模式
Inherits all attributes from JieGouGuang base, standalone processor pattern


【使用方法 / Usage】

方式1: 通过 JieGouGuang 基类（推荐，自动获取标定参数）
Method 1: Via JieGouGuang base (Recommended, auto-get calibration parameters)

    from jiegouguang_class import JieGouGuang

    # 初始化并导入标定参数
    jgg = JieGouGuang('left.png', 'right.png')
    jgg.import_biaodin('extrinsics.yml', 'intrinsics.yml')

    # 使用神经网络特征提取和匹配
    pcd = jgg.neural_feature_extracting()

    # 保存点云
    import open3d as o3d
    o3d.io.write_point_cloud("output.ply", pcd)


方式2: 直接传入图像和圆心（适用于已有圆心检测结果的场景）
Method 2: Direct image and keypoint input (For pre-detected circle centers)

    from jiegouguang.NeuralFeature import NeuralFeatureJieGouGuang
    import cv2
    import numpy as np

    # 加载图像
    img1 = cv2.imread('left.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('right.png', cv2.IMREAD_GRAYSCALE)

    # 已有的圆心坐标（通过其他方法检测）
    centers1 = np.array([[100, 200], [150, 250]], dtype=np.float32)
    centers2 = np.array([[105, 200], [155, 250]], dtype=np.float32)

    # 初始化处理器
    processor = NeuralFeatureJieGouGuang(
        img1=img1,
        img2=img2,
        centers1=centers1,
        centers2=centers2,
        device='cuda'  # 或 'cpu'
    )

    # 方式2.1: 自动提取 SuperPoint 描述子（默认）
    processor.feature_extracting()

    # 或 方式2.2: 传入外部 SuperPoint 描述子（256维）
    external_desc1 = ...  # 形状 (N1, 256)
    external_desc2 = ...  # 形状 (N2, 256)
    processor.feature_extracting(
        external_desc1=external_desc1,
        external_desc2=external_desc2
    )

    # 进行匹配
    processor.feature_matching()

    # 获取匹配结果
    print(f"匹配数: {len(processor.matched_kpts0)}")
    print(f"匹配率: {len(processor.matched_kpts0)/len(centers1)*100:.1f}%")
    print(f"平均置信度: {processor.match_confidences.mean():.3f}")

    # 如果有相机参数，可以生成点云
    if processor.P1 is not None and processor.P2 is not None:
        pcd = processor.pointcloud_from_disparity()


【重要说明 / Important Notes】

⚠️ 描述子维度限制: LightGlue 仅支持 256 维 SuperPoint 描述子
   Descriptor dimension: LightGlue only supports 256-dim SuperPoint descriptors
   ✅ 支持 Supported: SuperPoint (256-dim)
   ❌ 不支持 Not supported: SIFT (128-dim), ORB (32-byte)

⚠️ 亚像素检测（默认开启）: 使用高斯拟合提取亚像素精度圆心
   Subpixel detection (enabled by default): Gaussian fitting for subpixel accuracy
   关闭方法 Disable: use_subpixel=False

⚠️ 设备选择: 优先使用 CUDA，如果不可用自动回退到 CPU
   Device selection: CUDA preferred, auto-fallback to CPU if unavailable


【日期 / Date】2025-10-24
【作者 / Author】WHX
【参考 / Reference】
- shuangmu_lightglue_integrated.py
- features/provider_custom_kpts_superpoint.py
- features/matcher_lightglue.py

==================================================================================================
"""

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from typing import Dict, Optional

# =================================================================================================
# 导入 LightGlue 和 SuperPoint
# Import LightGlue and SuperPoint
# =================================================================================================
try:
    from lightglue import SuperPoint, LightGlue
except ImportError:
    raise ImportError(
        "LightGlue not installed. Install with:\n"
        "pip install lightglue\n"
        "Or: git clone https://github.com/cvg/LightGlue && cd LightGlue && pip install -e ."
    )


# =================================================================================================
# 主类: 神经网络特征提取和匹配处理器
# Main Class: Neural Network Feature Extraction and Matching Processor
# =================================================================================================
class NeuralFeatureJieGouGuang:
    """
    神经网络特征提取和匹配处理器
    Neural Network Feature Extraction and Matching Processor

    使用深度学习方法（SuperPoint + LightGlue）替代传统 SIFT/ORB
    Uses deep learning (SuperPoint + LightGlue) to replace traditional SIFT/ORB

    特别优化用于结构光圆心的匹配场景
    Specially optimized for structured light circle center matching scenarios
    """

    def __init__(self, base=None, device='cuda', use_subpixel=True,
                 img1=None, img2=None, centers1=None, centers2=None, K1=None, P1=None, P2=None,
                 *args, **kwargs):
        """
        可接受父类实例以继承其全部属性，或直接传入图像和关键点
        Accepts parent instance to inherit all attributes, or directly pass images and keypoints

        Args:
            base: JieGouGuang 基类实例，提供图像和标定参数
                 JieGouGuang base instance providing images and calibration parameters
            device: 计算设备 ('cuda' 或 'cpu')，默认自动检测
                   Computation device ('cuda' or 'cpu'), auto-detect by default
            use_subpixel: 是否使用亚像素精度圆心检测（默认True）
                         Whether to use subpixel circle detection (default True)

            --- 直接传入模式 Direct Input Mode ---
            img1: 左图灰度图像 (H, W) / Left grayscale image
            img2: 右图灰度图像 (H, W) / Right grayscale image
            centers1: 左图圆心坐标 (N1, 2) / Left image circle centers
            centers2: 右图圆心坐标 (N2, 2) / Right image circle centers
            K1: 左相机内参矩阵 (3, 3) / Left camera intrinsic matrix (用于 RANSAC)
            P1: 左相机投影矩阵 (3, 4) / Left camera projection matrix (用于三角化)
            P2: 右相机投影矩阵 (3, 4) / Right camera projection matrix (用于三角化)
        """
        # ===== 初始化默认属性 / Initialize default attributes =====
        # 必须先设置默认值，然后再覆盖
        self.img1_rectify = None
        self.img2_rectify = None
        self.centers_img1 = None
        self.centers_img2 = None
        self.K1 = None
        self.P1 = None
        self.P2 = None
        self.descriptors_img1 = None
        self.descriptors_img2 = None
        self.matched_kpts0 = None
        self.matched_kpts1 = None
        self.match_confidences = None

        # ===== 继承 base 的所有属性 / Inherit all attributes from base =====
        if base is not None:
            self.__dict__ = base.__dict__.copy()
        # ===== 或使用直接传入的数据 / Or use directly provided data =====
        elif img1 is not None and img2 is not None and centers1 is not None and centers2 is not None:
            self.img1_rectify = img1
            self.img2_rectify = img2
            self.centers_img1 = centers1
            self.centers_img2 = centers2
            self.K1 = K1  # 可选，用于 RANSAC
            self.P1 = P1  # 可选，用于三角化
            self.P2 = P2  # 可选，用于三角化
            print("[NeuralFeature] Initialized with direct image and keypoint input")
        else:
            super().__init__(*args, **kwargs)

        # ===== 亚像素检测选项 / Subpixel detection option =====
        self.use_subpixel = use_subpixel

        # ===== 神经网络设备选择 / Neural network device selection =====
        if device == 'cuda' and not torch.cuda.is_available():
            print("[NeuralFeature] CUDA not available, using CPU")
            device = 'cpu'
        self.device = torch.device(device)

        # ===== 初始化 SuperPoint 提取器 / Initialize SuperPoint extractor =====
        print(f"[NeuralFeature] Loading SuperPoint model on {self.device}...")
        self.extractor = SuperPoint(max_num_keypoints=None).eval().to(self.device)

        # ===== 初始化 LightGlue 匹配器 / Initialize LightGlue matcher =====
        print(f"[NeuralFeature] Loading LightGlue matcher on {self.device}...")
        self.matcher = LightGlue(features='superpoint').eval().to(self.device)

        print(f"[NeuralFeature] Initialized successfully on device: {self.device}")

    def extract_circle(self):
        """
        提取左右图像中的圆心
        Extract circle centers from left and right images

        从 base 继承的 img1_rectify, img2_rectify 中提取圆心
        Extract centers from inherited img1_rectify, img2_rectify

        使用 base 的 extract_circle_1 方法
        Uses base's extract_circle_1 method

        Returns:
            img1_with_center: 标记了圆心的左图像 / Left image with marked centers
            img2_with_center: 标记了圆心的右图像 / Right image with marked centers
        """
        mode_str = "亚像素精度" if self.use_subpixel else "像素级精度"
        print(f"[NeuralFeature] Extracting circle centers ({mode_str})...")

        # 调用继承自 base 的方法 / Call method inherited from base
        centers_img1, img1_with_center = self.extract_circle_1(self.img1_rectify, use_subpixel=self.use_subpixel)
        centers_img2, img2_with_center = self.extract_circle_1(self.img2_rectify, use_subpixel=self.use_subpixel)

        self.centers_img1 = centers_img1
        self.centers_img2 = centers_img2

        print(f"[NeuralFeature] Left image: {len(centers_img1)} circles")
        print(f"[NeuralFeature] Right image: {len(centers_img2)} circles")

        return img1_with_center, img2_with_center

    def feature_extracting(self, external_desc1=None, external_desc2=None):
        """
        使用 SuperPoint 提取圆心的神经网络描述子
        Extract neural network descriptors for circle centers using SuperPoint

        Args:
            external_desc1: 可选的外部左图描述子 (N1, 256) / Optional external left descriptors
            external_desc2: 可选的外部右图描述子 (N2, 256) / Optional external right descriptors
                          如果提供，将跳过 SuperPoint 提取，直接使用外部描述子
                          If provided, will skip SuperPoint extraction and use external descriptors
                          ⚠️ 注意: LightGlue 只支持 256 维 SuperPoint 描述子
                          ⚠️ Note: LightGlue only supports 256-dim SuperPoint descriptors

        核心技术 / Core Technology:
        1. 运行 SuperPoint 编码器获取密集描述子图 (H/8 x W/8 x 256)
           Run SuperPoint encoder to get dense descriptor map
        2. 在自定义圆心位置使用双线性插值采样描述子
           Sample descriptors at custom circle positions using bilinear interpolation
        3. 保证 100% 圆心保留（无信息损失）
           Guarantee 100% circle retention (no information loss)
        4. L2 归一化 256 维描述子
           L2-normalize 256-dim descriptors

        存储结果 / Stored Results:
        - self.descriptors_img1: 左图描述子 (N1, 256 or D)
        - self.descriptors_img2: 右图描述子 (N2, 256 or D)
        """
        if not hasattr(self, 'centers_img1') or self.centers_img1 is None or \
           not hasattr(self, 'centers_img2') or self.centers_img2 is None:
            raise ValueError("Must call extract_circle() first or provide centers in __init__!")

        # ===== 使用外部描述子（如果提供）/ Use external descriptors if provided =====
        if external_desc1 is not None and external_desc2 is not None:
            print("[NeuralFeature] Using external descriptors...")

            # 验证维度
            if len(external_desc1) != len(self.centers_img1):
                raise ValueError(f"External descriptor count mismatch for left image: "
                               f"{len(external_desc1)} vs {len(self.centers_img1)} centers")
            if len(external_desc2) != len(self.centers_img2):
                raise ValueError(f"External descriptor count mismatch for right image: "
                               f"{len(external_desc2)} vs {len(self.centers_img2)} centers")

            self.descriptors_img1 = external_desc1.astype(np.float32)
            self.descriptors_img2 = external_desc2.astype(np.float32)

            print(f"[NeuralFeature] External descriptors loaded: "
                  f"Left {self.descriptors_img1.shape}, Right {self.descriptors_img2.shape}")
            return

        # ===== 使用 SuperPoint 提取描述子（默认）/ Extract with SuperPoint (default) =====
        print("[NeuralFeature] Extracting neural network descriptors...")

        # ===== 提取左图描述子 / Extract left image descriptors =====
        self.descriptors_img1 = self._extract_descriptors_for_keypoints(
            self.img1_rectify,
            self.centers_img1
        )

        # ===== 提取右图描述子 / Extract right image descriptors =====
        self.descriptors_img2 = self._extract_descriptors_for_keypoints(
            self.img2_rectify,
            self.centers_img2
        )

        print(f"[NeuralFeature] Extracted descriptors: "
              f"Left {self.descriptors_img1.shape}, Right {self.descriptors_img2.shape}")

        # ===== 验证描述子数量与圆心数量一致 / Verify descriptor count matches keypoint count =====
        assert len(self.descriptors_img1) == len(self.centers_img1), \
            f"Descriptor count mismatch for left image: {len(self.descriptors_img1)} vs {len(self.centers_img1)}"
        assert len(self.descriptors_img2) == len(self.centers_img2), \
            f"Descriptor count mismatch for right image: {len(self.descriptors_img2)} vs {len(self.centers_img2)}"

    def _extract_descriptors_for_keypoints(self, image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """
        为给定关键点提取 SuperPoint 描述子（核心创新函数）
        Extract SuperPoint descriptors for given keypoints (core innovation)

        核心创新 / Core Innovation:
        传统方法: KD-Tree 最近邻查找 SuperPoint 关键点 → 丢失 40% 圆心
        本方法: 在 SuperPoint 密集描述子图上双线性插值采样 → 100% 圆心保留

        Args:
            image: 灰度图像 (H, W) / Grayscale image
            keypoints: 关键点坐标 (N, 2) [x, y] / Keypoint coordinates

        Returns:
            descriptors: (N, 256) L2归一化描述子 / L2-normalized descriptors
        """
        # ===== 步骤1: 图像预处理 / Step 1: Image preprocessing =====
        image_tensor = self._preprocess_image(image)

        with torch.no_grad():
            # ===== 步骤2: 获取密集描述子图 / Step 2: Get dense descriptor map =====
            dense_desc_map = self._get_dense_descriptors(image_tensor)
            # Shape: (1, 256, H/8, W/8)

            # ===== 步骤3: 在关键点位置采样描述子 / Step 3: Sample descriptors at keypoints =====
            descriptors = self._sample_descriptors(dense_desc_map, keypoints)
            # Shape: (N, 256)

        return descriptors.cpu().numpy()

    def _get_dense_descriptors(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        手动运行 SuperPoint 编码器，获取密集描述子图
        Manually run SuperPoint encoder to get dense descriptor map

        SuperPoint 网络架构 / SuperPoint Network Architecture:
        输入 Input: (1, 1, H, W)
          ↓ conv1a+ReLU → conv1b+ReLU → pool(2x): (1, 64, H/2, W/2) ← 第1次下采样 / 1st downsample
          ↓ conv2a+ReLU → conv2b+ReLU → pool(2x): (1, 64, H/4, W/4) ← 第2次下采样 / 2nd downsample
          ↓ conv3a+ReLU → conv3b+ReLU → pool(2x): (1, 128, H/8, W/8) ← 第3次下采样 / 3rd downsample
          ↓ conv4a+ReLU → conv4b+ReLU: (1, 256, H/8, W/8)
          ↓
        编码器输出 Encoder output: (1, 256, H/8, W/8)
          ↓
          ├─ 检测头 Detection head (我们不用 / Not used)
          └─ 描述子头 Descriptor head:
              ↓ convDa+ReLU → convDb: (1, 256, H/8, W/8)
              ↓ L2归一化 / L2 normalize
        输出 Output: (1, 256, H/8, W/8)

        Args:
            image_tensor: (1, 1, H, W) 预处理后的图像 / Preprocessed image

        Returns:
            (1, 256, H/8, W/8) 密集描述子图（已L2归一化）
            Dense descriptor map (L2-normalized)
        """
        x = image_tensor

        # ===== 共享编码器 Shared Encoder (conv1-4) =====
        # Stage 1
        x = self.extractor.relu(self.extractor.conv1a(x))
        x = self.extractor.relu(self.extractor.conv1b(x))
        x = self.extractor.pool(x)  # → H/2, W/2

        # Stage 2
        x = self.extractor.relu(self.extractor.conv2a(x))
        x = self.extractor.relu(self.extractor.conv2b(x))
        x = self.extractor.pool(x)  # → H/4, W/4

        # Stage 3
        x = self.extractor.relu(self.extractor.conv3a(x))
        x = self.extractor.relu(self.extractor.conv3b(x))
        x = self.extractor.pool(x)  # → H/8, W/8

        # Stage 4
        x = self.extractor.relu(self.extractor.conv4a(x))
        x = self.extractor.relu(self.extractor.conv4b(x))

        # ===== 描述子头 Descriptor Head =====
        desc = self.extractor.relu(self.extractor.convDa(x))
        desc = self.extractor.convDb(desc)

        # ===== L2归一化 L2 Normalization =====
        desc = F.normalize(desc, p=2, dim=1)

        return desc  # (1, 256, H/8, W/8)

    def _sample_descriptors(self, desc_map: torch.Tensor, keypoints: np.ndarray) -> torch.Tensor:
        """
        在任意位置采样描述子（双线性插值）
        Sample descriptors at arbitrary positions (bilinear interpolation)

        为什么需要双线性插值 / Why bilinear interpolation:
        - 圆心坐标: 精确像素坐标 (e.g., x=123.7, y=45.2)
          Circle coordinates: Precise pixel coordinates
        - 描述子图: 8倍下采样 (123.7/8 = 15.4625)
          Descriptor map: 8x downsampled
        - 双线性插值: 在非整数位置采样，周围4个点加权平均
          Bilinear interpolation: Sample at non-integer positions, weighted average of 4 neighbors

        Args:
            desc_map: (1, 256, H/8, W/8) SuperPoint 密集描述子图
            keypoints: (N, 2) 关键点坐标，在原始图像空间 [x, y]

        Returns:
            (N, 256) 采样得到的描述子（L2归一化）
        """
        _, D, H_desc, W_desc = desc_map.shape
        N = len(keypoints)

        # ===== 步骤1: 坐标转换 - 原始图像空间 → 描述子图空间 =====
        kpts_tensor = torch.from_numpy(keypoints).float().to(self.device)
        kpts_desc_space = kpts_tensor / 8.0  # SuperPoint: 8x downsample

        # ===== 步骤2: 归一化坐标到 [-1, 1] (grid_sample 要求) =====
        kpts_norm = kpts_desc_space.clone()
        kpts_norm[:, 0] = 2.0 * kpts_desc_space[:, 0] / (W_desc - 1) - 1.0  # x
        kpts_norm[:, 1] = 2.0 * kpts_desc_space[:, 1] / (H_desc - 1) - 1.0  # y

        # ===== 步骤3: 重塑为 grid_sample 格式 =====
        grid = kpts_norm.view(1, N, 1, 2)

        # ===== 步骤4: 双线性插值采样 =====
        sampled = F.grid_sample(
            desc_map,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        # ===== 步骤5: 重塑输出 (1, 256, N, 1) → (N, 256) =====
        sampled = sampled.squeeze(0).squeeze(2).permute(1, 0)

        # ===== 步骤6: L2归一化 =====
        sampled = F.normalize(sampled, p=2, dim=1)

        return sampled  # (N, 256)

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        图像预处理: numpy → PyTorch tensor
        Image preprocessing: numpy → PyTorch tensor

        Args:
            image: numpy数组 / numpy array
                - 灰度图 Grayscale: (H, W) uint8, [0, 255]
                - 彩色图 Color: (H, W, 3) uint8, [0, 255], BGR

        Returns:
            (1, 1, H, W) torch tensor, float32, [0, 1]
        """
        # 转换为灰度图 / Convert to grayscale
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 归一化到 [0, 1] / Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # 转换为 PyTorch tensor 并添加 batch 和 channel 维度
        # Convert to PyTorch tensor and add batch and channel dimensions
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

        # 移动到设备 / Move to device
        image_tensor = image_tensor.to(self.device)

        return image_tensor  # (1, 1, H, W)

    def feature_matching(self):
        """
        使用 LightGlue 进行鲁棒匹配
        Robust matching using LightGlue

        匹配流程 / Matching Pipeline:
        1. LightGlue Transformer 注意力机制匹配
           LightGlue Transformer attention mechanism matching
        2. One-to-one 匹配强制（避免 one-to-many 错误）
           One-to-one matching enforcement (avoid one-to-many errors)
        3. 自适应几何约束过滤:
           Adaptive geometric constraint filtering:
           - 极线约束 Epipolar constraint: |y0 - y1| < tolerance
           - 视差约束 Disparity constraint: d ∈ [d_min, d_max]
        4. RANSAC 几何验证（过滤几何不一致匹配）
           RANSAC geometric verification (filter geometrically inconsistent matches)

        存储结果 / Stored Results:
        - self.matched_kpts0: 匹配的左图关键点 (M, 2)
        - self.matched_kpts1: 匹配的右图关键点 (M, 2)
        - self.match_confidences: 匹配置信度 (M,)
        """
        print("[NeuralFeature] Matching with LightGlue...")

        if self.descriptors_img1 is None or self.descriptors_img2 is None:
            raise ValueError("Must call feature_extracting() first!")

        # ===== 准备 LightGlue 输入 / Prepare LightGlue input =====
        feats0 = {
            'keypoints': self.centers_img1,
            'descriptors': self.descriptors_img1,
            'image_size': np.array([self.img1_rectify.shape[1], self.img1_rectify.shape[0]])
        }
        feats1 = {
            'keypoints': self.centers_img2,
            'descriptors': self.descriptors_img2,
            'image_size': np.array([self.img2_rectify.shape[1], self.img2_rectify.shape[0]])
        }

        # ===== 使用 LightGlue 匹配 / Match with LightGlue =====
        matches_result = self._lightglue_match(feats0, feats1)

        print(f"[NeuralFeature] Initial matches from LightGlue: {matches_result['num_matches']}")

        # ===== 提取匹配的关键点对 / Extract matched keypoint pairs =====
        if matches_result['num_matches'] > 0:
            self.matched_kpts0 = matches_result['matched_kpts0']
            self.matched_kpts1 = matches_result['matched_kpts1']
            self.match_confidences = matches_result['scores']

            # ===== 应用几何约束过滤 / Apply geometric constraint filtering =====
            self._apply_geometric_filter()

            print(f"[NeuralFeature] Final matches after filtering: {len(self.matched_kpts0)}")
        else:
            print("[NeuralFeature] Warning: No matches found!")
            self.matched_kpts0 = np.array([])
            self.matched_kpts1 = np.array([])
            self.match_confidences = np.array([])

    def _lightglue_match(self, feats0: Dict[str, np.ndarray], feats1: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        LightGlue 匹配（核心匹配函数）
        LightGlue matching (core matching function)

        关键实现要点 / Key Implementation Notes:
        1. 【批次维度处理】CRITICAL: 必须使用 [0] 移除 LightGlue 输出的批次维度
           Batch dimension handling: MUST use [0] to remove batch dimension
           - 错误 Wrong: matches = output['matches0'].cpu().numpy()  → (1, N0) shape
           - 正确 Correct: matches = output['matches0'][0].cpu().numpy()  → (N0,) shape
           - 缺少 [0] 会导致 "one-to-many" 匹配错误 (可视化为放射状线条)
             Missing [0] causes "one-to-many" errors (radiating lines in visualization)

        2. Bug 修复历史 / Bug Fix History:
           - 2025-10-20: 修复批次维度 bug，匹配率提升 6 倍 (4.9% → 29.5%)
             Fixed batch dimension bug, 6x match rate improvement
        """
        # ===== 步骤1: 数据准备 - 转换为 torch 张量并添加批次维度 =====
        data0 = self._prepare_data(feats0)
        data1 = self._prepare_data(feats1)

        # ===== 步骤2: 准备 LightGlue 输入字典 =====
        input_dict = {
            'image0': data0,
            'image1': data1
        }

        # ===== 步骤3: 运行匹配器（推理模式，无梯度） =====
        with torch.no_grad():
            output = self.matcher(input_dict)

        # ===== 步骤4: 提取结果并转回 numpy =====
        # ⚠️ CRITICAL: 必须使用 [0] 移除批次维度
        matches = output['matches0'][0].cpu().numpy()  # (N0,) with -1 for unmatched
        scores = output['matching_scores0'][0].cpu().numpy()  # (N0,)

        # ===== 步骤5: 转换为 (K, 2) 格式并过滤有效匹配 =====
        valid_mask = matches >= 0
        match_indices = np.stack([
            np.where(valid_mask)[0],  # 第一幅图像的索引
            matches[valid_mask]       # 第二幅图像的索引
        ], axis=1).astype(np.int64)

        match_scores = scores[valid_mask].astype(np.float32)

        # ===== 步骤6: 提取匹配关键点的坐标 =====
        kpts0 = feats0['keypoints']
        kpts1 = feats1['keypoints']
        matched_kpts0 = kpts0[match_indices[:, 0]]
        matched_kpts1 = kpts1[match_indices[:, 1]]

        return {
            'matches': match_indices,
            'scores': match_scores,
            'matched_kpts0': matched_kpts0,
            'matched_kpts1': matched_kpts1,
            'num_matches': len(match_indices)
        }

    def _prepare_data(self, feats: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        将 numpy 特征转换为 torch 张量，格式符合 LightGlue 要求
        Convert numpy features to torch tensors with correct format for LightGlue
        """
        keypoints = torch.from_numpy(feats['keypoints']).float().unsqueeze(0).to(self.device)
        descriptors = torch.from_numpy(feats['descriptors']).float().unsqueeze(0).to(self.device)
        image_size = torch.tensor(feats['image_size'], dtype=torch.long).unsqueeze(0).to(self.device)

        return {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'image_size': image_size
        }

    def _apply_geometric_filter(self):
        """
        应用自适应几何约束过滤
        Apply adaptive geometric constraint filtering

        过滤策略 / Filtering Strategy:
        1. 计算极线偏差和视差范围
           Calculate epipolar deviation and disparity range
        2. 根据矫正质量自适应调整容差:
           Adapt tolerance based on rectification quality:
           - y_median < 3px: 优秀矫正，严格约束 / Excellent rectification, strict constraints
           - 3px < y_median < 10px: 中等矫正，放宽约束 / Medium rectification, relaxed constraints
           - y_median > 10px: 差矫正，主要依赖 RANSAC / Poor rectification, rely on RANSAC
        3. RANSAC 验证: 过滤几何不一致的匹配
           RANSAC verification: Filter geometrically inconsistent matches
        """
        if len(self.matched_kpts0) < 8:
            print("[NeuralFeature] Too few matches for geometric filter, skipping")
            return

        # ===== 从 base 继承的标定参数 / Calibration parameters inherited from base =====
        f = float(self.K1[0, 0])
        B = abs(float(self.cam_t[0]))

        # ===== 基于深度范围计算视差范围 / Calculate disparity range based on depth range =====
        d_min = f * B / self.max_dis  # 最远点 → 最小视差 / Farthest point → min disparity
        d_max = f * B / self.min_dis  # 最近点 → 最大视差 / Nearest point → max disparity

        # ===== 极线约束 / Epipolar constraint =====
        y_diffs = np.abs(self.matched_kpts0[:, 1] - self.matched_kpts1[:, 1])
        y_median = np.median(y_diffs)

        # ===== 视差约束 / Disparity constraint =====
        disparities = self.matched_kpts0[:, 0] - self.matched_kpts1[:, 0]

        # ===== 自适应策略 / Adaptive strategy =====
        if y_median < 3.0:
            print(f"[NeuralFeature] Excellent rectification (y_median={y_median:.2f}px), using strict constraints")
            epi_tol = 2.0
            y_valid = y_diffs < epi_tol
            d_valid = (disparities > d_min) & (disparities < d_max)
            geom_mask = y_valid & d_valid
        elif y_median < 10.0:
            print(f"[NeuralFeature] Medium rectification (y_median={y_median:.2f}px), relaxing constraints")
            epi_tol = y_median * 2.0
            y_valid = y_diffs < epi_tol
            d_valid = (disparities > d_min * 0.5) & (disparities < d_max * 1.5)
            geom_mask = y_valid & d_valid
        else:
            print(f"[NeuralFeature] Poor rectification (y_median={y_median:.2f}px), relying on RANSAC")
            geom_mask = (disparities > 0) & (y_diffs < 100.0)

        # ===== 应用几何过滤 / Apply geometric filter =====
        self.matched_kpts0 = self.matched_kpts0[geom_mask]
        self.matched_kpts1 = self.matched_kpts1[geom_mask]
        self.match_confidences = self.match_confidences[geom_mask]

        print(f"[NeuralFeature] After geometric filter: {len(self.matched_kpts0)} matches")

        # ===== RANSAC 验证 / RANSAC verification =====
        if len(self.matched_kpts0) >= 8:
            E, inlier_mask = cv2.findEssentialMat(
                self.matched_kpts0,
                self.matched_kpts1,
                self.K1,
                method=cv2.RANSAC,
                prob=0.9999,
                threshold=1.0
            )

            if inlier_mask is not None:
                inlier_mask = inlier_mask.ravel().astype(bool)
                self.matched_kpts0 = self.matched_kpts0[inlier_mask]
                self.matched_kpts1 = self.matched_kpts1[inlier_mask]
                self.match_confidences = self.match_confidences[inlier_mask]

                print(f"[NeuralFeature] After RANSAC: {len(self.matched_kpts0)} inliers")

    def pointcloud_from_disparity(self):
        """
        从匹配点对三角化生成 3D 点云
        Triangulate 3D point cloud from matched point pairs

        使用继承自 base 的投影矩阵 (P1, P2) 进行三角化
        Uses projection matrices (P1, P2) inherited from base for triangulation

        Returns:
            pcd: Open3D 点云对象 / Open3D point cloud object
        """
        print("[NeuralFeature] Triangulating point cloud...")

        if self.matched_kpts0 is None or len(self.matched_kpts0) == 0:
            raise ValueError("No matched keypoints available for triangulation!")

        # ===== 准备投影矩阵（从 base 继承）/ Prepare projection matrices (inherited from base) =====
        P1 = self.P1
        P2 = self.P2

        # ===== 三角化 / Triangulation =====
        points_4d_hom = cv2.triangulatePoints(
            P1,
            P2,
            self.matched_kpts0.T,
            self.matched_kpts1.T
        )

        # ===== 归一化齐次坐标 / Normalize homogeneous coordinates =====
        points_3d = points_4d_hom[:3, :] / points_4d_hom[3, :]
        points_3d = points_3d.T

        # ===== 深度过滤（使用从 base 继承的深度范围）/ Depth filtering (using depth range from base) =====
        depths = points_3d[:, 2]
        valid_depth_mask = (depths > self.min_dis) & (depths < self.max_dis)
        points_3d_filtered = points_3d[valid_depth_mask]

        print(f"[NeuralFeature] Generated {len(points_3d_filtered)} 3D points "
              f"(filtered from {len(points_3d)})")

        # ===== 深度统计 / Depth statistics =====
        if len(points_3d_filtered) > 0:
            depth_stats = points_3d_filtered[:, 2]
            print(f"[NeuralFeature] Depth range: {depth_stats.min():.1f} - {depth_stats.max():.1f} mm")

        # ===== 创建 Open3D 点云 / Create Open3D point cloud =====
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d_filtered.astype(np.float64))

        return pcd

    def visualize_matches(self, save_path="neural_matches.png"):
        """
        可视化匹配结果
        Visualize matching results

        Args:
            save_path: 保存路径 / Save path
        """
        if self.matched_kpts0 is None or len(self.matched_kpts0) == 0:
            print("[NeuralFeature] No matches to visualize")
            return

        # ===== 转换图像为彩色 / Convert images to color =====
        img1_color = cv2.cvtColor(self.img1_rectify, cv2.COLOR_GRAY2BGR)
        img2_color = cv2.cvtColor(self.img2_rectify, cv2.COLOR_GRAY2BGR)

        # ===== 拼接左右图像 / Concatenate left and right images =====
        h1, w1 = img1_color.shape[:2]
        h2, w2 = img2_color.shape[:2]
        vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        vis[:h1, :w1] = img1_color
        vis[:h2, w1:w1+w2] = img2_color

        # ===== 绘制匹配线 / Draw matching lines =====
        for i in range(len(self.matched_kpts0)):
            pt1 = tuple(self.matched_kpts0[i].astype(int))
            pt2 = tuple((self.matched_kpts1[i] + np.array([w1, 0])).astype(int))

            # 根据置信度选择颜色（绿色=高置信度，黄色=低置信度）
            # Choose color based on confidence (green=high, yellow=low)
            confidence = self.match_confidences[i]
            color = (
                int(255 * (1 - confidence)),  # B
                int(255 * confidence),         # G
                0                               # R
            )

            cv2.line(vis, pt1, pt2, color, 1)
            cv2.circle(vis, pt1, 2, (0, 0, 255), -1)
            cv2.circle(vis, pt2, 2, (0, 255, 0), -1)

        cv2.imwrite(save_path, vis)
        print(f"[NeuralFeature] Matches visualization saved to {save_path}")


# =================================================================================================
# 文件结束 / End of File
# =================================================================================================
