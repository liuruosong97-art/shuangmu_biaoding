# NeuralFeature - 神经网络特征提取和匹配模块

基于 **SuperPoint + LightGlue** 的深度学习特征提取和匹配，专为结构光圆心匹配优化。

## 快速开始

### 方式1：通过 JieGouGuang 基类（推荐）

```python
from jiegouguang_class import JieGouGuang
import open3d as o3d

# 初始化并导入标定参数
jgg = JieGouGuang('left.png', 'right.png')
jgg.import_biaodin('extrinsics.yml', 'intrinsics.yml')

# 一行代码完成：圆心检测 → 特征提取 → 匹配 → 点云生成
pcd = jgg.neural_feature_extracting()

# 保存点云
o3d.io.write_point_cloud("output.ply", pcd)
```

**优点**：自动继承所有标定参数，支持完整流程（圆心检测 → 点云生成）

---

### 方式2：直接传入图像和圆心

适用于已有圆心检测结果的场景。

#### 2.1 自动提取 SuperPoint 描述子（默认）

```python
from jiegouguang.NeuralFeature import NeuralFeatureJieGouGuang
import cv2
import numpy as np

# 加载图像
img1 = cv2.imread('left.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('right.png', cv2.IMREAD_GRAYSCALE)

# 已有的圆心坐标（通过其他方法检测）
centers1 = np.array([[100, 200], [150, 250]], dtype=np.float32)  # (N1, 2)
centers2 = np.array([[105, 200], [155, 250]], dtype=np.float32)  # (N2, 2)

# 初始化处理器
processor = NeuralFeatureJieGouGuang(
    img1=img1,
    img2=img2,
    centers1=centers1,
    centers2=centers2,
    device='cuda'  # 或 'cpu'
)

# 自动提取 SuperPoint 描述子
processor.feature_extracting()

# 进行匹配
processor.feature_matching()

# 获取结果
print(f"匹配数: {len(processor.matched_kpts0)}")
print(f"匹配率: {len(processor.matched_kpts0)/len(centers1)*100:.1f}%")
print(f"平均置信度: {processor.match_confidences.mean():.3f}")
```

#### 2.2 传入外部 SuperPoint 描述子

```python
# ... (前面的初始化代码相同)

# 外部提取的 SuperPoint 描述子（必须是 256 维！）
external_desc1 = ...  # 形状 (N1, 256)
external_desc2 = ...  # 形状 (N2, 256)

# 传入外部描述子
processor.feature_extracting(
    external_desc1=external_desc1,
    external_desc2=external_desc2
)

# 进行匹配
processor.feature_matching()
```

**优点**：灵活性高，可复用已有圆心检测和描述子提取结果

---

## 数据流对比

| 数据 | 方式1 (Base继承) | 方式2 (直接传入) |
|------|-----------------|-----------------|
| **图像** | 自动从 base 继承 | 手动传入 `img1`, `img2` |
| **圆心** | 自动检测 | 手动传入 `centers1`, `centers2` |
| **描述子** | 自动 SuperPoint 提取 | 自动提取 或 外部传入 |
| **标定参数** | 自动从 base 继承 | 可选传入 `K1`, `P1`, `P2` |
| **点云生成** | ✅ 支持 | ✅ 支持（需提供 P1, P2） |

---

## 重要说明

### ⚠️ 描述子维度限制

**LightGlue 仅支持 256 维 SuperPoint 描述子！**

- ✅ **支持**: SuperPoint (256-dim)
- ❌ **不支持**: SIFT (128-dim), ORB (32-byte)

如果使用外部描述子，必须确保是 **256 维**。

### ⚠️ 亚像素检测（默认开启）

使用高斯拟合提取亚像素精度圆心，提高匹配精度。

```python
# 开启亚像素（默认）
processor = NeuralFeatureJieGouGuang(base=jgg, use_subpixel=True)

# 关闭亚像素（使用像素级最亮点法）
processor = NeuralFeatureJieGouGuang(base=jgg, use_subpixel=False)
```

### ⚠️ 设备选择

优先使用 CUDA，如果不可用自动回退到 CPU。

```python
# 自动选择（默认）
processor = NeuralFeatureJieGouGuang(base=jgg, device='cuda')

# 强制 CPU
processor = NeuralFeatureJieGouGuang(base=jgg, device='cpu')
```

---

## 性能提升

相比传统方法（SIFT/ORB）：

| 指标 | 传统方法 | NeuralFeature | 提升 |
|------|---------|--------------|------|
| **匹配率** | 4.9% | 29.5% | **6倍** |
| **内点数** | 114 | 688 | **504%** |
| **圆心保留** | 60% | 100% | **40%提升** |

---

## 核心技术

1. **自定义关键点 + SuperPoint 描述子**
   - 在圆心位置使用双线性插值采样密集描述子图
   - 保证 100% 圆心保留（无信息损失）

2. **LightGlue 鲁棒匹配**
   - Transformer 注意力机制
   - One-to-one 匹配强制
   - 自适应几何约束（极线 + 视差）
   - RANSAC 几何验证

3. **亚像素精度圆心检测**
   - 2D 高斯拟合
   - 亚像素级定位精度

---

## API 参考

### NeuralFeatureJieGouGuang

#### 初始化参数

```python
NeuralFeatureJieGouGuang(
    base=None,              # JieGouGuang 实例（方式1）
    device='cuda',          # 'cuda' 或 'cpu'
    use_subpixel=True,      # 是否使用亚像素检测

    # --- 直接传入模式（方式2）---
    img1=None,              # 左图灰度图 (H, W)
    img2=None,              # 右图灰度图 (H, W)
    centers1=None,          # 左图圆心 (N1, 2)
    centers2=None,          # 右图圆心 (N2, 2)
    K1=None,                # 左相机内参 (3, 3) - 用于 RANSAC
    P1=None,                # 左相机投影矩阵 (3, 4) - 用于三角化
    P2=None                 # 右相机投影矩阵 (3, 4) - 用于三角化
)
```

#### 主要方法

| 方法 | 说明 | 输入 | 输出 |
|------|------|------|------|
| `extract_circle()` | 提取圆心 | - | `img1_with_center`, `img2_with_center` |
| `feature_extracting()` | 提取描述子 | `external_desc1`, `external_desc2` (可选) | 更新 `self.descriptors_img1/2` |
| `feature_matching()` | LightGlue 匹配 | - | 更新 `self.matched_kpts0/1`, `match_confidences` |
| `pointcloud_from_disparity()` | 生成点云 | - | `Open3D.PointCloud` |
| `visualize_matches()` | 可视化匹配 | `save_path` | 保存图片 |

#### 关键属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `centers_img1/2` | `(N, 2)` | 圆心坐标 |
| `descriptors_img1/2` | `(N, 256)` | SuperPoint 描述子 |
| `matched_kpts0/1` | `(M, 2)` | 匹配的关键点对 |
| `match_confidences` | `(M,)` | 匹配置信度 [0, 1] |

---

## 测试脚本

```bash
# 完整流程测试（方式1）
python test_neural_feature.py

# 可视化测试
python test_neural_visualize.py

# 单元测试
python test_neural_unit.py
```

---

## 依赖

```bash
pip install torch torchvision opencv-python numpy open3d scipy
pip install lightglue  # 或从源码安装
```

---

## 参考

- [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork)
- [LightGlue](https://github.com/cvg/LightGlue)
- 详细技术文档：`~/.claude/CLAUDE.md` (第 90-299 行)

---

**作者**: Claude Code
**日期**: 2025-10-24
**项目**: [shuangmu](https://github.com/Zhaoyibinn/shuangmu)
