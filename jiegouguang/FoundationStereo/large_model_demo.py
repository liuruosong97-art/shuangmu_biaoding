"""
Large Model Quick Start Guide for StereoInference

Demonstrates usage of the 3.1GB FoundationStereo ViT-Large model
with automatic detection and configuration.
"""

import sys
import os
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stereo_inference import StereoInference


def demo_large_model():
    """Demonstrate large model auto-detection and usage."""
    print("🚀 FoundationStereo Large Model Demo")
    print("="*50)

    # 1. Auto-detect and load large model (3.1GB)
    print("\n📦 Loading large model...")
    stereo_infer = StereoInference()  # Auto-detect external model

    # 2. Create test data
    print("\n🖼️  Creating test stereo pair...")
    H, W = 480, 640
    left_img = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
    right_img = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)

    # Add some correlation between left/right for realistic disparity
    shift = 20
    right_img[:, shift:] = left_img[:, :-shift] + np.random.randint(-10, 10, (H, W-shift, 3))

    # 3. Camera calibration (typical values)
    fx = fy = 400.0
    cx, cy = W/2, H/2
    K_rect = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    baseline_m = 0.095  # 95mm RealSense baseline

    print(f"   Image size: {H}×{W}")
    print(f"   Focal length: {fx:.1f} pixels")
    print(f"   Baseline: {baseline_m*1000:.1f}mm")

    # 4. Run inference
    print("\n⚡ Running large model inference...")
    results = stereo_infer.infer(
        left_img, right_img, K_rect, baseline_m,
        valid_iters=16  # Faster for demo
    )

    # 5. Display results
    print("\n📊 Results:")
    disparity = results['disparity']
    depth = results['depth']
    valid_mask = results['valid_mask']

    valid_pixels = valid_mask.sum()
    total_pixels = valid_mask.size

    print(f"   Valid pixels: {valid_pixels:,}/{total_pixels:,} ({100*valid_pixels/total_pixels:.1f}%)")

    if valid_pixels > 0:
        valid_disp = disparity[valid_mask]
        valid_depth = depth[valid_mask]

        print(f"   Disparity range: {valid_disp.min():.1f} - {valid_disp.max():.1f} pixels")
        print(f"   Depth range: {valid_depth.min():.3f} - {valid_depth.max():.3f} m")
        print(f"   Mean depth: {valid_depth.mean():.3f} m")

    print("\n✅ Large model demo completed successfully!")


def demo_model_comparison():
    """Compare different model configurations."""
    print("\n🔄 Model Configuration Comparison")
    print("="*50)

    # Test different ViT sizes
    vit_sizes = ['vitl']  # Large model

    for vit_size in vit_sizes:
        print(f"\n📋 Testing ViT-{vit_size.upper()}:")

        try:
            # Load model with specific ViT size
            stereo_infer = StereoInference(vit_size=vit_size)

            # Quick test
            test_img = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
            K_test = np.array([[200, 0, 160], [0, 200, 120], [0, 0, 1]], dtype=np.float32)

            results = stereo_infer.infer(
                test_img, test_img, K_test, 0.1,
                valid_iters=8  # Fast test
            )

            valid_ratio = results['valid_mask'].sum() / results['valid_mask'].size
            print(f"   ✅ {vit_size}: {valid_ratio:.1%} valid pixels")

        except Exception as e:
            print(f"   ❌ {vit_size}: {str(e)}")


def demo_usage_patterns():
    """Show different usage patterns."""
    print("\n🎯 Usage Patterns")
    print("="*50)

    # Pattern 1: Auto-detection (recommended)
    print("\n1️⃣  Auto-detection (recommended):")
    print("```python")
    print("stereo_infer = StereoInference()  # Auto-detect best model")
    print("```")

    # Pattern 2: Explicit path
    print("\n2️⃣  Explicit model path:")
    print("```python")
    print("stereo_infer = StereoInference('/media/root123/DX/pretrained_models/23-51-11/model_best_bp2.pth')")
    print("```")

    # Pattern 3: Custom ViT size
    print("\n3️⃣  Custom ViT backbone:")
    print("```python")
    print("stereo_infer = StereoInference(vit_size='vitl')  # Large ViT")
    print("```")

    # Pattern 4: Device specification
    print("\n4️⃣  Device specification:")
    print("```python")
    print("stereo_infer = StereoInference(device='cuda')  # Force CUDA")
    print("```")


if __name__ == "__main__":
    try:
        demo_large_model()
        demo_model_comparison()
        demo_usage_patterns()

        print("\n" + "="*50)
        print("🎉 All demos completed successfully!")
        print("="*50)

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("\nPlease ensure:")
        print("1. Large model exists at /media/root123/DX/pretrained_models/23-51-11/")
        print("2. CUDA environment is properly configured")
        print("3. All dependencies are installed")