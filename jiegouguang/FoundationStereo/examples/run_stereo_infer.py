"""
Example script demonstrating StereoInference usage.

This script:
1. Loads stereo images and calibration data
2. Runs FoundationStereo inference
3. Saves disparity, depth visualizations and point cloud (PLY)

Usage:
    python examples/run_stereo_infer.py \
        --left path/to/left.png \
        --right path/to/right.png \
        --intrinsic path/to/intrinsics.txt \
        --output ./output \
        --ckpt ./pretrained_models/23-51-11/model_best_bp2.pth
"""

import sys
import os
import argparse
import numpy as np
import cv2

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stereo_inference import StereoInference


def load_intrinsics(intrinsic_file: str):
    """
    Load camera intrinsics from file.

    Expected format:
        Line 1: fx 0 cx 0 fy cy 0 0 1 (9 values for 3x3 K matrix)
        Line 2: baseline_in_meters
    """
    with open(intrinsic_file, 'r') as f:
        lines = f.readlines()
        K_values = np.array(list(map(float, lines[0].strip().split()))).astype(np.float32)
        K = K_values.reshape(3, 3)
        baseline = float(lines[1].strip())

    return K, baseline


def _colorize_metric(metric: np.ndarray, valid_mask: np.ndarray, invert: bool = False) -> np.ndarray:
    """Colorize metric values with TURBO colormap and optional inversion."""
    if valid_mask is None:
        valid_mask = metric > 0

    color_input = np.zeros_like(metric, dtype=np.float32)

    if valid_mask.any():
        values = metric[valid_mask]
        min_val = values.min()
        max_val = values.max()
        delta = max_val - min_val
        if delta > 1e-6:
            color_input[valid_mask] = (values - min_val) / delta
        else:
            color_input[valid_mask] = 0.0

    if invert:
        color_input[valid_mask] = 1.0 - color_input[valid_mask]

    color_map = cv2.applyColorMap((color_input * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    color_map[~valid_mask] = 0
    return color_map


def save_disparity_visualization(disparity: np.ndarray, output_path: str):
    """Save disparity as colorized image."""
    valid_mask = disparity > 0
    disp_color = _colorize_metric(disparity, valid_mask, invert=False)
    cv2.imwrite(output_path, disp_color)
    print(f"Saved disparity visualization to: {output_path}")


def save_depth_visualization(depth: np.ndarray, valid_mask: np.ndarray, output_path: str,
                             mode: str = "match_disparity"):
    """Save depth as colorized image with configurable color mapping."""
    assert mode in {"match_disparity", "perceptual"}, "Invalid depth visualization mode"

    if mode == "match_disparity":
        inv_depth = np.zeros_like(depth, dtype=np.float32)
        inv_depth[valid_mask] = 1.0 / np.clip(depth[valid_mask], 1e-6, None)
        depth_color = _colorize_metric(inv_depth, valid_mask, invert=False)
    else:
        depth_color = _colorize_metric(depth, valid_mask, invert=True)

    cv2.imwrite(output_path, depth_color)
    print(f"Saved depth visualization to: {output_path} (mode={mode})")


def save_point_cloud_ply(points: np.ndarray, colors: np.ndarray, valid_mask: np.ndarray,
                         output_path: str):
    """
    Save point cloud as PLY file.

    Args:
        points: (H, W, 3) point cloud in meters
        colors: (H, W, 3) RGB colors (0-255)
        valid_mask: (H, W) boolean mask
        output_path: Output PLY file path
    """
    # Extract valid points and colors
    valid_points = points[valid_mask]
    valid_colors = colors[valid_mask]

    num_points = len(valid_points)

    # Write PLY header
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # Write points
        for i in range(num_points):
            x, y, z = valid_points[i]
            r, g, b = valid_colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

    print(f"Saved point cloud with {num_points} points to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="FoundationStereo inference example")
    parser.add_argument('--left', type=str, required=True, help='Path to left image')
    parser.add_argument('--right', type=str, required=True, help='Path to right image')
    parser.add_argument('--intrinsic', type=str, required=True, help='Path to intrinsics file')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    parser.add_argument('--ckpt', type=str, default=None,
                       help='Path to model checkpoint (auto-detect if not provided)')
    parser.add_argument('--scale', type=float, default=0.5,
                       help='Image scale factor (0.3-1.0, lower for less memory)')
    parser.add_argument('--iters', type=int, default=32,
                       help='Number of refinement iterations')
    parser.add_argument('--vit_size', type=str, default='vitl',
                       help='ViT backbone size (vitl, vits, vitb, vitg)')
    parser.add_argument('--depth_color_mode', type=str, default='match_disparity',
                       choices=['match_disparity', 'perceptual'],
                       help='Depth visualization mode: match_disparity aligns colors with disparity, '
                            'perceptual keeps the original near-warm mapping')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print("="*70)
    print("FoundationStereo Inference Example")
    print("="*70)

    # Load images
    print(f"\n1. Loading images...")
    left_img = cv2.imread(args.left)
    right_img = cv2.imread(args.right)

    if left_img is None or right_img is None:
        print("Error: Could not load images")
        return

    print(f"   Left image: {left_img.shape}")
    print(f"   Right image: {right_img.shape}")

    # Apply scale
    if args.scale != 1.0:
        H, W = left_img.shape[:2]
        new_H, new_W = int(H * args.scale), int(W * args.scale)
        left_img = cv2.resize(left_img, (new_W, new_H))
        right_img = cv2.resize(right_img, (new_W, new_H))
        print(f"   Scaled to: {left_img.shape} (scale={args.scale})")

    # Load calibration
    print(f"\n2. Loading calibration from: {args.intrinsic}")
    K_rect, baseline_m = load_intrinsics(args.intrinsic)

    # Scale intrinsics if needed
    if args.scale != 1.0:
        K_rect[:2] *= args.scale

    print(f"   Intrinsics K:")
    print(f"     fx={K_rect[0,0]:.2f}, fy={K_rect[1,1]:.2f}")
    print(f"     cx={K_rect[0,2]:.2f}, cy={K_rect[1,2]:.2f}")
    print(f"   Baseline: {baseline_m*1000:.2f}mm")

    # Initialize inference
    print(f"\n3. Loading model...")
    if args.ckpt:
        print(f"   Using specified checkpoint: {args.ckpt}")
        stereo_infer = StereoInference(args.ckpt, vit_size=args.vit_size)
    else:
        print(f"   Auto-detecting model checkpoint...")
        stereo_infer = StereoInference(vit_size=args.vit_size)  # Auto-detect

    # Run inference
    print(f"\n4. Running inference (iterations={args.iters})...")
    results = stereo_infer.infer(
        left_img, right_img, K_rect, baseline_m,
        valid_iters=args.iters
    )

    # Print statistics
    print(f"\n5. Results:")
    disparity = results['disparity']
    depth = results['depth']
    points = results['points_cam1']
    valid_mask = results['valid_mask']

    valid_depth = depth[valid_mask]
    valid_disp = disparity[valid_mask]

    print(f"   Valid pixels: {valid_mask.sum()}/{valid_mask.size} ({100*valid_mask.sum()/valid_mask.size:.1f}%)")
    if len(valid_depth) > 0:
        print(f"   Disparity range: {valid_disp.min():.2f} - {valid_disp.max():.2f} pixels")
        print(f"   Depth range: {valid_depth.min():.3f} - {valid_depth.max():.3f} m")
        print(f"   Mean depth: {valid_depth.mean():.3f} m")
        print(f"   Median depth: {np.median(valid_depth):.3f} m")
    else:
        print("   Warning: No valid depth values")

    # Save outputs
    print(f"\n6. Saving outputs to: {args.output}")

    # Save disparity
    disp_path = os.path.join(args.output, 'disparity.png')
    save_disparity_visualization(disparity, disp_path)

    # Save depth
    depth_path = os.path.join(args.output, 'depth.png')
    save_depth_visualization(depth, valid_mask, depth_path, mode=args.depth_color_mode)

    # Save raw depth as numpy array
    depth_npy_path = os.path.join(args.output, 'depth_meter.npy')
    np.save(depth_npy_path, depth)
    print(f"Saved raw depth data to: {depth_npy_path}")

    # Save point cloud
    # Convert BGR to RGB for colors
    colors = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    ply_path = os.path.join(args.output, 'pointcloud.ply')
    save_point_cloud_ply(points, colors, valid_mask, ply_path)

    print("\n" + "="*70)
    print("Inference complete!")
    print("="*70)


if __name__ == "__main__":
    main()
