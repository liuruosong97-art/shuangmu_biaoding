import cv2
import os
import numpy as np

from jiegouguang_class import JieGouGuang

img = cv2.imread('d455_jiegouguang_save/left2.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
H,W = img.shape

jiegouguang_class = JieGouGuang('d455_jiegouguang_save/left2.png','d455_jiegouguang_save/right2.png')
# jiegouguang_class = JieGouGuang('whx_biaoding/L/left_0041.png','whx_biaoding/R/right_0041.png')

jiegouguang_class.import_biaodin('biaoding/extrinsics_d435_20250915.yml','biaoding/intrinsics_d435_20250915.yml')
# jiegouguang_class.import_biaodin('biaoding/extrinsics_whx_zyb.yml','biaoding/intrinsics_whx_zyb.yml')


# biaoding_img = jiegouguang_class.draw_chess_board(jiegouguang_class.img1,jiegouguang_class.img2)
# biaoding_img_rectify = jiegouguang_class.draw_chess_board(jiegouguang_class.img1_rectify,jiegouguang_class.img2_rectify)

img1_with_center,img2_with_center = jiegouguang_class.extract_circle()

# img_matches = jiegouguang_class.feature_matching()

# jiegouguang_class.triangulate_points()


# img_out = np.hstack((img1_with_center, img2_with_center))


cv2.imwrite("test.png", img1_with_center)
cv2.imwrite("test2.png", img2_with_center)

# ========== 亚像素圆心检测 ==========
print("\n" + "="*60)
print("开始亚像素圆心检测")
print("="*60)

# 测试两种亚像素方法: 高斯拟合 和 质心法
methods = ['gaussian', 'centroid', 'all']
results = {}

for method in methods:
    print(f"\n{'='*40}")
    print(f"测试方法: {method.upper()}")
    print(f"{'='*40}")

    output_dir = f'subpixel_results/{method}'
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 运行亚像素检测
        centers, img_marked, comparison_imgs = jiegouguang_class.extract_circle_subpixel(
            jiegouguang_class.img1,
            method=method,
            visualize=True,
            save_path=output_dir
        )

        # 保存标记图像
        output_img = os.path.join(output_dir, f'centers_{method}.png')
        cv2.imwrite(output_img, img_marked)

        print(f"✓ 检测到 {len(centers)} 个圆心")
        print(f"✓ 标记图像保存: {output_img}")
        print(f"✓ 放大图保存: {output_dir}/spot_*.png")

        # 显示前3个圆心坐标（亚像素精度）
        if len(centers) > 0:
            print(f"前3个圆心坐标（亚像素精度）:")
            for i, (x, y) in enumerate(centers[:3]):
                print(f"  光斑 {i+1}: ({x:.3f}, {y:.3f})")

        results[method] = {
            'centers': centers,
            'count': len(centers),
            'output_dir': output_dir
        }

    except Exception as e:
        print(f"✗ 方法 {method} 失败: {e}")
        import traceback
        traceback.print_exc()

# 对比分析两种方法
print(f"\n{'='*60}")
print("亚像素方法对比总结")
print(f"{'='*60}")

print(f"\n{'方法':<15} {'检测数量':<10} {'输出目录'}")
print("-" * 50)
for method, result in results.items():
    print(f"{method:<15} {result['count']:<10} {result['output_dir']}")

# 分析两种方法的差异
if 'gaussian' in results and 'centroid' in results:
    print(f"\n亚像素精度对比分析:")
    centers_g = results['gaussian']['centers']
    centers_c = results['centroid']['centers']

    if len(centers_g) > 0 and len(centers_c) > 0:
        # 计算方法间的平均距离差异
        min_len = min(len(centers_g), len(centers_c))
        diff_gc = np.mean(np.linalg.norm(centers_g[:min_len] - centers_c[:min_len], axis=1))
        print(f"  高斯拟合 vs 质心法: {diff_gc:.4f} 像素")

print(f"\n查看20倍放大效果图:")
print(f"  ls subpixel_results/*/spot_*.png")
print(f"推荐查看 'all' 方法的放大图，可对比两种方法在同一光斑上的差异")

print("end")