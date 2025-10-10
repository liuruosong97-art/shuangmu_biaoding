#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stereo depth theoretical error sweep & control-variable plots.

公式与设定:
  fx = W / (2*tan(FOVx/2))
  d  = fx * B / Z
  σZ = (Z^2)/(fx*B) * σd
变量空间（默认）:
  分辨率:1MP(1280×960 4:3), 2MP(1920×1080 16:9), 3MP(2048×1536 4:3), 4MP(2688×1520 16:9)
  基线B:0.05, 0.055, 0.06, 0.065, 0.07, 0.075 (m)
  FOVx:66°, 70°, 76°, 80°, 86°
  σd（px）:0.5, 0.6, 0.7, 0.8, 0.9, 1.0
  Z:0.10–1.50 m, 步长 0.05 m

用法示例:
  1) 快速代表性图（默认锚点:2MP, B=0.06, FOV=76°, σd=0.7）+ 全量CSV
     python stereo_error_sweep.py --out ./out

  2) 全量图（很多图！）
     python stereo_error_sweep.py --out ./out --exhaustive

  3) 自定义锚点（例如固定 B 与 FOV）
     python stereo_error_sweep.py --out ./out --anchor-res "3MP (2048×1536, 4:3)" --anchor-b 0.065 --anchor-fov 80 --anchor-sd 0.8
"""

import math
import os
import argparse
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- 参数空间 ----------
RES_OPTIONS = [
    # {"name": "1MP (1280×960, 4:3)",   "W": 1280, "H":  960},
    # {"name": "2MP (1920×1080, 16:9)", "W": 1920, "H": 1080},
    # {"name": "3MP (2048×1536, 4:3)",  "W": 2048, "H": 1536},
    # {"name": "4MP (2688×1520, 16:9)", "W": 2688, "H": 1520},
    {"name": "8MP (3840×2140, 16:9)", "W": 3840, "H": 2140},
]
# BASELINES_M   = [0.05, 0.055, 0.06, 0.065, 0.07, 0.075]
BASELINES_M = [0.06]
# FOVX_DEG_LIST = [66, 70, 76, 80, 86]
FOVX_DEG_LIST = [76]  
# SIGMA_D_LIST  = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SIGMA_D_LIST  = [0.5]
# Z_LIST        = np.round(np.arange(0.10, 1.50 + 1e-9, 0.05), 3)  # 0.10..1.50 step 0.05
Z_LIST        = np.round(np.arange(0.10, 1.00 + 1e-9, 0.1), 3)
# ---------- 工具函数 ----------
def fx_from_fov_and_w(fovx_deg: float, W: int) -> float:
    return W / (2.0 * math.tan(math.radians(fovx_deg / 2.0)))

def compute_rows():
    rows = []
    for res in RES_OPTIONS:
        W, H = res["W"], res["H"]
        for fovx_deg in FOVX_DEG_LIST:
            fx = fx_from_fov_and_w(fovx_deg, W)
            for B in BASELINES_M:
                for sigma_d in SIGMA_D_LIST:
                    for Z in Z_LIST:
                        disparity_px = fx * B / Z
                        disparity_px_max, disparity_px_min = disparity_px+0.5, disparity_px-0.5
                        Z_max = fx * B / disparity_px_min
                        Z_min = fx * B / disparity_px_max
                        sigmaZ_m = Z_max - Z_min
                        rel_err = sigmaZ_m / Z
                        usable = disparity_px >= 2.0  # 简单可用性门槛
                        rows.append({
                            "Resolution": res["name"], "W": W, "H": H,
                            "FOVx_deg": fovx_deg, "fx_px": fx,
                            "Baseline_m": B, "Sigma_d_px": sigma_d, "Z_m": float(Z),
                            "Disparity_px": disparity_px,
                            "SigmaZ_m": sigmaZ_m,
                            "Relative_Error_%": rel_err * 100.0,
                            "Usable(d>=2px)": usable
                        })
    return pd.DataFrame(rows)

def safe_name(s: str) -> str:
    return (s.replace("×", "x").replace(":", "").replace("(", "")
             .replace(")", "").replace(" ", "").replace(",", ""))

def ensure_dir(p: str):
    if not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

# ---------- 画图函数（每张图一个变量扫，其他锚定） ----------
def plot_baseline_sweep(df, outdir, res_name, fovx, sigma_d):
    sel = df[(df["Resolution"]==res_name) & (df["FOVx_deg"]==fovx) & (df["Sigma_d_px"]==sigma_d)]
    plt.figure()
    for B in BASELINES_M:
        sub = sel[sel["Baseline_m"]==B]
        plt.plot(sub["Z_m"], sub["SigmaZ_m"], label=f"B={B:.3f} m")
    plt.xlabel("Z (m)")
    plt.ylabel("σZ (m)")
    plt.title(f"Depth Std vs Z | Baseline sweep\n{res_name}, FOVx={fovx}°, σd={sigma_d}px")
    plt.legend()
    fn = os.path.join(outdir, f"baseline_{safe_name(res_name)}_FOV{fovx}_sd{sigma_d}.png")
    plt.savefig(fn, dpi=180, bbox_inches="tight"); plt.close()
    return fn

def plot_fov_sweep(df, outdir, res_name, B, sigma_d):
    sel = df[(df["Resolution"]==res_name) & (df["Baseline_m"]==B) & (df["Sigma_d_px"]==sigma_d)]
    plt.figure()
    for fov in FOVX_DEG_LIST:
        sub = sel[sel["FOVx_deg"]==fov]
        plt.plot(sub["Z_m"], sub["SigmaZ_m"], label=f"FOVx={fov}°")
    plt.xlabel("Z (m)")
    plt.ylabel("σZ (m)")
    plt.title(f"Depth Std vs Z | FOV sweep\n{res_name}, B={B:.3f} m, σd={sigma_d}px")
    plt.legend()
    fn = os.path.join(outdir, f"fov_{safe_name(res_name)}_B{B:.3f}_sd{sigma_d}.png")
    plt.savefig(fn, dpi=180, bbox_inches="tight"); plt.close()
    return fn

def plot_resolution_sweep(df, outdir, B, fovx, sigma_d):
    sel = df[(df["Baseline_m"]==B) & (df["FOVx_deg"]==fovx) & (df["Sigma_d_px"]==sigma_d)]
    plt.figure()
    for res_name in [r["name"] for r in RES_OPTIONS]:
        sub = sel[sel["Resolution"]==res_name]
        plt.plot(sub["Z_m"], sub["SigmaZ_m"], label=res_name)
    plt.xlabel("Z (m)")
    plt.ylabel("σZ (m)")
    plt.title(f"Depth Std vs Z | Resolution sweep\nB={B:.3f} m, FOVx={fovx}°, σd={sigma_d}px")
    plt.legend()
    fn = os.path.join(outdir, f"resolution_B{B:.3f}_FOV{fovx}_sd{sigma_d}.png")
    plt.savefig(fn, dpi=180, bbox_inches="tight"); plt.close()
    return fn

def plot_sigma_d_sweep(df, outdir, res_name, B, fovx):
    sel = df[(df["Resolution"]==res_name) & (df["Baseline_m"]==B) & (df["FOVx_deg"]==fovx)]
    plt.figure()
    for sd in SIGMA_D_LIST:
        sub = sel[sel["Sigma_d_px"]==sd]
        plt.plot(sub["Z_m"], sub["SigmaZ_m"], label=f"σd={sd}px")
    plt.xlabel("Z (m)")
    plt.ylabel("σZ (m)")
    plt.title(f"Depth Std vs Z | σd sweep\n{res_name}, B={B:.3f} m, FOVx={fovx}°")
    plt.legend()
    fn = os.path.join(outdir, f"sigmad_{safe_name(res_name)}_B{B:.3f}_FOV{fovx}.png")
    plt.savefig(fn, dpi=180, bbox_inches="tight"); plt.close()
    return fn

# ---------- 批量出图（代表性 vs 全量） ----------
def run_representative_plots(df, outdir, res_anchor, B_anchor, fov_anchor, sd_anchor):
    ensure_dir(outdir)
    paths = []
    paths.append(plot_baseline_sweep  (df, outdir, res_anchor, fov_anchor, sd_anchor))
    paths.append(plot_fov_sweep       (df, outdir, res_anchor, B_anchor,   sd_anchor))
    paths.append(plot_resolution_sweep(df, outdir, B_anchor,   fov_anchor, sd_anchor))
    paths.append(plot_sigma_d_sweep   (df, outdir, res_anchor, B_anchor,   fov_anchor))
    return paths

def run_exhaustive_plots(df, outdir):
    ensure_dir(outdir)
    paths = []
    # 1) baseline sweep:对每个（res, fov, sd）生成
    for res in [r["name"] for r in RES_OPTIONS]:
        for fov in FOVX_DEG_LIST:
            for sd in SIGMA_D_LIST:
                paths.append(plot_baseline_sweep(df, outdir, res, fov, sd))
    # 2) fov sweep:对每个（res, B, sd）生成
    for res in [r["name"] for r in RES_OPTIONS]:
        for B in BASELINES_M:
            for sd in SIGMA_D_LIST:
                paths.append(plot_fov_sweep(df, outdir, res, B, sd))
    # 3) resolution sweep:对每个（B, fov, sd）生成
    for B in BASELINES_M:
        for fov in FOVX_DEG_LIST:
            for sd in SIGMA_D_LIST:
                paths.append(plot_resolution_sweep(df, outdir, B, fov, sd))
    # 4) sigma_d sweep:对每个（res, B, fov）生成
    for res in [r["name"] for r in RES_OPTIONS]:
        for B in BASELINES_M:
            for fov in FOVX_DEG_LIST:
                paths.append(plot_sigma_d_sweep(df, outdir, res, B, fov))
    return paths

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Stereo depth error sweep & plots")
    parser.add_argument("--out", type=str, default="wucha", help="输出目录")
    parser.add_argument("--csv-name", type=str, default="stereo_fullfactorial_Zsweep.csv", help="CSV 文件名")
    parser.add_argument("--exhaustive", action="store_true", help="生成全量图（很多图）")
    # 代表性图的锚点
    parser.add_argument("--anchor-res", type=str, default="2MP (1920×1080, 16:9)", help="锚点分辨率名")
    parser.add_argument("--anchor-b",   type=float, default=0.06, help="锚点基线 (m)")
    parser.add_argument("--anchor-fov", type=float, default=76.0, help="锚点FOVx (deg)")
    parser.add_argument("--anchor-sd",  type=float, default=0.7,  help="锚点σd (px)")
    args = parser.parse_args()

    ensure_dir(args.out)

    print(">> 计算全因子数据 ...")
    df = compute_rows()
    csv_path = os.path.join(args.out, args.csv-name if hasattr(args, "csv-name") else args.csv_name)  # handle hyphen attr
    # 兼容 argparse 把 --csv-name 解析成 csv_name
    csv_path = os.path.join(args.out, getattr(args, "csv_name", "stereo_fullfactorial_Zsweep.csv"))
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f">> 数据已保存: {csv_path}  (共 {len(df):,} 行)")

    print(">> 生成图表 ...")
    if args.exhaustive:
        paths = run_exhaustive_plots(df, os.path.join(args.out, "plots_all"))
    else:
        paths = run_representative_plots(
            df,
            os.path.join(args.out, "plots_rep"),
            args.anchor_res, args.anchor_b, args.anchor_fov, args.anchor_sd
        )

    print(f">> 已生成 {len(paths)} 张图，输出目录:{os.path.dirname(paths[0]) if paths else args.out}")

if __name__ == "__main__":
    main()

