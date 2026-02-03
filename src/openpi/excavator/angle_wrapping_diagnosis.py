#!/usr/bin/env python3
"""
角度包装问题诊断脚本
检查是否存在 [-π, π] 到 [0, 2π] 的转换问题
"""

import pandas as pd
import numpy as np
from pathlib import Path

def wrap_angle_to_pi(angle):
    """将角度归一化到 [-π, π] 范围"""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

def analyze_angle_wrapping():
    """分析是否存在角度包装问题"""
    csv_path = Path("/home/er/Code/openpi-v1/openpi/output_v3/v3_predictions.csv")

    if not csv_path.exists():
        print("❌ 预测文件不存在")
        return

    df = pd.read_csv(csv_path)

    print("=" * 80)
    print("🔍 角度包装问题诊断")
    print("=" * 80)

    joint_names = ["大臂 (Boom)", "小臂 (Arm)", "铲斗 (Bucket)", "回转 (Swing)"]

    for i in range(1, 5):
        true_col = f'true_J{i}'
        pred_col = f'pred_J{i}'

        print(f"\n📊 关节 {i}: {joint_names[i-1]}")
        print("-" * 80)

        if true_col not in df.columns or pred_col not in df.columns:
            print(f"  ❌ 列不存在")
            continue

        true_vals = df[true_col].values
        pred_vals = df[pred_col].values

        # 计算不同距离下的误差
        error_raw = true_vals - pred_vals
        error_wrapped = np.array([wrap_angle_to_pi(e) for e in error_raw])

        # 计算2π修正
        corrections_needed = 0
        correction_examples = []

        for j in range(len(error_raw)):
            if abs(error_raw[j]) > 1.5 * np.pi and abs(error_wrapped[j]) < 0.5 * np.pi:
                corrections_needed += 1
                if len(correction_examples) < 5:
                    correction_examples.append((j, true_vals[j], pred_vals[j], error_raw[j], error_wrapped[j]))

        # 统计数据
        mae_raw = np.mean(np.abs(error_raw))
        mae_wrapped = np.mean(np.abs(error_wrapped))

        # 数据范围
        true_min, true_max = true_vals.min(), true_vals.max()
        pred_min, pred_max = pred_vals.min(), pred_vals.max()

        print(f"\n  📈 数据范围:")
        print(f"     真实值: [{true_min:7.4f}, {true_max:7.4f}]")
        print(f"     预测值: [{pred_min:7.4f}, {pred_max:7.4f}]")

        print(f"\n  📐 误差分析:")
        print(f"     原始 MAE:     {mae_raw:.6f} rad")
        print(f"     包装后 MAE:   {mae_wrapped:.6f} rad")
        print(f"     改进程度:     {(1 - mae_wrapped/mae_raw)*100:.1f}%")

        print(f"\n  🔄 角度包装问题:")
        print(f"     需要修正的样本: {corrections_needed}/{len(error_raw)} ({corrections_needed/len(error_raw)*100:.1f}%)")

        if correction_examples:
            print(f"\n     前5个例子:")
            for idx, tv, pv, err_raw, err_wrap in correction_examples:
                print(f"       样本 {idx}:")
                print(f"         真实: {tv:8.4f} rad,  预测: {pv:8.4f} rad")
                print(f"         原始误差: {err_raw:8.4f} rad")
                print(f"         包装误差: {err_wrap:8.4f} rad  ✅")
                print(f"         差异：{abs(err_raw) - abs(err_wrap):.4f} rad")

        # 检查是否跨越 ±π 边界
        crosses_boundary = 0
        for j in range(len(true_vals)):
            if (true_vals[j] > 0) != (pred_vals[j] > 0):
                # 符号不同，可能是跨越了边界
                if abs(true_vals[j]) > 2.5 or abs(pred_vals[j]) > 2.5:
                    crosses_boundary += 1

        if crosses_boundary > 0:
            print(f"\n  ⚠️ 检测到 {crosses_boundary} 次符号反转（可能的边界跨越）")

if __name__ == "__main__":
    analyze_angle_wrapping()
