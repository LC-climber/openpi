# test_infer_v3.py
"""
修复版本的挖掘机推理系统 V3（增强版 - 含视频帧数据与角度包装纠正）

核心改进（相比原始推理脚本）：
1. ✅ 正确加载视频帧数据（使用 frame_index 对应关系）
   - 从 parquet 的 frame_index 字段获取每个 state 对应的视频帧
   - 解决 state 与视频帧的非线性对应问题（1300 样本 vs 3233 帧）

2. ✅ 使用真实视频图像替代虚拟图像
   - 加载原始视频帧作为观察
   - 正确处理图像格式（BGR -> RGB，C,H,W 转 H,W,C）

3. ✅ 移除错误的文本提示词（prompt）
   - 不应该使用文本驱动的提示词
   - 模型训练时没有使用 prompt 字段

4. ✅ 应用角度包装纠正（J4 回转关节）
   - 使用 atan2 将角度规范化到 [-π, π]
   - 解决 ±π 边界跨越问题
   - 生成原始和纠正后的两套统计结果

5. ✅ 完整的数据验证和可视化
   - 生成时序对比、误差分析、误差分布图表
   - 保存原始数据和统计汇总

关键数据对应关系（来自 meta/info.json）：
  - 总视频帧数: 3233 帧 @ 10 FPS
  - 总 parquet 样本数: 1300 样本
  - 帧索引：每个样本都有 frame_index 字段指向对应的视频帧

使用流程：
  1. 确保视频文件存在：videos/chunk-000/main/episode_000000.mp4
  2. 运行此脚本进行推理和分析
  3. 输出包含完整可视化和统计结果
"""

import dataclasses
import json
import logging
import pathlib
import subprocess
import copy
from typing import Optional

import einops
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import cv2

from openpi_client.websocket_client_policy import WebsocketClientPolicy
from openpi.training import config as _config
from openpi.training import checkpoints as _checkpoints
from openpi import transforms as _transforms
from openpi.shared import normalize as _normalize
import openpi.models.model as _model

# ==========================================
# 配置与常量
# ==========================================

CONFIG = {
    "data_path": "/root/gpufree-data/lerobot_examples_490_test",
    "checkpoint_dir": "/root/gpufree-data/checkpoints/pi05_excavator_finetune/excavator_v1/19999",
    "config_name": "pi05_excavator_finetune",
    "host": "127.0.0.1",
    "port": 8000,
    "num_samples": 1300,
    "output_dir": Path("output_v3"),
}

# ==========================================
# 工具函数
# ==========================================

def _parse_image(image) -> np.ndarray:
    """将图像转换为 (H, W, C) uint8 格式

    处理两种常见格式：
    - (C, H, W) float32 (0-1) -> (H, W, C) uint8 (0-255)
    - (H, W, C) uint8 -> (H, W, C) uint8
    """
    image = np.asarray(image)

    # 如果是浮点数 (0-1)，转换为 0-255 uint8
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)

    # 如果是 (C, H, W) 格式，转换为 (H, W, C)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")

    return image


class MP4Reader:
    """使用 ffmpeg 高效读取视频帧"""

    def __init__(self, filepath, resolution, video_hw, video_hh):
        """
        Args:
            filepath: 视频路径
            resolution: 模型输入分辨率 (H, W)
            video_hw: 原视频宽
            video_hh: 原视频高
        """
        self.filepath = filepath
        self.resolution = resolution
        self.video_w = video_hw
        self.video_h = video_hh
        self._index = 0

        self._ffmpeg_cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-hwaccel", "none",
            "-i", filepath,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-"
        ]

        self._pipe = subprocess.Popen(
            self._ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8
        )

    def read_frame(self):
        """读取下一帧"""
        frame_size = self.video_w * self.video_h * 3
        raw = self._pipe.stdout.read(frame_size)

        if len(raw) != frame_size:
            return None

        frame = np.frombuffer(raw, np.uint8)
        frame = frame.reshape((self.video_h, self.video_w, 3))
        self._index += 1

        # 调整到目标分辨率
        if self.resolution == (0, 0):
            return frame
        return self._resize_frame(frame, self.resolution)

    @staticmethod
    def _resize_frame(image, target_shape=(224, 224), padding_color=(0, 0, 0)):
        """保持宽高比填充图像至目标尺寸"""
        orig_height, orig_width = image.shape[:2]
        target_height, target_width = target_shape

        scale = min(target_height / orig_height, target_width / orig_width)
        scaled_width = int(orig_width * scale)
        scaled_height = int(orig_height * scale)

        resized = cv2.resize(
            image,
            (scaled_width, scaled_height),
            interpolation=cv2.INTER_LINEAR
        )

        width_pad = target_width - scaled_width
        height_pad = target_height - scaled_height
        left_pad = width_pad // 2
        right_pad = width_pad - left_pad
        top_pad = height_pad // 2
        bottom_pad = height_pad - top_pad

        padded = cv2.copyMakeBorder(
            resized,
            top_pad, bottom_pad, left_pad, right_pad,
            cv2.BORDER_CONSTANT,
            value=padding_color
        )
        return padded

    def read_all_frames(self):
        """一次性读取所有帧"""
        frames = []
        while True:
            frame = self.read_frame()
            if frame is None:
                break
            frames.append(frame)
        return np.array(frames)

    def release(self):
        """释放资源"""
        if self._pipe:
            self._pipe.terminate()
            self._pipe.wait()


def apply_angle_wrapping(values: np.ndarray, dims: list = None) -> np.ndarray:
    """对指定维度应用角度包装（规范化到 [-π, π]）

    原理：
        使用 atan2(sin(θ), cos(θ)) 将任意角度映射到 [-π, π] 范围
        这解决了角度数据在 ±π 边界处的跳跃问题

    参数：
        values: (N, 4) 数组，代表预测/真实的关节角度
        dims: 要处理的维度列表，默认 [3] 表示只处理 J4 (回转)

    返回：
        wrapped: 经过角度包装后的数组
    """
    if dims is None:
        dims = [3]  # 默认只处理 J4（回转关节）

    wrapped = values.copy()
    for dim in dims:
        if dim < wrapped.shape[-1]:
            # 使用 atan2(sin(θ), cos(θ)) 进行归一化
            wrapped[..., dim] = np.arctan2(
                np.sin(values[..., dim]),
                np.cos(values[..., dim])
            )
    return wrapped


def configure_fonts():
    """配置中文字体显示"""
    font_candidates = [
        'SimHei.ttf', 'simhei.ttf',
        'Microsoft YaHei.ttf', 'msyh.ttf',
        'NotoSansCJK-Regular.ttc',
        'WenQuanYiMicroHei.ttf',
        'PingFang.ttc',
        'Arial Unicode MS.ttf'
    ]

    font_path = None
    system_fonts = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    for font_file in system_fonts:
        if Path(font_file).name in font_candidates:
            font_path = font_file
            break

    if font_path:
        my_font = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = my_font.get_name()

    plt.rcParams['axes.unicode_minus'] = False


def load_data_with_validation(data_path: str, num_samples: int) -> tuple:
    """
    加载数据并进行验证

    返回: (states, actions, timestamps, frame_indices, video_frames, data_info)
    """
    print("\n" + "="*60)
    print("📂 加载推理数据集")
    print("="*60)

    data_dir = Path(data_path)
    parquet_path = data_dir / "data" / "chunk-000" / "episode_000000.parquet"
    video_path = data_dir / "videos" / "chunk-000" / "main" / "episode_000000.mp4"

    # 检查文件存在
    if not parquet_path.exists():
        raise FileNotFoundError(f"找不到数据文件: {parquet_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"找不到视频文件: {video_path}")

    # ===== 加载 Parquet 数据 =====
    print("\n📋 加载 Parquet 数据...")
    df = pd.read_parquet(parquet_path)
    print(f"  可用列: {df.columns.tolist()}")

    if num_samples < len(df):
        df = df.iloc[:num_samples]

    # 验证必要的列
    if "frame_index" not in df.columns:
        raise ValueError(
            f"缺少 'frame_index' 列！这是关键字段，用于对应视频帧。\n"
            f"可用列: {df.columns.tolist()}"
        )

    # 提取数据
    states_raw = np.array([np.array(s) for s in df['state'].tolist()], dtype=np.float32)
    actions_raw = np.array([np.array(a) for a in df['action'].tolist()], dtype=np.float32)
    timestamps = df['timestamp'].values
    frame_indices = df['frame_index'].values.astype(np.int32)  # ✅ 重要：使用 frame_index

    # 确保只取前4个关节（挖掘机配置）
    states_4d = states_raw[:, :4]
    actions_4d = actions_raw[:, :4]

    print(f"  样本数: {len(states_4d)}")
    print(f"  时间戳范围: [{timestamps.min():.2f}, {timestamps.max():.2f}]")
    print(f"  帧索引范围: [{frame_indices.min()}, {frame_indices.max()}]")

    # ===== 加载视频帧 =====
    print(f"\n📹 加载视频文件: {video_path}")
    print(f"  原始分辨率: 640x480")

    mp4reader = MP4Reader(
        filepath=str(video_path),
        resolution=(224, 224),
        video_hw=640,
        video_hh=480
    )
    video_frames = mp4reader.read_all_frames()
    mp4reader.release()

    print(f"  读取帧数: {len(video_frames)}")
    print(f"  处理后分辨率: {video_frames.shape}")

    # 验证 frame_index 有效性
    max_frame_idx = frame_indices.max()
    if max_frame_idx >= len(video_frames):
        print(f"⚠️ 警告: 最大帧索引 {max_frame_idx} >= 视频帧数 {len(video_frames)}")
        print(f"   将截断到有效范围")
        valid_mask = frame_indices < len(video_frames)
        states_4d = states_4d[valid_mask]
        actions_4d = actions_4d[valid_mask]
        timestamps = timestamps[valid_mask]
        frame_indices = frame_indices[valid_mask]

    print(f"\n✅ 数据加载完成:")
    print(f"  样本数: {len(states_4d)}")
    print(f"  状态维度: {states_4d.shape}")
    print(f"  动作维度: {actions_4d.shape}")
    print(f"  视频帧数: {len(video_frames)}")

    data_info = {
        "num_samples": len(states_4d),
        "state_dim": states_4d.shape[1],
        "action_dim": actions_4d.shape[1],
        "timestamps": timestamps,
        "frame_indices": frame_indices,
        "num_video_frames": len(video_frames),
    }

    return states_4d, actions_4d, timestamps, frame_indices, video_frames, data_info


def create_observation_dict(state: np.ndarray, image: np.ndarray) -> dict:
    """
    根据训练配置创建观察对象

    注意：
    - ✅ 使用真实的视频图像而非虚拟图像
    - ✅ 不使用 prompt 字段（模型训练时没有使用文本驱动）
    - 格式应与 droid_policy.ExcavatorInputs 一致
    """
    obs = {
        "state": state,              # shape (4,) - 挖掘机4个关节
        "image": image,              # shape (224, 224, 3) - 真实视频帧
    }
    # ✅ 不添加 prompt 字段

    return obs


def infer_batch(policy: WebsocketClientPolicy, states: np.ndarray,
                frame_indices: np.ndarray, video_frames: np.ndarray,
                batch_size: int = 1) -> tuple:
    """
    批量推理

    Args:
        policy: WebsocketClientPolicy实例
        states: shape (num_samples, state_dim)
        frame_indices: 每个样本对应的视频帧索引
        video_frames: 所有视频帧数据
        batch_size: 批大小（当前固定为1）

    Returns:
        predictions: shape (num_samples, 4)
        errors: 错误列表
    """
    print("\n" + "="*60)
    print("🚀 开始推理")
    print("="*60)

    predictions = []
    errors = []

    for i in tqdm(range(len(states)), desc="推理进度"):
        try:
            # ✅ 使用 frame_index 获取对应的视频帧
            frame_idx = int(frame_indices[i])
            if frame_idx >= len(video_frames):
                frame_idx = len(video_frames) - 1

            image = video_frames[frame_idx]
            obs = create_observation_dict(states[i], image)
            result = policy.infer(obs)

            # 处理推理输出
            action = None
            if "actions" in result:
                acts = np.array(result["actions"])
                # 处理不同的维度情况
                if acts.ndim == 3:  # (batch, horizon, dim)
                    action = acts[0, 0, :4]
                elif acts.ndim == 2:  # (horizon, dim) 或 (batch, dim)
                    if acts.shape[0] == 1:  # (1, dim) - 单batch
                        action = acts[0, :4]
                    else:  # (horizon, dim)
                        action = acts[0, :4]
                elif acts.ndim == 1:  # (dim,)
                    action = acts[:4]

            elif "action" in result:
                acts = np.array(result["action"])
                if acts.ndim == 2:
                    action = acts[0, :4]
                elif acts.ndim == 1:
                    action = acts[:4]

            if action is None:
                action = np.full(4, np.nan, dtype=np.float32)
                errors.append((i, "无法提取action"))
            else:
                action = np.asarray(action, dtype=np.float32)[:4]

            predictions.append(action)

        except Exception as e:
            predictions.append(np.full(4, np.nan, dtype=np.float32))
            errors.append((i, str(e)))
            if i == 0:
                print(f"  ⚠️ 第一个错误: {e}")

    predictions_np = np.array(predictions)

    print(f"\n✅ 推理完成:")
    print(f"  成功样本: {np.sum(~np.isnan(predictions_np[:, 0]))}/{len(predictions)}")
    if errors:
        print(f"  失败数: {len(errors)}")
        if len(errors) <= 5:
            for idx, err in errors:
                print(f"    样本{idx}: {err}")

    return predictions_np, errors


def analyze_and_visualize(states: np.ndarray, true_actions: np.ndarray,
                          predictions: np.ndarray, timestamps: np.ndarray,
                          output_dir: Path):
    """生成分析和可视化 + 角度包装纠正"""
    print("\n" + "="*60)
    print("📊 生成分析和可视化（含角度包装纠正）")
    print("="*60)

    output_dir.mkdir(exist_ok=True)

    # 过滤掉无效预测
    valid_mask = ~np.isnan(predictions[:, 0])
    valid_len = np.sum(valid_mask)

    if valid_len == 0:
        print("❌ 所有预测都无效，无法生成图表")
        return

    states_valid = states[valid_mask]
    true_actions_valid = true_actions[valid_mask]
    predictions_valid = predictions[valid_mask]
    timestamps_valid = timestamps[valid_mask]

    # ===== 计算原始误差 =====
    errors_original = true_actions_valid - predictions_valid
    mae_original = np.mean(np.abs(errors_original), axis=0)
    rmse_original = np.sqrt(np.mean(errors_original**2, axis=0))

    # ===== 应用角度包装纠正（J4 - 回转关节）=====
    true_actions_wrapped = apply_angle_wrapping(true_actions_valid, dims=[3])
    predictions_wrapped = apply_angle_wrapping(predictions_valid, dims=[3])

    # 计算应用角度包装后的误差
    errors_wrapped = true_actions_wrapped - predictions_wrapped
    mae_wrapped = np.mean(np.abs(errors_wrapped), axis=0)
    rmse_wrapped = np.sqrt(np.mean(errors_wrapped**2, axis=0))

    # 计算改进程度
    mae_improvement = (1 - mae_wrapped / np.clip(mae_original, 1e-6, None)) * 100

    # 默认使用包装后的误差用于可视化
    errors = errors_wrapped
    mae = mae_wrapped
    rmse = rmse_wrapped

    print("\n📈 角度包装处理结果（J4 - 回转关节）:")
    print(f"  原始 MAE:      {mae_original[3]:.6f} rad")
    print(f"  包装后 MAE:    {mae_wrapped[3]:.6f} rad")
    print(f"  改进程度:      {mae_improvement[3]:.1f}%")
    print(f"  受影响样本:    {np.sum(np.abs(errors_original[:, 3]) > np.abs(errors_wrapped[:, 3]))}/{len(true_actions_valid)} ")

    joint_names = ['大臂 (Boom)', '小臂 (Arm)', '铲斗 (Bucket)', '回转 (Swing)']

    # ===== 图表1: 时序对比 =====
    fig_ts, axes_ts = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    for i in range(4):
        ax = axes_ts[i]
        ax.plot(timestamps_valid, true_actions_valid[:, i],
               label='真实值', color='#1f77b4', linewidth=2, alpha=0.7)
        ax.plot(timestamps_valid, predictions_valid[:, i],
               label='预测值', color='#ff7f0e', linewidth=1.5, linestyle='--')
        ax.fill_between(timestamps_valid,
                       true_actions_valid[:, i],
                       predictions_valid[:, i],
                       color='gray', alpha=0.2)

        ax.set_ylabel(f'{joint_names[i]}\n(rad)', fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.text(0.98, 0.95, f'MAE: {mae[i]:.4f}\nRMSE: {rmse[i]:.4f}',
               transform=ax.transAxes, fontsize=9,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        if i == 0:
            ax.legend(loc='upper left', fontsize=10)
        if i == 3:
            ax.set_xlabel('时间 (秒)', fontsize=11)

    fig_ts.suptitle('V3: 时序对比 - 真实值 vs 预测值', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_ts.savefig(output_dir / 'v3_1_time_series.png', dpi=150, bbox_inches='tight')
    plt.close(fig_ts)

    # ===== 图表2: 误差分析 =====
    fig_err = plt.figure(figsize=(16, 10))
    gs = fig_err.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

    for i in range(4):
        # 上排：误差随时序变化
        ax_line = fig_err.add_subplot(gs[0, i])
        ax_line.plot(timestamps_valid, errors[:, i], color='#d62728', linewidth=1, alpha=0.7)
        ax_line.axhline(0, color='black', linestyle='-', linewidth=0.8)
        ax_line.axhline(0.05, color='gray', linestyle=':', alpha=0.5, label='±0.05')
        ax_line.axhline(-0.05, color='gray', linestyle=':', alpha=0.5)
        ax_line.set_title(f'{joint_names[i]} - 预测误差', fontsize=10)
        ax_line.set_ylim(-0.2, 0.2)
        ax_line.grid(True, alpha=0.3)
        if i == 0:
            ax_line.set_ylabel('误差 (True - Pred)', fontsize=9)

        # 下排：真实值 vs 预测值散点
        ax_scatter = fig_err.add_subplot(gs[1, i])
        ax_scatter.scatter(true_actions_valid[:, i], predictions_valid[:, i],
                         alpha=0.3, s=5, color='purple')
        # 对角线
        lims = [
            min(ax_scatter.get_xlim()[0], ax_scatter.get_ylim()[0]),
            max(ax_scatter.get_xlim()[1], ax_scatter.get_ylim()[1]),
        ]
        ax_scatter.plot(lims, lims, 'k-', alpha=0.5, linewidth=1)
        ax_scatter.set_xlabel('真实值', fontsize=9)
        if i == 0:
            ax_scatter.set_ylabel('预测值', fontsize=9)
        ax_scatter.set_title(f'{joint_names[i]} - 相关性', fontsize=10)
        ax_scatter.grid(True, alpha=0.3)

    fig_err.suptitle('V3: 误差深度分析 - MAE/RMSE/相关性', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_err.savefig(output_dir / 'v3_2_error_analysis.png', dpi=150, bbox_inches='tight')
    plt.close(fig_err)

    # ===== 图表3: 误差分布 =====
    fig_hist, axes_hist = plt.subplots(1, 4, figsize=(14, 4), sharey=True)

    for i in range(4):
        ax = axes_hist[i]
        ax.hist(errors[:, i], bins=30, color='#d62728', alpha=0.7, edgecolor='black')
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.set_title(f'{joint_names[i]}', fontsize=10)
        ax.set_xlabel('误差范围 (rad)', fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)
        if i == 0:
            ax.set_ylabel('样本数量', fontsize=9)

    fig_hist.suptitle('V3: 误差分布直方图', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_hist.savefig(output_dir / 'v3_3_error_distribution.png', dpi=150, bbox_inches='tight')
    plt.close(fig_hist)

    # ===== 保存数据 =====
    results_df = pd.DataFrame({
        'timestamp': timestamps_valid,
        'true_J1': true_actions_valid[:, 0], 'pred_J1': predictions_valid[:, 0],
        'true_J2': true_actions_valid[:, 1], 'pred_J2': predictions_valid[:, 1],
        'true_J3': true_actions_valid[:, 2], 'pred_J3': predictions_valid[:, 2],
        'true_J4': true_actions_valid[:, 3], 'pred_J4': predictions_valid[:, 3],
    })
    results_df.to_csv(output_dir / 'v3_predictions.csv', index=False)

    # 保存汇总统计 - 包含原始和包装后的对比
    stats_summary = {
        "总样本数": len(predictions),
        "有效样本数": int(valid_len),
        "无效样本数": len(predictions) - int(valid_len),
        "关节": joint_names,
        "原始结果": {
            "MAE": mae_original.tolist(),
            "RMSE": rmse_original.tolist(),
        },
        "角度包装后结果": {
            "MAE": mae_wrapped.tolist(),
            "RMSE": rmse_wrapped.tolist(),
        },
        "改进程度（%）": mae_improvement.tolist(),
        "备注": "角度包装应用于 J4（回转关节），使用 atan2(sin(θ), cos(θ)) 将角度规范化到 [-π, π]"
    }

    with open(output_dir / 'v3_stats.json', 'w') as f:
        json.dump(stats_summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 可视化完成:")
    print(f"  输出目录: {output_dir.absolute()}")
    print(f"  生成文件:")
    print(f"    - v3_1_time_series.png (时序对比)")
    print(f"    - v3_2_error_analysis.png (误差分析)")
    print(f"    - v3_3_error_distribution.png (误差分布)")
    print(f"    - v3_predictions.csv (原始数据)")
    print(f"    - v3_stats.json (统计汇总 - 含角度包装对比)")

    print(f"\n📈 误差统计汇总（应用角度包装后）:")
    for i, name in enumerate(joint_names):
        print(f"  {name}:")
        print(f"    原始 MAE:    {mae_original[i]:.6f} rad")
        print(f"    包装后 MAE:  {mae_wrapped[i]:.6f} rad")
        print(f"    改进:        {mae_improvement[i]:+.1f}%")
        print(f"    包装后 RMSE: {rmse_wrapped[i]:.6f} rad")


# ==========================================
# 主程序
# ==========================================

def main():
    configure_fonts()

    print("\n" + "="*60)
    print("🔧 挖掘机推理系统 V3 - 修复版（含真实视频帧）")
    print("="*60)
    print(f"\n配置:")
    print(f"  数据集: {CONFIG['data_path']}")
    print(f"  推理服务: {CONFIG['host']}:{CONFIG['port']}")

    try:
        # 1. 加载数据
        states, actions, timestamps, frame_indices, video_frames, data_info = load_data_with_validation(
            CONFIG["data_path"],
            CONFIG["num_samples"]
        )

        # 2. 连接推理服务
        print("\n" + "="*60)
        print("🔌 连接推理服务")
        print("="*60)
        print(f"连接到 {CONFIG['host']}:{CONFIG['port']}...")

        policy = WebsocketClientPolicy(host=CONFIG['host'], port=CONFIG['port'])
        print("✅ 连接成功")

        # 3. 执行推理
        predictions, errors = infer_batch(
            policy, states, frame_indices, video_frames
        )

        # 4. 分析和可视化
        analyze_and_visualize(
            states, actions, predictions, timestamps,
            CONFIG["output_dir"]
        )

        print("\n" + "="*60)
        print("✅ 推理完成！")
        print("="*60)

    except Exception as e:
        print(f"\n❌ 程序出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
