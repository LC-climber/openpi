# excavator_fast_prediction_v2.py
"""
快速挖掘机预测系统 V2 - 增强可视化版
修复了乱码问题，增加了误差分析和统计图表。
"""

import numpy as np
import pandas as pd
from pathlib import Path
from openpi_client.websocket_client_policy import WebsocketClientPolicy
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tqdm import tqdm
import sys
import os

# ==========================================
# 1. 字体设置 (修复乱码核心)
# ==========================================
def configure_fonts():
    """尝试配置中文字体，如果失败则回退到英文"""
    # 常见的中文字体文件名列表 (Linux/Windows/macOS)
    font_candidates = [
        'SimHei.ttf', 'simhei.ttf', # Windows 黑体
        'Microsoft YaHei.ttf', 'msyh.ttf', # Windows 微软雅黑
        'NotoSansCJK-Regular.ttc', # Linux 通用
        'WenQuanYiMicroHei.ttf', # Linux 文泉驿
        'PingFang.ttc', # macOS
        'Arial Unicode MS.ttf'
    ]
    
    font_path = None
    # 搜索系统字体目录
    system_fonts = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    for font_file in system_fonts:
        if os.path.basename(font_file) in font_candidates:
            font_path = font_file
            break
            
    if font_path:
        print(f"✅ 找到中文字体: {font_path}")
        my_font = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = my_font.get_name()
    else:
        print("⚠️ 未找到常用中文字体，图表可能显示乱码或回退到英文。")
        # 尝试设置一个通用的 sans-serif 作为回退
        plt.rcParams['font.family'] = ['sans-serif']

    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# 运行字体配置
configure_fonts()

# ==========================================
# 主程序
# ==========================================
def fast_predict_and_plot_v2():
    print("=" * 60)
    print("🚀 快速挖掘机预测系统 V2 (增强可视化版)")
    print("=" * 60)
    
    # --- 配置 ---
    data_path = "/root/gpufree-data/lerobot_examples_490_test"
    host = "127.0.0.1"
    port = 8000
    num_samples = 1300 # 预测样本数
    
    output_dir = Path('output_v2') # 使用新的输出目录
    output_dir.mkdir(exist_ok=True)

    try:
        # --- 1. 连接与数据加载 ---
        print(f"\n[1/4] 连接服务器 ({host}:{port})...")
        policy = WebsocketClientPolicy(host=host, port=port)
        print("✅ 连接成功")
        
        print(f"\n[2/4] 加载数据 ({data_path})...")
        parquet_path = Path(data_path) / "data" / "chunk-000" / "episode_000000.parquet"
        if not parquet_path.exists():
            print(f"❌ 错误: 找不到数据文件 {parquet_path}")
            return

        df = pd.read_parquet(parquet_path)
        if num_samples < len(df):
            df = df.iloc[:num_samples]
        
        # 提取数据
        states_raw = np.array([np.array(s) for s in df['state'].tolist()])
        actions_raw = np.array([np.array(s) for s in df['action'].tolist()])
        timestamps = df['timestamp'].values
        
        # 确保只取前4个关节 (如果数据集中有多余维度)
        states = states_raw[:, :4].astype(np.float32)
        true_actions = actions_raw[:, :4].astype(np.float32)
        
        print(f"数据加载完毕: {len(states)} 个样本, 维度: {states.shape[1]}")
        
        # --- 2. 批量预测 ---
        print(f"\n[3/4] 开始预测 (Horizon=10, 启用平滑)...")
        predictions = []
        dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # [新增] 平滑变量
        smooth_action = None
        alpha = 0.8  # 平滑系数 (0.1~1.0)，越小越平滑但延迟越高，建议 0.8
        
        for i in tqdm(range(len(states)), desc="推理进度"):
            try:
                obs = {
                    "state": states[i], 
                    "image": dummy_image,
                    "prompt": "predict excavator joint action"
                }
                
                result = policy.infer(obs)
                
                # --- 核心修复区域：智能维度处理 ---
                raw_action = None
                
                if 'actions' in result:
                    actions = np.array(result['actions']) # 确保是 numpy 数组
                    
                    if actions.ndim == 3: 
                        # 形状 (Batch, Time, Dim) -> 例如 (1, 10, 32)
                        # 取第0个Batch，第0个时间步，前4维
                        raw_action = actions[0, 0, :4]
                        
                    elif actions.ndim == 2:
                        # 形状 (Time, Dim) -> 例如 (10, 32) 或 (10, 4)
                        # 取第0个时间步，前4维
                        raw_action = actions[0, :4]
                        
                    elif actions.ndim == 1:
                        # 形状 (Dim,) -> 例如 (4,)
                        # 直接取前4维
                        raw_action = actions[:4]
                        
                elif 'action' in result:
                    # 兼容旧格式 (Batch, Dim) 或 (Dim,)
                    actions = np.array(result['action'])
                    if actions.ndim == 2:
                        raw_action = actions[0, :4]
                    else:
                        raw_action = actions[:4]
                
                # 最后的安全检查
                if raw_action is None:
                    raw_action = np.array([np.nan] * 4)
                
                # --------------------

                # EMA 平滑处理
                if smooth_action is None:
                    smooth_action = raw_action
                else:
                    smooth_action = alpha * raw_action + (1 - alpha) * smooth_action
                
                predictions.append(smooth_action)
                
            except Exception as e:
                predictions.append(np.array([np.nan] * 4))
                # 打印第一个错误以便调试，后续静默
                if i == 0: print(f"⚠️ 错误: {e}")

        predictions_np = np.array(predictions)
        
        # 检查是否全部失败
        if np.all(np.isnan(predictions_np)):
            print("\n❌ 所有预测均失败，无法生成图表。请检查连接或输入格式。")
            return

        # --- 3. 增强可视化生成 (核心改进) ---
        print(f"\n[4/4] 生成增强可视化图表 (Output: {output_dir})...")
        
        # 通用设置
        joint_names = ['大臂 (Boom)', '小臂 (Arm)', '铲斗 (Bucket)', '回转 (Swing)']
        colors_true = '#1f77b4' # 蓝色
        colors_pred = '#ff7f0e' # 橙色
        colors_err = '#d62728'  # 红色
        valid_mask = ~np.isnan(predictions_np[:, 0]) # 用于过滤无效预测
        valid_len = np.sum(valid_mask)

        # -------------------------------------------------
        # 图表 A: 时序对比增强图 (带误差填充)
        # -------------------------------------------------
        fig_ts, axes_ts = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        
        for i in range(4):
            ax = axes_ts[i]
            # 绘制真实值
            ax.plot(timestamps[valid_mask], true_actions[valid_mask, i], 
                   label='真实值 (Ground Truth)', color=colors_true, linewidth=2, alpha=0.6)
            # 绘制预测值
            ax.plot(timestamps[valid_mask], predictions_np[valid_mask, i], 
                   label='预测值 (Prediction)', color=colors_pred, linewidth=1.5, linestyle='--')
            # 填充误差区域
            ax.fill_between(timestamps[valid_mask], 
                           true_actions[valid_mask, i], 
                           predictions_np[valid_mask, i], 
                           color='gray', alpha=0.2, label='差异区域')
            
            ax.set_ylabel(f'{joint_names[i]}\n弧度 (rad)', fontsize=12)
            ax.grid(True, linestyle=':', alpha=0.6)
            if i == 0: ax.legend(loc='upper right')
            if i == 3: ax.set_xlabel('时间 (秒)', fontsize=12)

        fig_ts.suptitle('图A: 挖掘机关节动作时序对比 (带差异填充)', fontsize=16, y=0.99)
        plt.tight_layout()
        fig_ts.savefig(output_dir / 'A_time_series_comparison.png', dpi=200)
        plt.close(fig_ts)

        # -------------------------------------------------
        # 图表 B: 误差分析仪表盘 (时序误差 + 散点图)
        # -------------------------------------------------
        errors = true_actions[valid_mask] - predictions_np[valid_mask]
        mae = np.mean(np.abs(errors), axis=0) # 平均绝对误差

        fig_err = plt.figure(figsize=(16, 12))
        gs = fig_err.add_gridspec(2, 4)

        for i in range(4):
            # 上排：误差随时序变化图
            ax_line = fig_err.add_subplot(gs[0, i])
            ax_line.plot(timestamps[valid_mask], errors[:, i], color=colors_err, linewidth=1, alpha=0.7)
            ax_line.axhline(0, color='black', linestyle='-', linewidth=0.8) # 0线
            # 添加参考线 (例如 +/- 0.05 rad)
            ax_line.axhline(0.05, color='gray', linestyle=':', alpha=0.5)
            ax_line.axhline(-0.05, color='gray', linestyle=':', alpha=0.5)
            ax_line.set_title(f'{joint_names[i]} 预测误差')
            ax_line.set_ylim(-0.2, 0.2) # 固定纵坐标范围方便对比
            ax_line.grid(True, alpha=0.3)
            if i == 0: ax_line.set_ylabel('误差 (True - Pred)')
            ax_line.text(0.05, 0.9, f'MAE: {mae[i]:.4f}', transform=ax_line.transAxes, 
                        bbox=dict(facecolor='white', alpha=0.7))

            # 下排：真实值 vs 预测值 散点图
            ax_scatter = fig_err.add_subplot(gs[1, i])
            ax_scatter.scatter(true_actions[valid_mask, i], predictions_np[valid_mask, i], 
                             alpha=0.1, s=10, color='purple')
            # 绘制 y=x 对角线
            lims = [
                np.min([ax_scatter.get_xlim(), ax_scatter.get_ylim()]),
                np.max([ax_scatter.get_xlim(), ax_scatter.get_ylim()]),
            ]
            ax_scatter.plot(lims, lims, 'k-', alpha=0.5, zorder=0)
            ax_scatter.set_xlabel('真实值')
            if i == 0: ax_scatter.set_ylabel('预测值')
            ax_scatter.set_title(f'{joint_names[i]} 相关性分析')
            ax_scatter.grid(True, alpha=0.3)

        fig_err.suptitle('图B: 预测误差深度分析仪表盘', fontsize=16, y=0.99)
        plt.tight_layout()
        fig_err.savefig(output_dir / 'B_error_analysis_dashboard.png', dpi=200)
        plt.close(fig_err)

        # -------------------------------------------------
        # 图表 C: 误差分布直方图
        # -------------------------------------------------
        fig_hist, axes_hist = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
        for i in range(4):
            ax = axes_hist[i]
            ax.hist(errors[:, i], bins=30, color=colors_err, alpha=0.7, edgecolor='black')
            ax.axvline(0, color='black', linestyle='-', linewidth=1)
            ax.set_title(f'{joint_names[i]}')
            ax.set_xlabel('误差范围 (rad)')
            ax.grid(True, axis='y', alpha=0.3)
            if i == 0: ax.set_ylabel('样本数量 (Count)')

        fig_hist.suptitle('图C: 预测误差分布直方图 (检查系统性偏差)', fontsize=16, y=1.05)
        plt.tight_layout()
        fig_hist.savefig(output_dir / 'C_error_distribution.png', dpi=200)
        plt.close(fig_hist)

        # --- 4. 保存 CSV ---
        results_df = pd.DataFrame({
            'timestamp': timestamps[:valid_len],
            'true_J1': true_actions[:valid_len, 0], 'pred_J1': predictions_np[:valid_len, 0],
            'true_J2': true_actions[:valid_len, 1], 'pred_J2': predictions_np[:valid_len, 1],
            'true_J3': true_actions[:valid_len, 2], 'pred_J3': predictions_np[:valid_len, 2],
            'true_J4': true_actions[:valid_len, 3], 'pred_J4': predictions_np[:valid_len, 3],
        })
        results_df.to_csv(output_dir / 'predictions.csv', index=False)

        print(f"\n✅ 所有图表已生成至目录: {output_dir.absolute()}")
        print("  - A_time_series_comparison.png (时序对比+差异填充)")
        print("  - B_error_analysis_dashboard.png (误差时序+散点图)")
        print("  - C_error_distribution.png (误差分布直方图)")
        print("  - predictions.csv (原始数据)")

    except Exception as e:
        print(f"\n❌ 程序严重错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fast_predict_and_plot_v2()