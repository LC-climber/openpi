#!/usr/bin/env python3
"""
挖掘机推理系统 - 快速诊断脚本

使用方法：
  python3 quick_diagnosis.py

该脚本会自动执行所有诊断步骤，并生成诊断报告。
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime

class QuickDiagnosis:
    def __init__(self):
        self.report = {
            'timestamp': datetime.now().isoformat(),
            'findings': {},
            'recommendations': [],
            'severity': 'unknown'
        }
        self.output_dir = Path("/home/er/Code/openpi-v1/output_v3")
        self.log_dir = Path("/home/er/Code/openpi-v1/logs")
        self.log_dir.mkdir(exist_ok=True)

    def log(self, msg: str, level: str = "INFO"):
        """打印并记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prefix = {
            "INFO": "ℹ️ ",
            "SUCCESS": "✅",
            "WARNING": "⚠️ ",
            "ERROR": "❌",
            "DEBUG": "🔍"
        }.get(level, "→ ")

        full_msg = f"{prefix} [{timestamp}] {msg}"
        print(full_msg)

    def check_output_exists(self) -> bool:
        """检查 output_v3 是否存在"""
        self.log("检查 output_v3 目录...", "DEBUG")

        if not self.output_dir.exists():
            self.log(f"output_v3 不存在: {self.output_dir}", "ERROR")
            self.log("请先运行推理脚本：uv run openpi/src/openpi/excavator/test_infer_v3.py", "WARNING")
            return False

        required_files = [
            "v3_stats.json",
            "v3_predictions.csv",
            "v3_1_time_series.png",
            "v3_2_error_analysis.png"
        ]

        for file in required_files:
            if not (self.output_dir / file).exists():
                self.log(f"缺少必需文件: {file}", "WARNING")

        self.log("✅ output_v3 存在并包含必需的输出文件", "SUCCESS")
        return True

    def analyze_performance(self):
        """分析推理性能"""
        self.log("分析推理性能...", "DEBUG")

        stats_file = self.output_dir / "v3_stats.json"
        with open(stats_file) as f:
            stats = json.load(f)

        mae = np.array(stats['MAE'])
        rmse = np.array(stats['RMSE'])

        self.report['findings']['performance'] = {
            'total_samples': stats['总样本数'],
            'valid_samples': stats['有效样本数'],
            'invalid_samples': stats['无效样本数'],
            'success_rate': stats['有效样本数'] / stats['总样本数'] * 100,
            'mae': mae.tolist(),
            'rmse': rmse.tolist(),
            'mae_mean': float(np.mean(mae)),
            'rmse_mean': float(np.mean(rmse))
        }

        self.log(f"成功率: {stats['有效样本数']}/{stats['总样本数']} ({stats['有效样本数']/stats['总样本数']*100:.1f}%)", "INFO")
        self.log(f"平均 MAE: {np.mean(mae):.6f} rad", "INFO")
        self.log(f"平均 RMSE: {np.mean(rmse):.6f} rad", "INFO")

        # 评级
        if np.mean(mae) < 0.05:
            rating = "优秀"
            self.report['severity'] = "green"
        elif np.mean(mae) < 0.1:
            rating = "很好"
            self.report['severity'] = "yellow"
        elif np.mean(mae) < 0.15:
            rating = "良好"
            self.report['severity'] = "yellow"
        elif np.mean(mae) < 0.2:
            rating = "可接受"
            self.report['severity'] = "orange"
        else:
            rating = "差"
            self.report['severity'] = "red"

        self.log(f"性能评级: {rating}", "INFO" if np.mean(mae) < 0.15 else "WARNING")

    def analyze_data_distribution(self):
        """分析数据分布"""
        self.log("分析数据分布...", "DEBUG")

        try:
            train_path = Path("/root/gpufree-data/lerobot_examples_490_train/data/chunk-000/episode_000000.parquet")
            test_path = Path("/root/gpufree-data/lerobot_examples_490_test/data/chunk-000/episode_000000.parquet")

            if not train_path.exists() or not test_path.exists():
                self.log("无法访问数据集文件", "WARNING")
                return

            df_train = pd.read_parquet(train_path)
            df_test = pd.read_parquet(test_path)

            states_train = np.array([np.array(s) for s in df_train['state'].tolist()])
            states_test = np.array([np.array(s) for s in df_test['state'].tolist()])

            # 检查范围匹配
            train_min, train_max = states_train.min(axis=0), states_train.max(axis=0)
            test_min, test_max = states_test.min(axis=0), states_test.max(axis=0)

            out_of_range = np.sum((states_test < train_min) | (states_test > train_max))
            out_of_range_pct = out_of_range / states_test.size * 100

            self.report['findings']['data_distribution'] = {
                'train_shape': states_train.shape,
                'test_shape': states_test.shape,
                'out_of_range_samples': int(out_of_range),
                'out_of_range_percentage': float(out_of_range_pct)
            }

            self.log(f"训练数据范围: {train_min} ~ {train_max}", "DEBUG")
            self.log(f"测试数据范围: {test_min} ~ {test_max}", "DEBUG")

            if out_of_range_pct > 5:
                self.log(f"⚠️ {out_of_range_pct:.1f}% 的测试数据超出训练范围", "WARNING")
                self.report['recommendations'].append("数据分布不匹配：扩大训练数据或重新收集数据")
            else:
                self.log(f"✅ 数据分布匹配：{out_of_range_pct:.1f}% 超出范围", "SUCCESS")

        except Exception as e:
            self.log(f"数据分析失败: {e}", "WARNING")

    def analyze_predictions(self):
        """分析预测结果"""
        self.log("分析预测结果...", "DEBUG")

        try:
            df = pd.read_csv(self.output_dir / "v3_predictions.csv")

            # 检查系统性偏差
            bias_detected = False
            for i in range(4):
                true_col = f'true_J{i+1}'
                pred_col = f'pred_J{i+1}'

                if true_col in df.columns and pred_col in df.columns:
                    error = df[true_col] - df[pred_col]
                    bias = np.mean(error.dropna())

                    if abs(bias) > 0.05:
                        self.log(f"⚠️ 关节{i+1}存在系统性偏差: {bias:.6f}", "WARNING")
                        bias_detected = True

            if not bias_detected:
                self.log("✅ 没有发现系统性偏差", "SUCCESS")

            self.report['findings']['prediction_quality'] = {
                'bias_detected': bias_detected
            }

        except Exception as e:
            self.log(f"预测分析失败: {e}", "WARNING")

    def check_training_logs(self):
        """检查训练日志"""
        self.log("检查训练日志...", "DEBUG")

        try:
            checkpoint_dir = Path("/root/gpufree-data/checkpoints/pi05_excavator_finetune/excavator_v1")
            checkpoints = sorted([d for d in checkpoint_dir.iterdir() if d.is_dir()],
                                key=lambda x: int(x.name))

            self.log(f"找到 {len(checkpoints)} 个 checkpoints", "INFO")
            self.log(f"最终 checkpoint: step {checkpoints[-1].name}", "INFO")

            # 检查是否有日志
            log_files = list(checkpoint_dir.glob("*.log")) + \
                       list(checkpoint_dir.parent.glob("*.log"))

            if log_files:
                self.log(f"✅ 找到 {len(log_files)} 个训练日志文件", "SUCCESS")
                self.report['findings']['training'] = {
                    'log_files_count': len(log_files),
                    'status': 'logs_available'
                }
            else:
                self.log("⚠️ 未找到训练日志文件", "WARNING")
                self.report['recommendations'].append("添加训练日志以便后续诊断：修改训练脚本保存 loss 曲线")
                self.report['findings']['training'] = {
                    'log_files_count': 0,
                    'status': 'no_logs'
                }

        except Exception as e:
            self.log(f"日志检查失败: {e}", "WARNING")

    def generate_diagnosis(self):
        """生成诊断结论"""
        self.log("\n" + "="*70, "INFO")
        self.log("诊断结论", "INFO")
        self.log("="*70, "INFO")

        mae = self.report['findings']['performance']['mae_mean']

        if mae < 0.05:
            diagnosis = "✅ 模型表现优秀，推理效果很好"
            root_cause = "无严重问题"
        elif mae < 0.1:
            diagnosis = "✅ 模型表现良好，推理基本正常"
            root_cause = "可能存在微小的数据或配置问题"
        elif mae < 0.15:
            diagnosis = "⚠️ 模型表现一般，存在可改进空间"
            if 'data_distribution' in self.report['findings']:
                if self.report['findings']['data_distribution']['out_of_range_percentage'] > 5:
                    root_cause = "主要原因：数据分布不匹配"
                else:
                    root_cause = "主要原因：微调模型可能不够好或配置不优"
            else:
                root_cause = "原因不确定，需要进一步检查"
        else:
            diagnosis = "❌ 模型表现差，存在严重问题"
            if 'data_distribution' in self.report['findings']:
                if self.report['findings']['data_distribution']['out_of_range_percentage'] > 20:
                    root_cause = "根本原因：数据严重不匹配"
                else:
                    root_cause = "根本原因：微调问题（模型未收敛或配置错误）"
            else:
                root_cause = "根本原因不确定，需要检查训练日志"

        self.log(diagnosis, "INFO")
        self.log(f"根本原因: {root_cause}", "INFO")

        self.log("\n建议行动:", "INFO")
        for i, rec in enumerate(self.report['recommendations'], 1):
            self.log(f"  {i}. {rec}", "INFO")

        return root_cause

    def save_report(self):
        """保存诊断报告"""
        report_file = self.log_dir / f"diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(report_file, 'w') as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)

        self.log(f"诊断报告已保存: {report_file}", "SUCCESS")

    def run(self):
        """执行完整诊断"""
        self.log("="*70, "INFO")
        self.log("挖掘机推理系统 - 快速诊断", "INFO")
        self.log("="*70, "INFO")

        # Step 1: 检查输出
        if not self.check_output_exists():
            self.log("\n无法进行诊断，请先运行推理脚本", "ERROR")
            return

        self.log("", "INFO")

        # Step 2: 分析性能
        self.analyze_performance()
        self.log("", "INFO")

        # Step 3: 分析数据分布
        self.analyze_data_distribution()
        self.log("", "INFO")

        # Step 4: 分析预测质量
        self.analyze_predictions()
        self.log("", "INFO")

        # Step 5: 检查训练日志
        self.check_training_logs()
        self.log("", "INFO")

        # Step 6: 生成诊断
        root_cause = self.generate_diagnosis()

        # Step 7: 保存报告
        self.log("", "INFO")
        self.save_report()

        self.log("\n" + "="*70, "INFO")
        self.log("诊断完成", "SUCCESS")
        self.log("="*70, "INFO")
        self.log("\n下一步：", "INFO")
        self.log("  • 查看完整的诊断指南：doc/v3/[8] 完整诊断指南.md", "INFO")
        self.log("  • 根据诊断结果执行相应的修复步骤", "INFO")


if __name__ == "__main__":
    try:
        diagnosis = QuickDiagnosis()
        diagnosis.run()
    except KeyboardInterrupt:
        print("\n诊断被中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n诊断过程出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
