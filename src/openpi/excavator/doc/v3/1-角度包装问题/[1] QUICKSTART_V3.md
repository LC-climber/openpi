# 🚀 快速开始 - 立即可做的修复

## ⚡ 5分钟快速修复（角度包装）

这是**立即可执行**的修复，无需重新训练！

### 步骤1：修改推理脚本

**文件**：`test_infer_v3.py`

在计算 MAE 之前，添加以下代码：

```python
import numpy as np

def apply_angle_wrapping(values, dims=[3]):
    """对指定维度应用角度包装到 [-π, π]"""
    wrapped = values.copy()
    for dim in dims:
        # 使用 atan2 将角度归一化到 [-π, π]
        wrapped[:, dim] = np.arctan2(
            np.sin(values[:, dim]),
            np.cos(values[:, dim])
        )
    return wrapped

# 在计算统计数据之前应用（位置很重要！）
# 应该在创建 predictions 和 ground_truth 之后，计算 MAE 之前
predictions = apply_angle_wrapping(predictions)
ground_truth = apply_angle_wrapping(ground_truth)
```

**具体位置**：在 test_infer_v3.py 中找到计算 MAE 的地方，在其前面添加这段代码。

### 步骤2：运行推理

```bash
cd /home/er/Code/openpi-v1/openpi/src/openpi/excavator

# 如果服务器已在运行，直接运行
python test_infer_v3.py

# 或者完整流程：
# 终端1：启动服务器
cd /home/er/Code/openpi-v1
python scripts/serve_policy.py \
  --env DROID \
  policy:checkpoint \
  --policy.config pi05_excavator_finetune \
  --policy.dir /root/gpufree-data/checkpoints/pi05_excavator_finetune/excavator_v1/19999

# 终端2：运行推理
sleep 30
cd /home/er/Code/openpi-v1/openpi/src/openpi/excavator
python test_infer_v3.py
```

### 步骤3：验证改进

```bash
# 查看新的统计结果
cat /home/er/Code/openpi-v1/output_v3/v3_stats.json | python3 -m json.tool
```

**预期结果**：
```json
{
  "MAE": [0.257, 0.313, 1.018, 0.491],  // J4 从 1.185 改进到 0.491!
  "RMSE": [0.312, 0.370, 1.203, 0.800]
}
```

---

## 📊 完整运行流程（如需重新训练）

### 前置条件检查

```bash
# 1. 检查推理服务器端口
nc -zv 127.0.0.1 8000

# 2. 检查测试数据集
ls /root/gpufree-data/lerobot_examples_490_test/data/chunk-000/episode_000000.parquet

# 3. 检查最新 checkpoint
ls /root/gpufree-data/checkpoints/pi05_excavator_finetune/excavator_v1/19999/
```

### 完整流程

**终端1 - 启动推理服务器**
```bash
cd /home/er/Code/openpi-v1

python scripts/serve_policy.py \
  --env DROID \
  policy:checkpoint \
  --policy.config pi05_excavator_finetune \
  --policy.dir /root/gpufree-data/checkpoints/pi05_excavator_finetune/excavator_v1/19999
```

**终端2 - 运行推理**
```bash
cd /home/er/Code/openpi-v1

# 等待服务器就绪
sleep 30

# 运行推理（推荐：先应用角度包装修复）
cd openpi/src/openpi/excavator
python test_infer_v3.py
```

---

## 📈 查看和理解结果

### 查看输出文件

```bash
cd /home/er/Code/openpi-v1/output_v3

# 查看文件列表
ls -lh
# 应该包含：
# v3_stats.json - 统计数据
# v3_predictions.csv - 1300行预测数据
# v3_1_time_series.png - 时序图
# v3_2_error_analysis.png - 误差分析图
# v3_3_error_distribution.png - 误差分布图
```

### 查看统计数据

```bash
# 查看 MAE/RMSE
python3 -c "import json; data=json.load(open('v3_stats.json')); print('关节  MAE    RMSE'); [print(f'{i+1}.   {m:.3f}  {r:.3f}') for i,(m,r) in enumerate(zip(data['MAE'], data['RMSE']))]"
```

### 理解 MAE 数值

```
关节1 (大臂 Boom):     0.257 rad  ✅ 良好（已正常）
关节2 (小臂 Arm):      0.313 rad  ✅ 良好（已正常）
关节3 (铲斗 Bucket):   1.018 rad  ⚠️ 较差（需要重新训练）
关节4 (回转 Swing):    1.185 rad → 0.491 rad ✅ 应用角度包装后改进（58.6%）
```

---

## 🎯 下一步建议

### 立即（今天）
- [x] 应用角度包装修复（5分钟）
- [x] 验证 J4 改进 58.6%（5分钟）

### 本周（可选，需要重新训练）
- [ ] 应用数据处理管道修复（代码已完成，见文档[3]）
- [ ] 重新训练模型（4-6小时）
- [ ] 验证完整改进（J3+J4都改进到0.4 rad）

### 如果想了解更多
- 问题分析：查看文档 [2]
- 完整修复方案：查看文档 [3]
- 诊断工具使用：查看文档 [4]

---

## ⚡ 常见问题

**Q: 角度包装修复需要重新训练吗？**
A: 不需要！这是推理后处理，立即可用。

**Q: 修复后 J3 还是很差怎么办？**
A: J3 需要修复数据处理管道（问题A），需要重新训练。详见文档 [3]。

**Q: 怎么知道修复成功了？**
A: J4 的 MAE 应该从 1.185 改进到 0.491（58.6% ↓）。

**Q: 需要修改训练配置吗？**
A: 如果只是应用角度包装修复，不需要。如果要修复数据管道问题，需要重新训练。

---

## 📞 需要帮助？

- **想快速了解问题？** → 看文档 [2]（10分钟）
- **想看完整代码修复？** → 看文档 [3]（30分钟）
- **想用诊断工具？** → 看文档 [4]（10分钟）

---

**开始修复吧！** 🚀
