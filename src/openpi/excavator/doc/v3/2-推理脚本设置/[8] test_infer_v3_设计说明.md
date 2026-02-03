# 🎨 test_infer_v3.py 设计说明

## 📖 概述

**目的**：创建一个完整的推理、验证和诊断脚本，用于挖掘机模型的性能评估

**设计理念**：
- 数据验证优先（5 步框架）
- 问题可诊断（完整的输出和可视化）
- 结果可追踪（JSON + CSV + PNG）
- 代码可维护（清晰的模块划分）

**代码规模**：490 行，7 个主要函数

---

## 🏗️ 架构设计

### 总体流程

```
启动脚本
    ↓
[1] 配置字体和路径
    ↓
[2] 检查 Parquet 数据结构
    ↓
[3] 加载数据（5 步验证）
    ↓
[4] 连接到推理服务
    ↓
[5] 批量推理
    ↓
[6] 分析和可视化
    ↓
[7] 保存结果（JSON + CSV + PNG）
```

### 模块划分

```python
# 配置层
configure_fonts()

# 验证层
inspect_parquet_structure()
load_data_with_validation()

# 推理层
create_observation_dict()
infer_batch()

# 分析层
analyze_and_visualize()

# 主程序
main()
```

---

## 📝 7 个主要函数详解

### 函数1：configure_fonts()

**目的**：配置 Matplotlib 中文字体支持

**设计原因**：
```
挖掘机数据包含中文关节名称（大臂、小臂、铲斗、回转）
可视化结果需要正确显示中文标签
```

**实现**：
```python
def configure_fonts():
    """配置 Matplotlib 中文字体"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
```

**关键决策**：
- 使用 SimHei 字体（中文宋体）
- 禁用 Unicode 负号转义（确保正负号正确显示）

---

### 函数2：inspect_parquet_structure()

**目的**：验证 LeRobot Parquet 数据集的结构

**设计原因**：
```
LeRobot 数据格式可能多变
需要在加载前检查是否兼容
```

**实现逻辑**：
```python
def inspect_parquet_structure():
    """
    1. 检查数据文件是否存在
    2. 读取第一行数据
    3. 验证关键字段
    4. 打印数据统计
    """
```

**验证项目**：
- ✅ 文件存在性
- ✅ Parquet 格式有效性
- ✅ 关键字段存在（state, image, action）
- ✅ 数据维度（应该是多维数组）
- ✅ 样本数量

**输出示例**：
```
✅ 数据结构检查完成
  样本数: 1300
  关键字段: ['state', 'image', 'action']
  State 维度: (1300, 4)
  Image 维度: (1300, 256, 256, 3)
```

---

### 函数3：load_data_with_validation()

**目的**：使用 5 步验证框架加载数据

**设计理念 - 5 步数据验证**：

#### 第1步：数据类型检查
```python
# 验证每列的数据类型
assert state.dtype in [np.float32, np.float64]
assert image.dtype in [np.uint8, np.float32]
assert action.dtype in [np.float32, np.float64]
```

**为什么重要**：
- 类型不对会导致计算错误
- 浮点 vs 整数差异很大

#### 第2步：维度验证
```python
# 验证每列的形状
assert state.shape == (1300, 4)      # 4 个关节
assert image.shape == (1300, 256, 256, 3)  # RGB 图像
assert action.shape == (1300, 4)    # 4 个动作维度
```

**为什么重要**：
- 维度错误会导致模型推理失败
- 需要在处理前确认

#### 第3步：值范围检查
```python
# 验证是否有异常值
assert np.isfinite(state).all()  # 无 NaN 或 Inf
assert np.isfinite(action).all()
assert image.min() >= 0 and image.max() <= 255
```

**为什么重要**：
- NaN/Inf 会破坏梯度计算
- 图像值超出范围表示数据损坏

#### 第4步：关节范围验证
```python
# 检查关节角度是否在合理范围
for i in range(4):
    min_val = state[:, i].min()
    max_val = state[:, i].max()
    # 关节角度通常在 [-π, π] 或 [0, 2π]
```

**为什么重要**：
- 异常的关节范围表示数据问题
- 特别是检测角度包装问题的线索

#### 第5步：统计一致性
```python
# 检查数据分布是否有异常
mean_state = state.mean(axis=0)
std_state = state.std(axis=0)
# 确保标准差不为零（表示有变化）
assert std_state.min() > 1e-6
```

**为什么重要**：
- 某个关节完全恒定值表示数据问题
- 标准差为零会导致归一化失败

**完整的验证框架**：
```python
def load_data_with_validation():
    """5 步数据验证"""
    print("第1步：数据类型检查...")
    # 检查 state.dtype, image.dtype, action.dtype

    print("第2步：维度验证...")
    # 检查形状是否正确

    print("第3步：值范围检查...")
    # 检查 NaN/Inf 和数值范围

    print("第4步：关节范围验证...")
    # 检查关节角度的合理性

    print("第5步：统计一致性...")
    # 检查均值、标准差等统计量

    return state, image, action
```

**设计决策**：
- ✅ 每步都有具体的检查项
- ✅ 发现问题立即停止（不继续处理坏数据）
- ✅ 打印详细的验证结果
- ✅ 易于扩展新的验证项

---

### 函数4：create_observation_dict()

**目的**：将数据转换为 WebSocket API 期望的格式

**设计原因**：
```
推理服务期望特定的观察数据格式
需要统一转换接口
```

**实现逻辑**：
```python
def create_observation_dict(state, image, timestep):
    """
    将原始数据转换为模型期望格式

    输入：
        state: (4,) 数组
        image: (256, 256, 3) 图像
        timestep: 当前时间步

    输出：
        observation: 符合 WebSocket API 的字典
    """
```

**数据转换**：
```python
# 状态转换
joint_position = state.tolist()  # 转为 Python list

# 图像转换
image_bytes = image.tobytes()    # 转为字节流

# 时间戳
step_index = timestep

# 组装字典
observation = {
    "step_index": step_index,
    "observation": {
        "state": joint_position,
        "image": {
            "base_0_rgb": image.tolist()  # 编码为 JSON
        }
    }
}
```

**设计决策**：
- 使用标准的嵌套字典结构
- 图像编码为 Base64（便于 JSON 序列化）
- 包含时间戳信息（便于调试时序问题）

---

### 函数5：infer_batch()

**目的**：与 WebSocket 服务通信，获取推理结果

**设计原因**：
```
推理服务在单独的进程/机器上运行
需要通过网络通信获取结果
需要处理网络错误和超时
```

**实现逻辑**：
```python
def infer_batch(observations, uri="ws://localhost:8765"):
    """
    通过 WebSocket 发送数据，获取推理结果

    设计特点：
    1. 自动重连机制
    2. 超时处理
    3. 错误恢复
    4. 详细的日志输出
    """
```

**关键设计**：

#### 连接建立
```python
async def infer():
    async with websockets.connect(uri) as websocket:
        # 发送和接收数据
```

**为什么使用异步**：
- WebSocket 通信涉及网络等待
- 异步允许非阻塞通信

#### 批量发送和接收
```python
for i, obs in enumerate(observations):
    # 1. 序列化观察为 JSON
    message = json.dumps(obs)

    # 2. 发送到服务
    await websocket.send(message)

    # 3. 接收推理结果
    response = await websocket.recv()

    # 4. 解析结果
    result = json.loads(response)
```

**设计决策**：
- ✅ 逐个发送（便于流式处理）
- ✅ 立即接收（减少内存占用）
- ✅ 完整的错误处理

#### 错误处理
```python
try:
    # 推理逻辑
except websockets.exceptions.WebSocketException as e:
    print(f"❌ WebSocket 错误: {e}")
except json.JSONDecodeError as e:
    print(f"❌ JSON 解析错误: {e}")
except Exception as e:
    print(f"❌ 推理失败: {e}")
```

**设计决策**：
- 分类处理不同类型的错误
- 记录详细的错误信息
- 允许部分失败的恢复

---

### 函数6：analyze_and_visualize()

**目的**：分析推理结果并生成可视化

**设计理念**：
```
不只是产生数字
要让数字可以被理解和直观看到
```

**分析维度1：统计分析**
```python
# 对每个关节计算
for i in range(4):
    mae = np.abs(predictions[:, i] - ground_truth[:, i]).mean()
    rmse = np.sqrt(((predictions[:, i] - ground_truth[:, i])**2).mean())

    # 保存到 JSON
    stats['MAE'].append(mae)
    stats['RMSE'].append(rmse)
```

**为什么重要**：
- MAE：平均误差，易理解
- RMSE：考虑大误差的加权

**分析维度2：时序分析**
```python
# 绘制时间序列图
fig, axes = plt.subplots(4, 1, figsize=(12, 10))
for i in range(4):
    axes[i].plot(ground_truth[:, i], label='Ground Truth')
    axes[i].plot(predictions[:, i], label='Prediction')
    axes[i].set_ylabel(JOINT_NAMES[i])
```

**可视化内容**：
- ✅ 真实轨迹
- ✅ 预测轨迹
- ✅ 重叠显示便于对比

**分析维度3：误差分析**
```python
# 误差分布（箱线图）
errors = predictions - ground_truth
plt.boxplot([errors[:, i] for i in range(4)])
```

**可视化内容**：
- ✅ 误差的中位数
- ✅ 误差的分布范围
- ✅ 异常值检测

**分析维度4：误差分布**
```python
# 误差的直方图
for i in range(4):
    plt.hist(np.abs(errors[:, i]), bins=50)
```

**可视化内容**：
- ✅ 误差的频率分布
- ✅ 识别是否有多个峰值（表示多种错误类型）

**设计决策**：
- ✅ 多个维度的分析
- ✅ 相互补充的可视化
- ✅ 易于发现问题的视觉表示

---

### 函数7：main()

**目的**：主程序，协调所有步骤

**设计架构**：
```python
def main():
    # 初始化
    configure_fonts()

    # 验证
    inspect_parquet_structure()

    # 加载数据
    state, image, action = load_data_with_validation()

    # 推理
    predictions = infer_batch(observations)

    # 分析
    stats, fig = analyze_and_visualize(predictions, action)

    # 保存
    save_results(stats, predictions, figs)
```

**设计特点**：
- ✅ 清晰的步骤顺序
- ✅ 每步都有输入和输出
- ✅ 易于添加新步骤
- ✅ 易于跳过某些步骤进行调试

**错误处理**：
```python
try:
    main()
except KeyboardInterrupt:
    print("被用户中断")
except Exception as e:
    print(f"致命错误: {e}")
    traceback.print_exc()
```

---

## 📊 输出设计

### 输出文件1：v3_stats.json

**目的**：统计结果，便于后续处理

**格式**：
```json
{
  "总样本数": 1300,
  "有效样本数": 1300,
  "成功率": "100%",
  "关节": ["大臂 (Boom)", "小臂 (Arm)", "铲斗 (Bucket)", "回转 (Swing)"],
  "MAE": [0.257, 0.313, 1.018, 1.185],
  "RMSE": [0.312, 0.370, 1.203, 2.194]
}
```

**为什么是 JSON**：
- ✅ 易于解析
- ✅ 易于与其他工具集成
- ✅ 易于版本控制

---

### 输出文件2：v3_predictions.csv

**目的**：详细的预测数据，便于诊断

**格式**：
```csv
step_index,true_J1,true_J2,true_J3,true_J4,pred_J1,pred_J2,pred_J3,pred_J4
0,0.123,0.456,0.789,1.234,0.125,0.458,0.790,1.235
1,0.124,0.457,0.790,1.235,0.126,0.459,0.791,1.236
...
```

**为什么是 CSV**：
- ✅ 易于在 Excel 中打开
- ✅ 易于用 Python/pandas 分析
- ✅ 易于与其他诊断工具集成

**包含的列**：
- `step_index`：时间步索引
- `true_J1-J4`：真实值
- `pred_J1-J4`：预测值

**使用场景**：
```python
# 用户可以加载并进行自定义分析
df = pd.read_csv('v3_predictions.csv')
error = df['true_J1'] - df['pred_J1']
print(f"J1 MAE: {error.abs().mean()}")
```

---

### 输出文件3-5：PNG 可视化

#### v3_1_time_series.png
```
4 个子图，显示每个关节的时序
用户可以看到轨迹的整体匹配情况
```

#### v3_2_error_analysis.png
```
4 个关节的误差箱线图
用户可以快速看到误差的分布和异常值
```

#### v3_3_error_distribution.png
```
4 个关节的误差分布直方图
用户可以看到误差的频率分布
```

---

## 🎯 设计决策解释

### 决策1：为什么需要 5 步验证？

```
V1 错误示例：
  加载数据后直接推理
  结果：失败且错误不清楚

V3 改进：
  第1-5步逐步验证
  发现问题立即报告
  用户可以快速定位问题所在
```

**好处**：
- ✅ 数据问题早期发现
- ✅ 问题定位精确
- ✅ 节省调试时间

---

### 决策2：为什么同时保存 JSON、CSV、PNG？

```
JSON：     用于脚本自动化处理
CSV：      用于人工分析和其他工具
PNG：      用于直观理解
```

**不同用途的用户**：
- 数据科学家：看 CSV 和 PNG
- 自动化系统：读 JSON
- 工程师：全部查看

---

### 决策3：为什么使用异步 WebSocket？

```
同步方式：
  发送1 → 等待 → 接收1 → 发送2 → ...
  1300 个样本需要 1300 次往返
  网络延迟会累积

异步方式：
  流式发送接收
  减少阻塞等待
  效率更高
```

---

### 决策4：为什么生成多个诊断可视化？

```
单个图表的问题：
  时序图显示趋势，但看不到误差分布
  箱线图显示离群值，但看不到整体趋势

多个图表的优势：
  ✅ 从不同角度理解问题
  ✅ 互相补充和验证
  ✅ 易于向别人解释问题
```

---

## 🔌 与诊断工具的配合

### test_infer_v3.py 的输出 → 诊断工具的输入

```
v3_stats.json
    ↓
angle_wrapping_diagnosis.py
    ├─ 检查 J4 改进百分比
    └─ 若 > 30% → 存在角度包装问题

v3_predictions.csv
    ↓
quick_diagnosis.py
    ├─ 计算性能评级
    ├─ 检查数据分布
    └─ 生成诊断结论
```

**设计理念**：
- ✅ test_infer_v3.py 生成原始数据
- ✅ 诊断工具深入分析
- ✅ 形成完整的诊断框架

---

## 💡 可扩展性设计

### 如何添加新的诊断指标？

```python
# 在 analyze_and_visualize 中添加
def analyze_and_visualize(predictions, ground_truth):
    # ... 现有分析 ...

    # 新增：循环距离分析（针对角度）
    circular_error = circular_distance(predictions, ground_truth)
    stats['circular_error'] = circular_error.mean()

    # 新增：极端误差分析
    extreme_errors = predictions[np.abs(predictions - ground_truth) > threshold]

    return stats, figs
```

**设计考虑**：
- ✅ 模块化结构便于扩展
- ✅ 输出格式统一便于集成
- ✅ 注释清晰便于理解

### 如何支持新的模型格式？

```python
# 修改 infer_batch 的 URI
def infer_batch(observations, model_name="pi05_excavator"):
    uri = f"ws://localhost:8765/{model_name}"
    # ...
```

**设计考虑**：
- ✅ 参数化配置
- ✅ 易于切换不同模型

---

## 🚀 使用流程

### 快速使用

```bash
# 1. 确保推理服务运行
# 2. 运行脚本
python test_infer_v3.py

# 3. 查看结果
cat output_v3/v3_stats.json
# 或在 Excel 中打开 output_v3/v3_predictions.csv
# 或查看生成的 PNG 图表
```

### 诊断使用

```bash
# 1. 生成推理数据
python test_infer_v3.py

# 2. 运行角度诊断
python angle_wrapping_diagnosis.py
# 查看 J4 改进百分比

# 3. 运行全面诊断
python quick_diagnosis.py
# 查看整体诊断结论
```

---

## 📚 代码注释示例

脚本中的重要函数都有详细注释：

```python
def apply_angle_wrapping(values, dims=[3]):
    """对指定维度应用角度包装

    原理：
        复杂的角度有时候会超过 ±π，导致线性距离夸大
        例如：-π 和 +π 实际上是同一个角度
        通过 atan2(sin(x), cos(x)) 可以映射到 [-π, π]

    参数：
        values: (N, 4) 数组
        dims: 要包装的维度列表，默认 [3] 表示 J4

    返回：
        wrapped: 包装后的数组

    例子：
        >>> angle = 3.5 rad  # 接近 π
        >>> wrapped = apply_angle_wrapping(angle)
        >>> wrapped ≈ -2.78 rad  # 等价角度，但在 [-π, π] 范围内
    """
```

---

## ✅ 总结

**test_infer_v3.py 是一个完整的诊断工具，设计特点**：

1. **验证优先**：5 步数据验证框架确保数据质量
2. **诊断完整**：多维度分析和可视化
3. **输出多格式**：JSON + CSV + PNG 服务不同用途
4. **易于集成**：与其他诊断工具配合
5. **可扩展**：结构清晰便于添加新功能
6. **可维护**：清晰的模块划分和详细注释

**这个脚本不仅仅是推理，更是完整的问题诊断系统！** 🔍✨
