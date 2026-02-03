# 数据集格式说明（meta/info.json）

## 文件位置

```
gpufree-data/lerobot_examples_490_test/meta/info.json
```

## 核心内容解读

### 1. 元数据信息

```json
{
  "codebase_version": "v2.1",
  "robot_type": "excavator",
  "total_episodes": 1,
  "total_frames": 3233,
  "total_tasks": 1,
  "total_videos": 2,
  "total_chunks": 1,
  "chunks_size": 1000,
  "fps": 10
}
```

| 字段 | 值 | 说明 |
|------|-----|------|
| `total_frames` | 3233 | 视频总帧数（@10 FPS） |
| `fps` | 10 | 视频帧率 |
| `total_episodes` | 1 | 只有1个演示序列 |
| `total_chunks` | 1 | 数据分为1个chunk |
| `chunks_size` | 1000 | 每个chunk包含最多1000行 |

**关键启示**：
- 数据集来自单个 10 FPS 的视频
- 3233 帧对应约 5.4 分钟的视频
- Parquet 样本数（~1300）远小于帧数，意味着不是每帧都被采样

### 2. 文件路径模板

```json
{
  "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
  "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
}
```

展开后：
- **Parquet**: `data/chunk-000/episode_000000.parquet`
- **视频**: `videos/chunk-000/{video_key}/episode_000000.mp4`

其中 `{video_key}` 可能是 `main`、`front` 等。

### 3. 数据特征定义

#### 3.1 视频特征（"main"）

```json
{
  "main": {
    "dtype": "video",
    "shape": [480, 640, 3],
    "names": ["height", "width", "channel"],
    "info": {
      "video.height": 480,
      "video.width": 640,
      "video.codec": "av1",
      "video.pix_fmt": "yuv420p",
      "video.is_depth_map": false,
      "video.fps": 10,
      "video.channels": 3,
      "has_audio": false
    }
  }
}
```

**说明**：
- 字段名称：`"main"`（这是关键！）
- 原始分辨率：480×640（高×宽）
- 格式：BGR 24-bit（来自 ffmpeg）
- 编码：AV1（现代视频编码）

#### 3.2 状态特征

```json
{
  "state": {
    "dtype": "float32",
    "shape": [4],
    "names": ["state"]
  }
}
```

**说明**：
- 4 维浮点数向量
- 对应挖掘机的 4 个关节：[boom, arm, bucket, swing]

#### 3.3 动作特征

```json
{
  "action": {
    "dtype": "float32",
    "shape": [4],
    "names": ["action"]
  }
}
```

**说明**：
- 同样是 4 维浮点向量
- 表示每个关节的目标动作

#### 3.4 时间戳特征

```json
{
  "timestamp": {
    "dtype": "float32",
    "shape": [1],
    "names": null
  }
}
```

#### 3.5 **帧索引特征（关键！）**

```json
{
  "frame_index": {
    "dtype": "int64",
    "shape": [1],
    "names": null
  }
}
```

**这是最重要的字段**：
- 每一行 state/action 对应的视频帧索引
- 类型：64-bit 整数
- 值范围：[0, 3233)
- 用途：将 parquet 行与视频帧准确对应

#### 3.6 其他元数据字段

```json
{
  "episode_index": {
    "dtype": "int64",
    "shape": [1]
  },
  "index": {
    "dtype": "int64",
    "shape": [1]
  },
  "task_index": {
    "dtype": "int64",
    "shape": [1]
  }
}
```

---

## modality.json 说明

```json
{
  "state": {
    "excavator_arm": {
      "start": 0,
      "end": 4,
      "original_key": "state"
    }
  },
  "action": {
    "excavator_arm": {
      "start": 0,
      "end": 4
    }
  },
  "video": {
    "camera_front": {
      "original_key": "main"
    }
  },
  "annotation": {
    "human.task_description": {
      "original_key": "task_index"
    }
  }
}
```

**说明**：
- **state**: 全部 4 维都用于 "excavator_arm"
- **action**: 全部 4 维都用于 "excavator_arm"
- **video**: "main" 字段映射为 "camera_front"
- **task**: task_index 包含任务描述

**关键发现**：
- 确认了视频字段来自 "main"
- 确认了 state/action 都是 4 维的

---

## 数据集验证脚本

```python
import json
import pandas as pd
from pathlib import Path

data_root = Path("/root/gpufree-data/lerobot_examples_490_test")

# 1. 读取 meta
with open(data_root / "meta/info.json") as f:
    info = json.load(f)

print("=== Meta Info ===")
print(f"Total frames: {info['total_frames']}")
print(f"FPS: {info['fps']}")
print(f"Duration: {info['total_frames'] / info['fps']:.1f}s")

# 2. 读取 modality
with open(data_root / "meta/modality.json") as f:
    modality = json.load(f)

print("\n=== Modality ===")
print(f"State field: {modality['state']}")
print(f"Video field: {modality['video']}")

# 3. 读取 parquet
df = pd.read_parquet(data_root / "data/chunk-000/episode_000000.parquet")

print(f"\n=== Parquet ===")
print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"Frame index range: [{df['frame_index'].min()}, {df['frame_index'].max()}]")
print(f"Frame index max < total_frames: {df['frame_index'].max() < info['total_frames']}")

# 4. 检查视频
video_path = data_root / f"videos/chunk-000/{modality['video']['camera_front']['original_key']}/episode_000000.mp4"
print(f"\n=== Video ===")
print(f"Video exists: {video_path.exists()}")
print(f"Video path: {video_path}")

# 5. 样本检查
print(f"\n=== Sample ===")
sample_idx = 0
print(f"State shape: {df['state'].iloc[sample_idx].shape}")
print(f"Action shape: {df['action'].iloc[sample_idx].shape}")
print(f"Frame index for sample {sample_idx}: {df['frame_index'].iloc[sample_idx]}")
```

**预期输出**：
```
=== Meta Info ===
Total frames: 3233
FPS: 10
Duration: 323.3s

=== Modality ===
State field: {'excavator_arm': {'start': 0, 'end': 4, 'original_key': 'state'}}
Video field: {'camera_front': {'original_key': 'main'}}

=== Parquet ===
Total rows: 1300
Columns: ['state', 'action', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index']
Frame index range: [0, 3231]
Frame index max < total_frames: True

=== Video ===
Video exists: True
Video path: .../videos/chunk-000/main/episode_000000.mp4

=== Sample ===
State shape: (4,)
Action shape: (4,)
Frame index for sample 0: 0
```

---

## 推理时的应用

### 1. 加载数据时

```python
import pandas as pd
import json

# 从 meta 获取视频字段名
with open("meta/info.json") as f:
    meta = json.load(f)

# 从 meta 知道了 "main" 是视频字段
video_field_name = "main"  # 从 modality 或 info 中确定

# 从 parquet 加载
df = pd.read_parquet("data/chunk-000/episode_000000.parquet")
frame_indices = df['frame_index'].values
```

### 2. 推理循环中

```python
for i in range(len(states)):
    # 使用 frame_index 获取对应的视频帧
    frame_idx = int(frame_indices[i])
    image = video_frames[frame_idx]  # 索引到第 frame_idx 帧

    obs = {
        "state": states[i],
        "image": image
    }
    result = policy.infer(obs)
```

---

## 常见问题

### Q1：为什么需要 frame_index？

**A**：因为 parquet 中不是每一帧都被采样：
- 视频有 3233 帧
- parquet 只有 1300 行
- frame_index 告诉我们每一行对应哪一帧

没有 frame_index 的话，无法建立对应关系。

### Q2：frame_index 的值范围是多少？

**A**：[0, 3232]（因为帧索引是 0-based，3233 帧的索引是 0-3232）

### Q3：为什么视频分辨率是 480×640 而模型输入是 224×224？

**A**：
- 480×640 是原始视频分辨率
- 224×224 是模型的标准输入
- 推理脚本中的 MP4Reader 会自动缩放

### Q4：什么是 "main"？

**A**：在这个数据集中，"main" 是视频摄像机的字段名。另一个类似的命名可能是 "front"、"left_wrist" 等。

---

## 关键要点总结

✅ **数据集有 3233 帧的视频 + 1300 个 state/action 样本**

✅ **frame_index 字段建立了它们之间的映射**

✅ **视频字段名是 "main"（在 modality 中定义）**

✅ **每个样本都有时间戳和帧索引，便于精确对应**

✅ **这个设计允许灵活的采样策略（不必是每帧都采样）**
