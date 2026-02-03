import os
import h5py
import shutil
import argparse
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 注意：如果你的数据集要上传到 HuggingFace，记得修改这里的用户名
REPO_NAME = "your_hf_username/excavator-motion"

def run(data_dir: str, out_dir: str):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    # 创建 LeRobot 数据集结构
    # 这里的 shape 和 dtype 必须和下面读取到的数据严格一致
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        fps=10,
        root=out_dir,
        robot_type="excavator",
        features={
            "main": {
                "dtype": "video",
                "shape": (384, 480, 3), # 修改：根据之前的报错，调整为实际尺寸 (384, 480)
                "names": ["height", "width", "channel"],
            },
            "elevation": {
                "dtype": "video",
                "shape": (200, 200, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",     # 严格要求 float32
                "shape": (4,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",     # 严格要求 float32
                "shape": (4,),
                "names": ["action"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # 遍历目录下的所有 h5 文件
    # 使用列表推导式并过滤掉非 .h5 文件，防止读取系统隐藏文件报错
    data_list = [os.path.join(data_dir, name) for name in os.listdir(data_dir) if name.endswith('.h5')]
    
    print(f"Found {len(data_list)} h5 files to process.")

    for data_path in data_list:
        try:
            with h5py.File(data_path, 'r') as f:
                # 1. 读取图像数据 (图像通常不需要手动转 dtype，LeRobot 会处理)
                mains = np.array(f['observations']['images']['main'])
                elevations = np.array(f['observations']['images']['elevation'])
                
                # 2. 读取 State 并强制转换为 float32
                # HDF5 默认读出来是 float64，会导致 LeRobot 报错，必须转！
                states = np.array(f['observations']['qpos']).astype(np.float32)
                
                # 3. 读取 Action 并强制转换为 float32 (包含兼容性逻辑)
                if 'action' in f:
                    actions = np.array(f['action']).astype(np.float32)
                elif 'actions' in f:
                    actions = np.array(f['actions']).astype(np.float32)
                else:
                    # 如果找不到 action，使用 qpos 代替（位置控制模式）
                    # print(f"注意: {data_path} 使用 qpos 代替 action")
                    actions = np.array(f['observations']['qpos']).astype(np.float32)

                n_frames = actions.shape[0]
                
                # 4. 将数据写入数据集
                for i in range(n_frames):
                    dataset.add_frame(frame={
                            "main": mains[i],
                            "elevation": elevations[i],
                            "state": states[i],
                            "action": actions[i]
                        },
                        task="", # 留空或填入具体的任务描述
                    )
                
                dataset.save_episode()
                
        except Exception as e:
            print(f"Error processing file {data_path}: {e}")
            # 可以选择 continue 跳过错误文件，或者直接崩溃
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help='path of data to be converted')
    parser.add_argument("--out_dir", type=str, required=True, help='path for saving conversion results')
    args = parser.parse_args()

    run(args.data_dir, args.out_dir)