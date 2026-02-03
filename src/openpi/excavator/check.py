# import json
# import os
# from pathlib import Path

# # --- 配置区域 ---
# # 你的数据集 meta 目录
# META_DIR = Path("/root/gpufree-data/lerobot_examples_490_train/meta")
# TARGET_KEY = "elevation"

# def remove_key_recursive(data, target):
#     """递归删除字典或列表中的指定 key"""
#     has_changed = False
    
#     if isinstance(data, dict):
#         # 1. 如果当前字典里有这个 key，直接删除
#         if target in data:
#             print(f"   ✂️  删除字段: {target}")
#             del data[target]
#             has_changed = True
        
#         # 2. 继续深入检查字典的其他值
#         for key, value in data.items():
#             if remove_key_recursive(value, target):
#                 has_changed = True
                
#     elif isinstance(data, list):
#         # 3. 如果是列表，遍历检查每一项
#         for item in data:
#             if remove_key_recursive(item, target):
#                 has_changed = True
                
#     return has_changed

# def clean_all_json():
#     print(f"📂 正在扫描目录: {META_DIR}")
    
#     if not META_DIR.exists():
#         print("❌ 错误: meta 目录不存在！请检查路径。")
#         return

#     # 扫描所有 .json 文件 (通常主要是 info.json)
#     json_files = list(META_DIR.glob("*.json"))
    
#     if not json_files:
#         print("⚠️  未找到 .json 文件。")
#         return

#     for json_file in json_files:
#         print(f"\n📄 处理文件: {json_file.name}")
        
#         try:
#             with open(json_file, "r") as f:
#                 data = json.load(f)
            
#             # 执行递归删除
#             if remove_key_recursive(data, TARGET_KEY):
#                 # 只有发生变化时才写回文件
#                 with open(json_file, "w") as f:
#                     json.dump(data, f, indent=4)
#                 print(f"✅ 已保存修改: {json_file.name}")
#             else:
#                 print("   (无须修改)")
                
#         except Exception as e:
#             print(f"❌ 处理出错: {e}")

#     print("\n🎉 清理完成！现在代码不会再找 elevation 摄像头了。")

# if __name__ == "__main__":
#     clean_all_json()




import os
import json
import shutil
import pandas as pd
from pathlib import Path

# --- 配置 ---
DATASET_DIR = Path("/root/gpufree-data/lerobot_examples_490_train")
DATA_DIR = DATASET_DIR / "data"
META_DIR = DATASET_DIR / "meta"
INFO_PATH = META_DIR / "info.json"

def fix_parquet_files():
    print(f"📂 正在扫描 Parquet 文件: {DATA_DIR}")
    files = list(DATA_DIR.rglob("*.parquet"))
    
    if not files:
        print("❌ 未找到 Parquet 文件！")
        return

    print(f"🔄 正在修改 {len(files)} 个文件的列名 (action -> actions, state -> states)...")
    
    for fpath in files:
        try:
            # 读取
            df = pd.read_parquet(fpath)
            
            # 检查并重命名列
            renamed = False
            if "action" in df.columns:
                df.rename(columns={"actions": "action"}, inplace=True)
                renamed = True
            if "state" in df.columns:
                df.rename(columns={"states": "state"}, inplace=True)
                renamed = True
                
            # 如果有改动，保存回去
            if renamed:
                df.to_parquet(fpath)
                print(f"  ✅ 已修正: {fpath.name}")
            else:
                print(f"  ⚠️ 跳过 (已是复数?): {fpath.name}")
                
        except Exception as e:
            print(f"  ❌ 处理失败 {fpath.name}: {e}")

def fix_info_json():
    print(f"\n🔧 正在修正 info.json: {INFO_PATH}")
    if not INFO_PATH.exists(): return

    with open(INFO_PATH, "r") as f:
        data = json.load(f)
    
    features = data.get("features", {})
    changed = False

    # 1. 修改 action -> actions
    if "action" in features:
        features["actions"] = features.pop("action")
        # 还要修改内部的 names 列表
        if "names" in features["actions"]:
             features["actions"]["names"] = ["actions"] # 或者保留原样，但这通常只是标签
        changed = True
        print("  ✅ Key 'action' -> 'actions'")

    # 2. 修改 state -> states
    if "state" in features:
        features["states"] = features.pop("state")
        changed = True
        print("  ✅ Key 'state' -> 'states'")

    if changed:
        with open(INFO_PATH, "w") as f:
            json.dump(data, f, indent=4)
        print("💾 info.json 保存完成！")
    else:
        print("  (无需修改)")

def clean_stats():
    print("\n🧹 清理旧统计缓存...")
    for f in ["episodes_stats.jsonl", "stats.json"]:
        p = META_DIR / f
        if p.exists():
            p.unlink()
            print(f"  🗑️ 删除: {f}")

if __name__ == "__main__":
    # 1. 改 Parquet 文件里的列名
    fix_parquet_files()
    # 2. 改 info.json 里的定义
    fix_info_json()
    # 3. 删缓存
    clean_stats()
    print("\n🎉 数据集格式转换完成！现在它是 OpenPI 喜欢的样子了。")