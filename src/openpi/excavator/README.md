# 推理
(openpi) (base) root@gpufree-container:~/openpi# uv run scripts/serve_policy.py \
  --env DROID \
  policy:checkpoint \
  --policy.config pi05_droid \
  --policy.dir /root/gpufree-data/models/pi05_base
  # policy.dir这里是模型参数位置


75、306、490_notrain是推理1300步的结果

# 使用挖掘机数据进行微调

uv run scripts/compute_norm_stats.py --config-name pi05_excavator_finetune

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi05_excavator_finetune --exp-name=my_experiment --overwrite
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_excavator_finetune --exp-name=excavator_v1 --overwrite
# 恢复训练，使用参数 --resume
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_excavator_finetune --exp-name=excavator_v1 --resume


# 找出“谁”占用了空间
du -h --max-depth=1 /root | sort -rh

# 把整个 checkpoints 文件夹移动到数据盘里
mv /root/openpi/checkpoints /root/gpufree-data/

# 创建软链接（关键步骤）： 这条命令会在原来的位置创建一个“快捷方式”，指向新位置
ln -s /root/gpufree-data/checkpoints /root/openpi/checkpoints

# 为了保证数据盘的空间足够，需要删除已经保存的 5000， 10000步的权重文件
rm -rf /root/gpufree-data/checkpoints/pi05_excavator_finetune/excavator_v1/5000
rm -rf /root/gpufree-data/checkpoints/pi05_excavator_finetune/excavator_v1/10000
# 删除旧的 15000 存档
rm -rf /root/gpufree-data/checkpoints/pi05_excavator_finetune/excavator_v1/15000


# 推理（使用训练好的参数）
uv run scripts/serve_policy.py \
  --env DROID \
  policy:checkpoint \
  --policy.config pi05_excavator_finetune \
  --policy.dir /root/gpufree-data/checkpoints/pi05_excavator_finetune/excavator_v1/19999


uv run /root/openpi/test_infer_v2.py
