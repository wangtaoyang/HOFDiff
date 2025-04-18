#!/bin/bash
cd /data/user2/wty/HOF/MOFDiff || exit

# 设置变量
diffusion_model_path="/data/user2/wty/HOF/MOFDiff/mofdiff/data/mof_models/mof_models/bwdb_hoff"
bb_cache_path="/data/user2/wty/HOF/MOFDiff/mofdiff/data/lmdb_data/bb_emb_space.pt"

# 循环遍历 GPU 索引 0 到 7
for i in {0..7}
do
    CUDA_VISIBLE_DEVICES=$i python mofdiff/scripts/sample.py --model_path ${diffusion_model_path} --bb_cache_path ${bb_cache_path} --seed $i &
done

# 等待所有后台进程完成
wait
