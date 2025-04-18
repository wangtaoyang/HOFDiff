#!/bin/bash

# 设置初始随机种子和递增步长
seed=10
increment=8

# 设置最大循环次数（根据需求调整）
max_iterations=100

# 模型路径和其他参数
model_path="/data/user2/wty/HOF/MOFDiff/mofdiff/data/mof_models/mof_models/bwdb_hoff"
bb_cache_path="/data/user2/wty/HOF/MOFDiff/mofdiff/data/lmdb_data/bb_emb_space.pt"

# 输出路径模板
sample_path_template="${model_path}/samples_4096_seed_%s/samples.pt"

# 循环生成并组装 HOF 结构
for ((i=1; i<=max_iterations; i++)); do
    echo "当前种子: $seed"

    # 运行 sample.py 脚本
    CUDA_VISIBLE_DEVICES=2 python mofdiff/scripts/sample.py \
        --model_path "$model_path" \
        --bb_cache_path "$bb_cache_path" \
        --seed $seed

    # 生成的 sample.pt 路径
    sample_path=$(printf "$sample_path_template" "$seed")
    
    # 检查 sample.pt 是否生成成功
    if [[ -f "$sample_path" ]]; then
        echo "Sample 文件生成成功: $sample_path"
        
        # 运行 assemble.py 脚本
        python mofdiff/scripts/assemble.py --input "$sample_path"
        
        # 检查是否生成了 cif 文件
        cif_dir=$(dirname "$sample_path")/cif
        if ls "$cif_dir"/*.cif 1> /dev/null 2>&1; then
            echo "CIF 文件生成成功，路径: $cif_dir"
        else
            echo "CIF 文件生成失败，检查 assemble.py 的逻辑。"
        fi
    else
        echo "Sample 文件生成失败，跳过种子 $seed。"
    fi

    # 更新种子
    seed=$((seed + increment))
done
