#!/bin/bash
cd . || exit


diffusion_model_path="./hofdiff/data/hof_models/hof_models/bwdb_hoff"
bb_cache_path="./hofdiff/data/lmdb_data/bb_emb_space.pt"

# Loop through GPU indexes 0 to 7
for i in {0..7}
do
    CUDA_VISIBLE_DEVICES=$i python hofdiff/scripts/sample.py --model_path ${diffusion_model_path} --bb_cache_path ${bb_cache_path} --seed $i &
done

wait
