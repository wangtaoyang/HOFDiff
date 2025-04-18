
import torch
import sys
import os
# all_data_val, all_z_val = torch.load('/opt/wty/hof_synthesis/HOF/MOFDiff/mofdiff/data/lmdb_data/bwdb_val_20_200.pt')
# all_data_train, all_z_train = torch.load('/opt/wty/hof_synthesis/HOF/MOFDiff/mofdiff/data/lmdb_data/bwdb_train_20_200.pt')

# # 合并all_data_val和all_data_train两个list
# all_data = all_data_val + all_data_train
# all_z = torch.cat([all_z_val, all_z_train], dim=0)
# # 保存在bb_emb_space.pt文件中
# torch.save((all_data, all_z), '/opt/wty/hof_synthesis/HOF/MOFDiff/mofdiff/data/lmdb_data/hof_bb_emb_space.pt')
all_data, all_z = torch.load('/opt/wty/hof_synthesis/HOF/MOFDiff/mofdiff/data/lmdb_data/hof_bb_emb_space.pt')
print(all_data[0])
print(all_z.shape)
