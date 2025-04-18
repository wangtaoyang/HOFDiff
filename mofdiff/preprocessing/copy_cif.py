import os
import re
import shutil
import csv

def process_folders(source_dir, target_dir, csv_file):
    # 正则表达式匹配符合 "molecule_{i}.cif" 格式的文件
    pattern = re.compile(r"^molecule_\d+\.cif$")

    # 打开 CSV 文件以追加数据
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # 遍历 source_dir 下的所有子目录
        print(len(os.listdir(source_dir)))
        for cif_id in os.listdir(source_dir):
            subdir = os.path.join(source_dir, cif_id)
            if os.path.isdir(subdir):
                # 统计不符合格式的 cif 文件数量
                non_conforming_files = [f for f in os.listdir(subdir)
                                        if f.endswith('.cif') and not pattern.match(f)]

                # 构建源文件和目标文件的完整路径
                src_file = os.path.join(target_dir, cif_id + '.cif')
                dst_file = os.path.join(subdir, cif_id + '.cif')
                if os.path.exists(dst_file):
                    print(dst_file)
                    writer.writerow([cif_id])
                # 复制文件
                if os.path.exists(src_file):
                    # shutil.copy(src_file, dst_file)
                    print(f"Copied {src_file} to {dst_file}")
                        # 在 CSV 文件中记录 cif_id
                    # else:
                        # print(f"Source file does not exist: {src_file}")

# 脚本使用示例
source_directory = '/opt/wty/hof_synthesis/HOF/MOFDiff/mofdiff/data/hof_data'  # 子文件夹所在的父目录
target_directory = '/opt/wty/hof_synthesis/HOF/MOFDiff/tobacco_cif'  # cif_id.cif 文件所在的目录
csv_filepath = '/opt/wty/hof_synthesis/HOF/MOFDiff/mofdiff/data/hof_data/hof.csv'  # CSV 文件路径
process_folders(source_directory, target_directory, csv_filepath)
