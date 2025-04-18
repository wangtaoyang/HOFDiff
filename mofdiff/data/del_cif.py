import os
import re
import shutil

def delete_non_matching_files(base_folder):
    # 定义符合条件的文件名正则表达式
    # pattern = re.compile(r"^molecule_\d+\.cif$")
    
    # 遍历 base_folder 下的所有一级子文件夹
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        
        # 确保它是一个文件夹
        if os.path.isdir(folder_path):
            # 遍历文件夹中的所有文件
            file_path = os.path.join(folder_path, folder_name + '.cif')
            os.remove(file_path)
            print(f"Deleted file: {file_path}")

# 使用示例
base_folder = '/data/user2/wty/HOF/MOFDiff/mofdiff/data/hof_data'
delete_non_matching_files(base_folder)
