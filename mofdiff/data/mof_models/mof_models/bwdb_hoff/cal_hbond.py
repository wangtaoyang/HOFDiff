# # Description: 用于计算cif文件中的氢键信息
# import subprocess
# import os


# # def process_cif_files(directory):
    
# #     # 遍历目录中的所有文件
# #     print(f"遍历目录: {directory}")
# #     for root, dirs, files in os.walk(directory):
# #         for file in files:
# #             if not file.endswith(".cif"):
# #                 continue
# #             filepath = os.path.join(directory, file)
# #             print(filepath)
# #             print(f"Processing {filepath} with platon...")
# #             # 调用platon命令
# #             process = subprocess.Popen(['/data/user2/wty/HOF/PLATON/platon', filepath], stdin=subprocess.PIPE, text=True)
# #             # 向platon发送命令
# #             process.stdin.write("CALC HBONDS\n")
# #             process.stdin.write("exit\n")
# #             process.stdin.close()
# #             # 等待platon命令执行完成
# #             process.wait()
# #             print(f"Finished processing {filepath}.")

# # # 调用函数处理特定目录中的cif文件
# # process_cif_files("total")

# import os
# import subprocess
# from concurrent.futures import ProcessPoolExecutor

# def process_single_cif(filepath):
#     """Process a single .cif file with platon."""
#     print(f"Processing {filepath} with platon...")
#     process = subprocess.Popen(
#         ['/data/user2/wty/HOF/PLATON/platon', filepath], 
#         stdin=subprocess.PIPE, text=True
#     )
#     # Send commands to platon
#     process.stdin.write("CALC HBONDS\n")
#     process.stdin.write("exit\n")
#     process.stdin.close()
#     # Wait for the command to finish
#     process.wait()
#     print(f"Finished processing {filepath}.")

# def process_cif_files_parallel(directory):
#     """Process all .cif files in a directory in parallel."""
#     print(f"遍历目录: {directory}")
#     # Collect all .cif file paths
#     cif_files = []
#     # for root, dirs, files in os.walk(directory):
#     #     for file in files:
#     #         if file.endswith(".cif"):
#     #             filepath = os.path.join(root, file)
#     #             cif_files.append(filepath)
#     for file in os.listdir(directory):
#         if file.endswith(".cif"):
#             filepath = os.path.join(directory, file)
#             cif_files.append(filepath)

#     print(f"Found {len(cif_files)} .cif files.")

#     # Use ProcessPoolExecutor for parallel processing
#     with ProcessPoolExecutor(max_workers=100) as executor:
#         executor.map(process_single_cif, cif_files)

# # 调用函数处理特定目录中的cif文件
# if __name__ == "__main__":
#     process_cif_files_parallel("hofchecker_1")

import os
import json
import re
from pathlib import Path

def remove_empty_lists_from_json(src_json_path, dest_json_path):
    # 读取源 JSON 文件
    with open(src_json_path, 'r') as src_file:
        data = json.load(src_file)
    
    # 删除值为空列表的键值对
    cleaned_data = {k: v for k, v in data.items() if v != []}
    
    # 将修改后的数据写入目标 JSON 文件
    with open(dest_json_path, 'w') as dest_file:
        json.dump(cleaned_data, dest_file, indent=4)
    print(f"已删除空列表的键值对，并保存到 {dest_json_path}")

class HbondExtractor:
    def __init__(self, cifs_path):
        self.cifs_path = cifs_path

    def get_Hbond_lists(self, cif_id):
        donors, hs, acceptors = [], [], []
        lis_path = os.path.join(self.cifs_path, f"{cif_id}.lis")
        # 假如没有lis文件直接返回空list
        if not os.path.exists(lis_path):
            print(f"No LIS file found for CIF ID {cif_id}.")
            return donors, hs, acceptors
        with open(lis_path, 'r') as file:
            content = file.read()
            # 找到"H....Acceptor"到"Translation of ARU-Code to CIF and Equivalent Position Code"之间的数据块
            data_block_match = re.search(r"(Nr Typ Res Donor.*?)(?=\n[A-Z])", content, re.DOTALL | re.MULTILINE)
        if data_block_match:
            data_block = data_block_match.group(0)
            lines = data_block.splitlines()
            for idx, line in enumerate(lines):
                # 假如line中有？则直接跳过
                if "?" in line:
                    continue
                line = re.sub(r'Intra', ' ', line)
                # 把形如“数字*”的字串替换为“数字 ”
                line = re.sub(r'\d\*', '1 ', line)
                # 替换形如 "_[a-z]" 的后缀
                line = re.sub(r'_[a-z*]', ' ', line)
                line = re.sub(r'_[0-9*]', ' ', line)
                line = re.sub(r'_', ' ', line)
                line = re.sub(r'>', ' ', line)
                line = re.sub(r'<', ' ', line)
                columns = line.split()
                if len(columns) > 1 and (columns[0].isdigit() or columns[0].startswith('**')) and columns[1].isdigit():  # 检查每行是否以数字开头
                    # 提取“元素符号+数字”格式
                    donor = re.search(r'[A-Za-z]+\d+[A-Z]*$', columns[2])
                    h = re.search(r'[A-Za-z]+\d+[A-Z]*$', columns[3])
                    acceptor = re.search(r'[A-Za-z]+\d+[A-Z]*$', columns[4])
                    # 将匹配到的结果添加到列表中, 并且donor不以C开头
                    if donor and not donor.group().startswith('C'):
                        donors.append((donor.group(), idx))
                        if h:
                            hs.append((h.group(), idx))
                        if acceptor:
                            acceptors.append((acceptor.group(), idx))
            # 假如三个list 的长度不相等，则输出cif_id并打印错误信息
            if len(donors) != len(acceptors):
                print('donors:', donors)
                print('hs:', hs)
                print('acceptors:', acceptors)
                print(f"Error in {cif_id}: Donor, H, Acceptor lists have different lengths.")
        return donors, hs, acceptors

    def get_atom_indices(self, cif_id, atoms):
        cif_path = os.path.join(self.cifs_path, f"{cif_id}.cif")
        # print("cif_path:", cif_path)
        # print("atoms:", atoms)
        if not os.path.exists(cif_path):
            print(f"No CIF file found for CIF ID {cif_id}.")
            return []
        atom_indices = []
        with open(cif_path, 'r') as file:
            lines = file.readlines()
            atom_block = False
            atom_list_start_index = None
            for idx, line in enumerate(lines):
                if line.strip() == "_atom_site_type_symbol":
                    atom_block = True
                    atom_list_start_index = idx + 1
                elif atom_block and line.strip() == "loop_":
                    break
                elif atom_block:
                    columns = line.split()
                    # print("columns[0]:", columns[0])
                    # print("atoms:", atoms)
                    if len(columns) > 1 and columns[0] in atoms:
                        atom_indices.append(idx - atom_list_start_index)
        return atom_indices
    
    def get_atom_binary_list(self, cif_id, atoms):
        cif_path = os.path.join(self.cifs_path, f"{cif_id}.cif")
        if not os.path.exists(cif_path):
            print(f"No CIF file found for CIF ID {cif_id}.")
            return []
        
        binary_list = []
        with open(cif_path, 'r') as file:
            lines = file.readlines()
            atom_block = False
            for line in lines:
                if line.strip() == "_atom_site_occupancy":
                    atom_block = True
                elif atom_block and line.strip() == "loop_":
                    break
                elif atom_block:
                    columns = line.split()
                    if len(columns) > 1:
                        if columns[0] in atoms:
                            binary_list.append(1)
                        else:
                            binary_list.append(0)
        return binary_list

    def create_json_from_cifs(self, output_json_path):
        hbond_data = {}
        # 遍历文件夹中的所有 .cif 文件
        for filename in os.listdir(self.cifs_path):
            if filename.endswith(".cif"):
                cif_id = os.path.splitext(filename)[0]
                donors, hs, acceptors = self.get_Hbond_lists(cif_id)
                # 将donors, hs, acceptors合并为一个列表，包含元素符号和行号
                all_atoms = list(set(donors + hs + acceptors))
                # print("all_atoms:", all_atoms)
                atom_symbols = [atom[0] for atom in all_atoms]
                atom_indices = self.get_atom_indices(cif_id, atom_symbols)
                hbond_data[cif_id] = atom_indices
        # 将结果写入JSON文件
        with open(output_json_path, 'w') as json_file:
            json.dump(hbond_data, json_file, indent=4)
        print(f"JSON file created at {output_json_path}")

# 使用示例
if __name__ == "__main__":
    cifs_path = "/data/user2/wty/HOF/MOFDiff/mofdiff/data/mof_models/mof_models/bwdb_hoff/hofchecker_1"  # 替换为你的cif文件夹路径
    output_json_path = "/data/user2/wty/HOF/moftransformer/data/HOF_pretrain_new/fold8/20000HOFdiff_hbond_index.json"  # 输出JSON文件的路径
    extractor = HbondExtractor(cifs_path)
    extractor.create_json_from_cifs(output_json_path)
    
    # src_json_path = Path('/data/user2/wty/HOF/moftransformer/data/HOF_pretrain/all_tobacco_hbond.json')  # 替换为你的源 JSON 文件路径
    # dest_json_path = Path('/data/user2/wty/HOF/moftransformer/data/HOF_pretrain/all_tobacco_hbond.json')  # 替换为你的目标 JSON 文件路径
    # remove_empty_lists_from_json(src_json_path, dest_json_path)

    