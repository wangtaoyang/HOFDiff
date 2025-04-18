# Description: 用于计算cif文件中的氢键信息
import subprocess
import os

def process_cif_files(directory):
    
    # 遍历目录中的所有文件
    print(f"遍历目录: {directory}")
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            filepath = os.path.join(directory, dir_name, dir_name + ".cif")
            print(f"Processing {filepath} with platon...")
            # 调用platon命令
            process = subprocess.Popen(['../../../PLATON/platon', filepath], stdin=subprocess.PIPE, text=True)
            # 向platon发送命令
            process.stdin.write("CALC HBONDS\n")
            process.stdin.write("exit\n")
            process.stdin.close()
            # 等待platon命令执行完成
            process.wait()
            print(f"Finished processing {filepath}.")

# def process_cif_files(directory):
    
#     # 遍历目录中的所有文件
#     print(f"遍历目录: {directory}")
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if not file.endswith(".cif"):
#                 continue
#             filepath = os.path.join(directory, file)
#             print(filepath)
#             print(f"Processing {filepath} with platon...")
#             # 调用platon命令
#             process = subprocess.Popen(['../../../PLATON/platon', filepath], stdin=subprocess.PIPE, text=True)
#             # 向platon发送命令
#             process.stdin.write("CALC HBONDS\n")
#             process.stdin.write("exit\n")
#             process.stdin.close()
#             # 等待platon命令执行完成
#             process.wait()
#             print(f"Finished processing {filepath}.")

# 调用函数处理特定目录中的cif文件
process_cif_files("hof_data")