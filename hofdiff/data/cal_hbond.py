# Description: Used to calculate hydrogen bond information in cif files
import subprocess
import os

def process_cif_files(directory):
    
    # 遍历目录中的所有文件
    print(f"遍历目录: {directory}")
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            filepath = os.path.join(directory, dir_name, dir_name + ".cif")
            print(f"Processing {filepath} with platon...")
            process = subprocess.Popen(['../../../PLATON/platon', filepath], stdin=subprocess.PIPE, text=True)
            process.stdin.write("CALC HBONDS\n")
            process.stdin.write("exit\n")
            process.stdin.close()
            process.wait()
            print(f"Finished processing {filepath}.")

process_cif_files("hof_data")