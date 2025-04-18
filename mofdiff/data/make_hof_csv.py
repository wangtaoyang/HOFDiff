import os
import csv

def write_subfolder_names_to_csv(directory, output_csv):
    # 获取所有一级子文件夹的名称
    subfolders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    
    # 将子文件夹名称写入CSV文件的第一列
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        for subfolder in subfolders:
            writer.writerow([subfolder])

if __name__ == "__main__":
    directory = "/opt/wty/hof_synthesis/HOF/MOFDiff/mofdiff/data/hof_data"  # 替换为你的目录路径
    output_csv = "/opt/wty/hof_synthesis/HOF/MOFDiff/mofdiff/data/hof_data/hof.csv"  # 输出的CSV文件名
    write_subfolder_names_to_csv(directory, output_csv)
    print(f"子文件夹名称已写入 {output_csv}")