import os
import shutil
from pathlib import Path
from mofchecker import MOFChecker
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")

def check_and_copy_mof(cif_path, output_dir):
    """
    检查 MOF 的有效性，并在有效时将文件复制到目标文件夹。
    参数:
        cif_path (str): MOF 的 CIF 文件路径。
        output_dir (Path): 输出文件夹路径。
    返回:
        tuple: (cif_path, is_valid)，路径和有效性标识。
    """
    try:
        mofchecker = MOFChecker.from_cif(cif_path)
        result = mofchecker.get_mof_descriptors()
        # 根据指定条件判断 MOF 是否有效
        is_valid = not result["has_atomic_overlaps"] and \
                   not result["has_metal"] and \
                   not result["has_overcoordinated_c"] and \
                   not result["has_overcoordinated_n"] and \
                   not result["has_overcoordinated_h"]
                #    not result["has_lone_molecule"] and \
                #    not result["has_undercoordinated_c"] and \
                #    not result["has_undercoordinated_n"]
        if is_valid:
            shutil.copy(cif_path, output_dir / cif_path.name)
            print(f"Copied valid CIF: {cif_path.name}")
        return cif_path, is_valid
    except Exception as e:
        print(f"Error checking MOF {cif_path}: {e}")
        return cif_path, False


def process_mofs(input_dir, output_dir, n_workers=4):
    """
    遍历输入文件夹中的 CIF 文件，检查有效性，并将有效文件复制到目标文件夹。
    参数:
        input_dir (str): 输入文件夹路径，包含 CIF 文件。
        output_dir (str): 输出文件夹路径，用于存储有效的 CIF 文件。
        n_workers (int): 并行处理的进程数。
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cif_files = list(input_dir.glob("*.cif"))
    print(f"Found {len(cif_files)} CIF files to process.")

    # 使用 ProcessPoolExecutor 并行化处理
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # 用 tqdm 包裹文件列表，显示进度条
        list(
            tqdm(
                executor.map(check_and_copy_mof, cif_files, [output_dir] * len(cif_files)),
                total=len(cif_files),
                desc="Processing CIF files"
            )
        )

    print(f"Processing complete. Valid CIF files have been copied to {output_dir}.")


if __name__ == "__main__":
    # 替换为实际路径
    input_directory = "/data/user2/wty/HOF/MOFDiff/mofdiff/data/mof_models/mof_models/bwdb_hoff/temp_hbond"  # CIF 文件的源文件夹
    output_directory = "/data/user2/wty/HOF/MOFDiff/mofdiff/data/mof_models/mof_models/bwdb_hoff/hofchecker_1"  # 用于存储有效 CIF 文件的目标文件夹

    # 调整 n_workers 的值以控制并行进程数
    process_mofs(input_directory, output_directory, n_workers=100)
