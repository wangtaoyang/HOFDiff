from pathlib import Path
from multiprocessing import Pool
from pymatgen.io.cif import CifWriter
from functools import partial
import argparse
import json

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from mofdiff.common.relaxation import lammps_relax
from mofdiff.common.mof_utils import mof_properties
# from mofid.id_constructor import extract_topology

from p_tqdm import p_umap

# 假设 lammps_relax 是您已有的一个函数，用于处理 CIF 文件
def lammps_relax(ciffile, save_dir):
    """
    模拟 lammps_relax 的行为。
    在实际使用时，请替换为您的实际函数实现。
    """
    # 这里假设返回 relaxed 的结构和相关信息
    from pymatgen.core import Structure
    struct = Structure.from_file(ciffile)  # 加载结构
    relax_info = {"natoms": len(struct), "path": ciffile}
    return struct, relax_info

def relax_mof(ciffile, save_dir):
    """
    Relax a single MOF structure.
    """
    name = ciffile.parts[-1].split(".")[0]
    try:
        struct, relax_info = lammps_relax(str(ciffile), str(save_dir))
    except TimeoutError:
        return None

    if struct is not None:
        struct = struct.get_primitive_structure()
        if struct is not None:
            CifWriter(struct).write_file(save_dir / f"{name}.cif")
            relax_info["natoms"] = struct.frac_coords.shape[0]
            relax_info["path"] = str(save_dir / f"{name}.cif")
            return relax_info
    return None

def relax_worker(args):
    """
    Worker function to relax a single CIF file.
    Args:
        args: Tuple containing ciffile (Path object) and save_dir (Path object).
    """
    ciffile, save_dir = args
    return relax_mof(ciffile, save_dir)

def process_cif_files(input_dir, output_dir, max_natoms=20000, ncpu=1):
    """
    Process all CIF files in the input directory, relax them, and save to output directory.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get list of CIF files in the input directory
    cif_files = list((input_dir).glob("*.cif"))

    # Create arguments for each CIF file
    args = [(ciffile, output_dir) for ciffile in cif_files]

    # Process files with multiprocessing if ncpu > 1
    if ncpu > 1:
        with Pool(ncpu) as pool:
            results = pool.map(relax_worker, args)
    else:
        results = [relax_worker(arg) for arg in args]

    # Filter out None results
    results = [res for res in results if res is not None]
    print(f"Processed {len(results)} CIF files successfully.")


if __name__ == "__main__":
    # 设置输入和输出目录路径
    input_dir = "/data/user2/wty/HOF/MOFDiff/mofdiff/data/mof_models/mof_models/bwdb_hoff/temp_hbond"  # 替换为源CIF文件夹路径
    output_dir = "/data/user2/wty/HOF/MOFDiff/mofdiff/data/mof_models/mof_models/bwdb_hoff/test_relax"  # 替换为目标文件夹路径
    ncpu = 4  # 使用的CPU核数（可调整）

    process_cif_files(input_dir, output_dir, ncpu=ncpu)
