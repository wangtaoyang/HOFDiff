from pathlib import Path
import argparse
import pickle
import pandas as pd
from p_tqdm import p_umap
from openbabel import openbabel as ob
import numpy as np
import re

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from mofdiff.common.atomic_utils import pyg_graph_from_cif, assemble_local_struct
import multiprocessing as mp
from mofdiff.common.constants import COVALENT_RADII

mp.set_start_method("spawn", force=True)

def has_no_overlapping_atoms(cif_path, threshold=0.7):
    """
    判断给定的 CIF 文件中是否有重叠的原子。如果没有重叠原子则返回 True，否则返回 False。

    :param cif_path: CIF 文件路径
    :param threshold: 判定原子是否重叠的阈值，默认为 0.7
    :return: 没有重叠原子返回 True，有重叠原子返回 False
    """
    print(f"Checking {cif_path} for overlapping atoms.")
    obConversion = ob.OBConversion()
    obConversion.SetInFormat("cif")
    mol = ob.OBMol()

    if not obConversion.ReadFile(mol, cif_path):
        print(f"Failed to read {cif_path} file.")
        return False

    # 分离出所有连通分支
    fragments = mol.Separate()

    for frag in fragments:
        frag_mol = ob.OBMol(frag)
        other_atoms = []

        # 遍历分子中的每个原子，检查原子间是否有重叠
        for atom in ob.OBMolAtomIter(frag_mol):
            pos = np.array([atom.GetX(), atom.GetY(), atom.GetZ()])
            e1 = atom.GetType()
            
            for other_atom in other_atoms:
                other_pos = np.array([other_atom.GetX(), other_atom.GetY(), other_atom.GetZ()])
                e2 = other_atom.GetType()
                
                # 去掉e1e2的数字，只留下字母
                e1 = ''.join([i for i in e1 if not i.isdigit()])
                e2 = ''.join([i for i in e2 if not i.isdigit()])
                # 根据原子类型，计算它们的共价半径
                try:
                    min_threshold = min(COVALENT_RADII[e1], COVALENT_RADII[e2])
                except KeyError as e:
                    # print(f"Warning: Unrecognized atom type '{e.args[0]}' encountered.")
                    continue  # Skip or handle the unrecognized atom type
                if np.linalg.norm(pos - other_pos) < threshold * min_threshold:
                    return False  # 找到重叠的原子，直接返回 False

            other_atoms.append(atom)

    return True  # 没有重叠原子，返回 True

def assign_cif_files(base_path):
    # 正则表达式，用于匹配符合 "molecule_{i}.cif" 格式的文件名
    cif_pattern = re.compile(r"^molecule_\d+\.cif$")
    
    # 用于存储不符合特定格式的cif文件路径
    non_conforming_cifs = []
    hid = base_path.parts[-1]
    # 遍历base_path目录下的所有文件
    for file in os.listdir(base_path):
        if file.endswith('.cif') and not cif_pattern.match(file) and file != f'{hid}.cif':
            bb_path = os.path.join(base_path, file)
            if has_no_overlapping_atoms(bb_path):
                print(f"{bb_path} has no overlapping")
                non_conforming_cifs.append(os.path.join(base_path, file))
    
    return non_conforming_cifs
    # 检查是否正好有三个非特定格式的cif文件
    # if len(non_conforming_cifs) == 3:
    #     g_nodes, g_linkers, g_node_bridges = non_conforming_cifs
    #     return g_nodes, g_linkers, g_node_bridges
    # else:
    #     raise ValueError("There are not exactly three non-conforming .cif files in the directory.")

def preprocess_graph(df_path, save_path, num_workers, device="cpu"):
    df = pd.read_csv(str(df_path))
    Path(save_path).mkdir(exist_ok=True, parents=True)

    def assemble_mof(m_id, use_asr=True):
        try:
            base_path = Path(f'/data/user2/wty/HOF/MOFDiff/mofdiff/data/hof_data/{m_id}')
            # print(base_path)
            # use the metaloxo algorithm for deconstruction.
            # g_nodes_path, g_linkers_path, g_node_bridges_path = assign_cif_files(base_path)
            bb_cifs = assign_cif_files(base_path)
            print("bb_cifs:",bb_cifs)
            g_bb_cifs = []
            for bb_cif in bb_cifs:
                g_bb_cifs.append(pyg_graph_from_cif(Path(bb_cif), Hbond=False))
            # print("g_bb_cifs:", len(g_bb_cifs))
            if use_asr:
                g_asr = pyg_graph_from_cif(Path(base_path / f'{m_id}.cif'), Hbond=True)
            else:
                g_asr = None
            # print("g_node_bridges",g_node_bridges)
            print("g_asr",g_asr)
            data = assemble_local_struct(
                g_bb_cifs, g_asr, device=device, 
            )
        except FileNotFoundError:
            print(f"FileNotFoundError: {m_id}")
            return None
        except UnboundLocalError:
            print(f"UnboundLocalError: {m_id}")
            return None
        except IndexError:
            print(f"IndexError: {m_id}")
            return None
        except ValueError:
            print(f"ValueError: {m_id}")
            return None
        return data

    def save_mof_graph_pkl(m_id):
        props = df.loc[m_id]
        if (Path(save_path) / f"{m_id}.pkl").exists():
            return m_id, False

        failed = True
        data = assemble_mof(m_id)
        if data is not None:
            data.m_id = m_id
            data.prop_dict = dict(props)
            data.top = 'hof_top'
            with open(str(Path(save_path) / f"{m_id}.pkl"), "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            failed = False

        return m_id, failed

    done = [idx.parts[-1][:-4] for idx in Path(save_path).glob("*.pkl")]
    undone = list(set(df["hof_id"]) - set(done))
    num_data_points = len(undone)
    print(f"{num_data_points}/{len(df)} data points to process.")
    df["material_idx"] = df["hof_id"]
    df.set_index("hof_id", inplace=True)
    df["hof_id"] = df["material_idx"]

    results = p_umap(
        save_mof_graph_pkl,
        undone,
        num_cpus=num_workers,
    )

    failed_ids = [x[0] for x in results if x[1]]
    with open(Path(save_path) / "failed_id.txt", "a+") as f:
        f.write("\n".join(failed_ids))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--df_path",
        type=str,
        help="path to dataframe of material id/properties.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="path to save graph pickle files.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    preprocess_graph(
        '/data/user2/wty/HOF/MOFDiff/mofdiff/data/hof_data/hof.csv',
        '/data/user2/wty/HOF/MOFDiff/mofdiff/data/hof_save_path',
        args.num_workers,
        args.device,
    )
