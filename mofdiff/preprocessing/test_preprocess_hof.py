from pathlib import Path
import argparse
import pickle
import pandas as pd
from p_tqdm import p_umap
import re

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from mofdiff.common.atomic_utils import pyg_graph_from_cif, assemble_local_struct
import multiprocessing as mp

mp.set_start_method("spawn", force=True)

def assign_cif_files(base_path):
    # 正则表达式，用于匹配符合 "molecule_{i}.cif" 格式的文件名
    cif_pattern = re.compile(r"^molecule_\d+\.cif$")
    
    # 用于存储不符合特定格式的cif文件路径
    non_conforming_cifs = []
    hid = base_path.parts[-1]
    # 遍历base_path目录下的所有文件
    for file in os.listdir(base_path):
        if file.endswith('.cif') and not cif_pattern.match(file) and file != f'{hid}.cif':
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
            base_path = Path(f'/opt/wty/hof_synthesis/HOF/MOFDiff/mofdiff/data/hof_data/{m_id}')
            print(base_path)
            # use the metaloxo algorithm for deconstruction.
            # g_nodes_path, g_linkers_path, g_node_bridges_path = assign_cif_files(base_path)
            bb_cifs = assign_cif_files(base_path)
            print("bb_cifs:",bb_cifs)
            g_bb_cifs = []
            for bb_cif in bb_cifs:
                g_bb_cifs.append(pyg_graph_from_cif(Path(bb_cif), Hbond=False))
            print("g_bb_cifs:", len(g_bb_cifs))
            if use_asr:
                g_asr = pyg_graph_from_cif(Path(base_path / f'{m_id}.cif'), Hbond=True)
            else:
                g_asr = None
            # print("g_node_bridges",g_node_bridges)
            # print("g_asr",g_asr)
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
        "--mofid_path",
        type=str,
        help="path to extracted mofids.",
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
        '/opt/wty/hof_synthesis/HOF/MOFDiff/mofdiff/data/hof_data/hof_test.csv',
        '/opt/wty/hof_synthesis/HOF/MOFDiff/mofdiff/data/hof_save_path',
        args.num_workers,
        args.device,
    )
