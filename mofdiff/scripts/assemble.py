"""
Assembled atomic MOFs from sampled CG MOF strctures and save them to cif files.
"""
import argparse
from pathlib import Path
import json
import time
import torch
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from mofdiff.common.atomic_utils import mof2cif_with_bonds
from mofdiff.common.optimization import (
    annealed_optimization,
    assemble_mof,
    feasibility_check,
)
from tqdm import tqdm


def assemble_one(
    mof,
    verbose=True,
    rounds=3,
    sigma_start=3.0,
    sigma_end=0.3,
    max_neighbors_start=30,
    max_neighbors_end=1,
):
    # if not feasibility_check(mof):
    #     return None
    # print("mof:",mof)
    sigma_schedule = np.linspace(sigma_start, sigma_end, rounds)
    max_neighbors_schedule = (
        np.linspace(max_neighbors_start, max_neighbors_end, rounds).round().astype(int)
    )

    now = time.time()
    results, v = annealed_optimization(
        mof,
        0,
        optimize=False,
        sigma_schedule=sigma_schedule,
        max_neighbors_schedule=max_neighbors_schedule,
        maxiter=1000,
        verbose=verbose,
    )
    # print("results:", results)
    elapsed = time.time() - now
    vecs = torch.from_numpy(results["x"]).view(mof.num_atoms, 3).float()
    # print("vecs:",vecs)
    # print(" ")
    # for bb in mof.bbs:
    #     print("bb.atom_types:", bb.atom_types)
    bb_local_vectors = [bb.local_vectors for bb in mof.bbs]
    # print("mof:", mof.all_atom_coords)
    assembled_rec = assemble_mof(mof, vecs, bb_local_vectors=bb_local_vectors)
    # print("assembled_rec:",assembled_rec.frac_coords)
    # print("assembled_rec:",assembled_rec)
    assembled_rec.opt_v = v
    assembled_rec.assemble_time = elapsed
    return assembled_rec


def main(input_file, verbose=True, rounds=3):
    samples = torch.load(input_file)
    save_dir = Path(input_file).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    N_samples = len(samples["mofs"])
    samples["assembled"] = [[] for _ in range(N_samples)]
    samples["assemble_info"] = [[] for _ in range(N_samples)]

    cif_dir = save_dir / "cif"
    cif_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(N_samples)):
    # i = 3
        try:    
            mof = samples["mofs"][i].detach().cpu()
            # print(mof)
            info = "info"
            assembled = assemble_one(mof, verbose=verbose, rounds=rounds)
            if assembled is None:
                info = "infeasible for assembly"
                print("Matched connection criteria violated")
            else:
                mof2cif_with_bonds(assembled, cif_dir / f"sample_{i}.cif")
            samples["assembled"][i].append(assembled)
            samples["assemble_info"][i].append(info)
        except Exception as e:
            print(f"Error: {e}")
            info = e
            samples["assembled"][i].append(None)
            samples["assemble_info"][i].append(info)

    torch.save(samples, save_dir / "assembled.pt")

    if "opt_args" in samples:
        with open(save_dir / "opt_args.json", "w") as f:
            json.dump(samples["opt_args"], f)

    print(f"{input_file} -- done!")
    return save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--rounds", type=int, default=3)

    args = parser.parse_args()
    main(args.input, args.verbose, args.rounds)
