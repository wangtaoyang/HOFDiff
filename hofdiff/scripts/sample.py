"""
sample coarse-grained MOFs using a trained CG diffusion model.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import torch
import numpy as np
from pytorch_lightning import seed_everything
from scipy.spatial import KDTree

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from hofdiff.common.atomic_utils import arrange_decoded_mofs
from hofdiff.common.eval_utils import load_hofdiff_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="PATH/hofdiff_ckpt",
    )
    parser.add_argument(
        "--bb_cache_path",
        type=str,
        default="PATH/bb_emb_space.pt",
    )

    parser.add_argument("--n_samples", default=4096, type=int)
    parser.add_argument("--batch_size", default=4096, type=int)
    parser.add_argument("--seed", default=7, type=int)

    # get datamodule prop_list
    args = parser.parse_args()
    seed_everything(args.seed)
    model_path = Path(args.model_path)
    model, cfg, bb_encoder = load_hofdiff_model(model_path)
    model = model.to("cuda")
    # state_dict = torch.load(args.bb_cache_path)
    # print(state_dict)
    # print(state_dict.keys())
    all_data, all_z = torch.load(args.bb_cache_path)
    kdtree = KDTree(all_z)
    output = defaultdict(list)
    n_batch = int(np.ceil(args.n_samples / args.batch_size))
    for idx in range(n_batch):
        z = torch.randn(args.batch_size, model.hparams.latent_dim).to("cuda")
        samples = model.sample(z.shape[0], z, save_freq=False)
        print("samples:",samples)
        mofs = arrange_decoded_mofs(samples, kdtree, all_data, False)
        results = {"mofs": mofs, "z": all_z}
        for k, v in results.items():
            output[k].extend(v)
    output["z"] = torch.stack(output["z"], dim=0)
    savedir = f"samples_{args.n_samples}_seed_{args.seed}"
    (model_path / savedir).mkdir(exist_ok=True)
    torch.save(output, model_path / savedir / "samples.pt")
