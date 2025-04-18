# HOFDiff

HOFDiff is a diffusion model based on MOFDiff, designed for generating coarse-grained HOF structures. Building upon the MOFDiff model, we have added several modifications for HOF decomposition and reconstruction. Additionally, we slightly adjusted the building block combination logic in MOFDiff to enable its application in HOF generation. This modified version of MOFDiff is used to train HOFDiff for generating coarse-grained HOF structures.



## Installation

We recommend using [mamba](https://mamba.readthedocs.io/en/latest/) rather than conda to install the dependencies to increase installation speed. First install `mamba` following the intructions in the [mamba repository](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html). (Note: a `reqirements.txt` mirror of `env.yml` is provided for compatibility with CI/CD; however, we do not recommend building the environment with `pip`.)

Install dependencies via `mamba`:

```bash
mamba env create -f env.yml
```



Then install `mofdiff` as a package:

```bash
pip install -e 
```

 

Configure the `.env` file to set correct paths to various directories, dependent on the desired functionality. An [example](https://github.com/wangtaoyang/HOFDiff/blob/main/.env) `.env` file is provided in the repository.

For model training, please set the learning-related paths.

- PROJECT_ROOT: the parent MOFDiff directory
- DATASET_DIR: the directory containing the .lmdb file produced by processing the data
- LOG_DIR: the directory to which logs will by written
- HYDRA_JOBS: the directory to which Hydra output will be written
- WANDB_DIR: the directory to which WandB output will be written

## Process data

In this step, we will split and reorganize the first real HOF dataset we have collected. Using a connected graph algorithm, the real HOF will first be broken down into distinct molecular fragments, labeled as molecule_{x}.cif. Then, considering periodic boundary conditions, these fragments will be recombined to form different building blocks of the HOF material, which will be used for encoder training. Finally, the building blocks will be reconstructed into a complete HOF to generate the .pkl files for HOFDiff training.

```bash
# 1. Split the HOF CIF files into molecule_{i}.cif and output them to the target folder. The cif_folder contains several HOF CIF files. After running the script, the output_folder will contain subfolders named after each HOF, with each folder containing several split molecule_{i}.cif files.
python mofdiff/preprocessing/split_hof.py /path/to/cif_folder /path/to/output_folder

# 2. Merge the molecule_{i}.cif files from the output_folder into several bb.cif files based on periodic boundary conditions.
python mofdiff/preprocessing/combine_bb.py --root_path /path/to/your/root/output_folder

# 3. Reassemble the valid molecule_{i}.cif files into complete HOF CIF files.
python mofdiff/preprocessing/combine_cifid.py --root_path /path/to/your/root/output_folder

# 4. Generate the data for HOFDiff training.
python mofdiff/preprocessing/preprocess_hof.py --df_path /path/to/hof.csv --save_path /path/to/hof_save_path --device cpu --num_workers 4

# 5. Store the data in pkl format.
python mofdiff/preprocessing/save_to_lmdb.py --graph_path ${raw_path}/graphs --save_path ${raw_path}/lmdbs
```



## Training

### training the building block encoder

Before training the diffusion model, we need to train the building block encoder. The building block encoder is a graph neural network that encodes the building blocks of HOFs. The building block encoder is trained with the following command:

```bash
python mofdiff/scripts/train.py --config-name=bb
```



The default output directory is `${oc.env:HYDRA_JOBS}/bb/${expname}/`. `oc.env:HYDRA_JOBS` is configured in `.env`. `expname` is configured in `configs/bb.yaml`. We use hydra for config management. All configs are stored in `configs/` You can override the default output directory with command line arguments. For example:

```
python mofdiff/scripts/train.py --config-name=bb expname=bwdb_bb_dim_64 model.latent_dim=64
```

### training coarse-grained diffusion model for HOFs

The output directory where the building block encoder is saved: `bb_encoder_path` is needed for training the diffusion model. By default, this path is `${oc.env:HYDRA_JOBS}/bb/${expname}/`, as defined [above](https://github.com/wangtaoyang/HOFDiff#training-the-building-block-encoder). Train/validation splits are defined in [splits](https://github.com/wangtaoyang/HOFDiff/blob/main/splits), with examples provided for BW-DB. With the building block encoder trained to convergence, train the CG diffusion model with the following command:

```bash
python mofdiff/scripts/train.py data.bb_encoder_path=${bb_encoder_path}
```



## Generating CG HOF structures

To use the pretrained models, please extract `pretrained.tar.gz` and `bb_emb_space.tar.gz` into `${oc.env:PROJECT_ROOT}/pretrained`.

With a trained CG diffusion model `${diffusion_model_path}`, generate random CG HOF structures with the following command, where `${bb_cache_path}` is the path to the trained building encoder `bb_emb_space.pt`, either sourced from the pretrained models or generated as described [above](https://github.com/wangtaoyang/HOFDiff#training-the-building-block-encoder).

```bash
python mofdiff/scripts/sample.py --model_path ${diffusion_model_path} --bb_cache_path ${bb_cache_path}
```

Available arguments for `sample.py` and `optimize.py` can be found in the respective files. The generated CG HOF structures will be saved in `${sample_path}=${diffusion_model_path}/${sample_tag}` as `samples.pt`.

The CG structures generated with the diffusion model are not guaranteed to be realizable. We need to assemble the CG structures to recover the all-atom HOF structures. The following sections describe how to assemble the CG HOF structures, and all steps further do not require a GPU.

## Assemble all-atom HOFs

Assemble all-atom HOF structures from the CG HOF structures with the following command:

```bash
python mofdiff/scripts/assemble.py --input ${sample_path}/samples.pt
```

This command will assemble the CG HOF structures in `${sample_path}` and save the assembled HOFs in `${sample_path}/assembled.pt`. The cif files of the assembled HOFs will be saved in `${sample_path}/cif`.