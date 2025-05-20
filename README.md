# HOFDiff

HOFDiff is a diffusion model based on MOFDiff, designed for generating coarse-grained HOF structures. Building upon the MOFDiff model, we have added several modifications for HOF decomposition and reconstruction. Additionally, we slightly adjusted the building block combination logic in MOFDiff to enable its application in HOF generation. This modified version of MOFDiff is used to train HOFDiff for generating coarse-grained HOF structures.



## Installation

We recommend using [mamba](https://mamba.readthedocs.io/en/latest/) rather than conda to install the dependencies to increase installation speed. First install `mamba` following the intructions in the [mamba repository](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html). (Note: a `reqirements.txt` mirror of `env.yml` is provided for compatibility with CI/CD; however, we do not recommend building the environment with `pip`.)

Install dependencies via `mamba`:

```bash
mamba env create -f env.yml
```

To ensure compatibility with your PyTorch and CUDA versions, download the corresponding `.whl` files from [https://data.pyg.org/whl](https://data.pyg.org/whl) and install them via `pip`:

```bash
pip install torch_cluster-1.6.3+pt23cu121-cp39-cp39-linux_x86_64.whl
pip install torch_scatter-2.1.2+pt23cu121-cp39-cp39-linux_x86_64.whl
pip install torch_sparse-0.6.18+pt23cu121-cp39-cp39-linux_x86_64.whl
```

> Match `pt23`, `cu121`, and `cp39` to your PyTorch, CUDA, and Python versions.

Then install `hofdiff` as a package:

```bash
pip install -e .
```

 

Configure the `.env` file to set correct paths to various directories, dependent on the desired functionality. An [example](https://github.com/wangtaoyang/HOFDiff/blob/main/.env) `.env` file is provided in the repository.

For model training, please set the learning-related paths.

- PROJECT_ROOT: the parent HOFDiff directory
- DATASET_DIR: the directory containing the .lmdb file produced by processing the data
- LOG_DIR: the directory to which logs will by written
- HYDRA_JOBS: the directory to which Hydra output will be written
- WANDB_DIR: the directory to which WandB output will be written

## Process data

In this step, we will split and reorganize the first real HOF dataset we have collected. Using a connected graph algorithm, the real HOF will first be broken down into distinct molecular fragments, labeled as molecule_{x}.cif. Then, considering periodic boundary conditions, these fragments will be recombined to form different building blocks of the HOF material, which will be used for encoder training. Finally, the building blocks will be reconstructed into a complete HOF to generate the .pkl files for HOFDiff training.

```bash
# 1. Split the HOF CIF files into molecule_{i}.cif and output them to the target folder. The cif_folder contains several HOF CIF files. After running the script, the output_folder will contain subfolders named after each HOF, with each folder containing several split molecule_{i}.cif files.
python hofdiff/preprocessing/split_hof.py --cif_file_path ./hof_database_cifs_raw --output_directory ./hofdiff/data/hof_data

# 2. Merge the `molecule_{i}.cif` files from the `output_folder` into several `bb.cif` files based on periodic boundary conditions.  Note: This step may take a significant amount of time.
python hofdiff/preprocessing/combine_bb.py --bbs_path ./hofdiff/data/hof_data

# 3. Reassemble the valid molecule_{i}.cif files into complete HOF CIF files.
python hofdiff/preprocessing/combine_cifid.py --root_path ./hofdiff/data/hof_data

# 4. Generate the data for HOFDiff training.
python hofdiff/preprocessing/preprocess_hof.py --df_path ./hofdiff/data/hof.csv --save_path ./hofdiff/data/hof_save_path --device cpu --num_workers 4

# 5. Store the data in pkl format.
python hofdiff/preprocessing/save_to_lmdb.py --pkl_path ./hofdiff/data/hof_save_path  --lmdb_path ./hofdiff/data/lmdb_data
```

If you plan to train or generate structures using our real HOF dataset, we recommend downloading our preprocessed data from **[https://figshare.com/articles/dataset/HOFDiff/28856456](https://figshare.com/articles/dataset/HOFDiff/28856456)** to save time.

### Option 1: Skip to Step 4

Download and extract the `hof_save_path` folder directly to complete **Step 4**.

### Option 2: Use Preprocessed LMDB Files

Download the `lmdb_data` folder, which contains:

- Precomputed building block embeddings
- A preprocessed HOF database

---

### Skip Training (Use Pretrained Models)

If you want to skip the training steps, you can directly download our pretrained models from the same link:

- **Building Block Encoder**  
  Download `hofdiff_bb_encoder.ckpt` and place it at:  
  `${oc.env:HYDRA_JOBS}/bb/${expname}/`  
  - `oc.env:HYDRA_JOBS` is defined in your `.env` file  
  - `expname` is defined in `configs/bb.yaml`

- **Diffusion Model**  
  Download `hofdiff_diffusion_model.ckpt` and place it at:  
  `${oc.env:HYDRA_JOBS}/hof_models/${expname}/`  
  - `oc.env:HYDRA_JOBS` is defined in your `.env` file  
  - `expname` is defined in `configs/hofdiff.yaml`

Once these models are in place, you can jump directly to the **"Generating CG HOF Structures"** section.

## Training

### training the building block encoder

Before training the diffusion model, we need to train the building block encoder. The building block encoder is a graph neural network that encodes the building blocks of HOFs. The building block encoder is trained with the following command:

```bash
python hofdiff/scripts/train.py --config-name=bb
```



The default output directory is `${oc.env:HYDRA_JOBS}/bb/${expname}/`. `oc.env:HYDRA_JOBS` is configured in `.env`. `expname` is configured in `configs/bb.yaml`. We use hydra for config management. All configs are stored in `configs/` You can override the default output directory with command line arguments. For example:

```
python hofdiff/scripts/train.py --config-name=bb 
```

### training coarse-grained diffusion model for HOFs

The output directory where the building block encoder is saved: `bb_encoder_path` is needed for training the diffusion model. By default, this path is `${oc.env:HYDRA_JOBS}/bb/${expname}/`, as defined [above](https://github.com/wangtaoyang/HOFDiff#training-the-building-block-encoder). Train/validation splits are defined in [splits](https://github.com/wangtaoyang/HOFDiff/blob/main/splits), with examples provided for BW-DB. With the building block encoder trained to convergence, train the CG diffusion model with the following command:

```bash
python hofdiff/scripts/train.py data.bb_encoder_path=${bb_encoder_path}
```



## Generating CG HOF structures

With a trained CG diffusion model `${diffusion_model_path}`, you can generate random coarse-grained (CG) HOF structures using the following command:

- `${bb_cache_path}` should point to the trained building block encoder output `bb_emb_space.pt`, which will be:
  - automatically generated in the `DATASET_DIR` folder after training the diffusion model, or  
  - directly available by downloading and extracting the `lmdb_data` folder from our dataset on **[Figshare](https://figshare.com/articles/dataset/HOFDiff/28856456)**.

```bash
python hofdiff/scripts/sample.py --model_path ${diffusion_model_path} --bb_cache_path ${bb_cache_path}
```

Available arguments for `sample.py`  can be found in the respective files. The generated CG HOF structures will be saved in `${sample_path}=${diffusion_model_path}/${sample_tag}` as `samples.pt`.

The CG structures generated with the diffusion model are not guaranteed to be realizable. We need to assemble the CG structures to recover the all-atom HOF structures. The following sections describe how to assemble the CG HOF structures, and all steps further do not require a GPU.

## Assemble all-atom HOFs

Assemble all-atom HOF structures from the CG HOF structures with the following command:

```bash
python hofdiff/scripts/assemble.py --input ${sample_path}/samples.pt
```

This command will assemble the CG HOF structures in `${sample_path}` and save the assembled HOFs in `${sample_path}/assembled.pt`. The cif files of the assembled HOFs will be saved in `${sample_path}/cif`.



## Generating Large Batches of Simulated HOFs

In this experiment, we use the following script to generate a large number of virtual HOFs on 8 RTX 4090 GPUs (n_sample = 4096, batch_size = 4096):

```bash
#!/bin/bash
cd . || exit

diffusion_model_path="./hofdiff/data/hof_models/hof_models/bwdb_hoff"
bb_cache_path="./hofdiff/data/lmdb_data/bb_emb_space.pt"

# Loop through GPU indexes 0 to 7
for i in {0..7}
do
    CUDA_VISIBLE_DEVICES=$i python hofdiff/scripts/sample.py --model_path ${diffusion_model_path} --bb_cache_path ${bb_cache_path} --seed $i &
done

wait
```

This will generate approximately 300,000 virtual HOFs for screening. You can move all of them to a single folder, e.g., `all_hof`.

## Screening with Platon and MOFchecker

Platon can filter CIF files containing hydrogen bond information:

```bash
python hofdiff/scripts/filter_hbond_cifs.py --input_dir ./all_hof --output_dir ./hbond_cifs
```

MOFchecker performs quick sanity checks on crystal structures of metal-organic frameworks (MOFs). We can use some HOF-appropriate rules from MOFChecker to select more reasonable HOF structures.

First, download mofchecker following the instructions at: https://github.com/lamalab-org/mofchecker

Then use this script to further filter HOFs containing hydrogen bonds:

```bash
python hofdiff/scripts/hof_checker.py --input_dir ./hbond_cifs --output_dir ./hofdiff-1300 --n_workers 4
```