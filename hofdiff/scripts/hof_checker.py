import os
import shutil
import argparse
from pathlib import Path
from mofchecker import MOFChecker
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

def check_and_copy_mof(cif_path, output_dir):
    """
    Checks the validity of a MOF structure and copies the file to the target directory if valid.

    Args:
        cif_path (str): Path to the MOF CIF file.
        output_dir (Path): Path to the output directory.

    Returns:
        tuple: (cif_path, is_valid), where is_valid indicates whether the structure is valid.
    """
    try:
        mofchecker = MOFChecker.from_cif(cif_path)
        result = mofchecker.get_mof_descriptors()

        # HOFDiff-20k filtering criteria
        # is_valid = not result["has_atomic_overlaps"] and \
        #            not result["has_metal"] and \
        #            not result["has_overcoordinated_c"] and \
        #            not result["has_overcoordinated_n"] and \
        #            not result["has_overcoordinated_h"]
        # HOFDiff-1300        
        is_valid = not result["has_atomic_overlaps"] and \
                   not result["has_metal"] and \
                   not result["has_overcoordinated_c"] and \
                   not result["has_overcoordinated_n"] and \
                   not result["has_overcoordinated_h"] and \
                   not result["has_lone_molecule"] and \
                   not result["has_undercoordinated_c"] and \
                   not result["has_undercoordinated_n"]
        if is_valid:
            shutil.copy(cif_path, output_dir / cif_path.name)
            print(f"Copied valid CIF: {cif_path.name}")
        return cif_path, is_valid
    except Exception as e:
        print(f"Error checking MOF {cif_path}: {e}")
        return cif_path, False


def process_mofs(input_dir, output_dir, n_workers=4):
    """
    Iterates over CIF files in the input directory, checks their validity, and copies valid files to the target directory.

    Args:
        input_dir (str): Path to the input directory containing CIF files.
        output_dir (str): Path to the output directory for storing valid CIF files.
        n_workers (int): Number of processes for parallel processing.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cif_files = list(input_dir.glob("*.cif"))
    print(f"Found {len(cif_files)} CIF files to process.")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        list(
            tqdm(
                executor.map(check_and_copy_mof, cif_files, [output_dir] * len(cif_files)),
                total=len(cif_files),
                desc="Processing CIF files"
            )
        )

    print(f"Processing complete. Valid CIF files have been copied to {output_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and copy valid MOF CIF files using mofchecker.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input folder containing CIF files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output folder to store valid CIF files.")
    parser.add_argument("--n_workers", type=int, default=4, help="Number of parallel processes to use.")

    args = parser.parse_args()
    process_mofs(args.input_dir, args.output_dir, args.n_workers)
