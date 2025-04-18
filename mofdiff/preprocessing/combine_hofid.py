import argparse
import pickle
import pandas as pd
from p_tqdm import p_umap
from openbabel import openbabel as ob
import numpy as np
from openbabel import pybel
import sys
import os
import re
from ase.io import read, write
from ase.geometry import distance
from pathlib import Path
from collections import defaultdict
import time

# Importing necessary utility functions from mofdiff
from mofdiff.common.atomic_utils import pyg_graph_from_cif, assemble_local_struct, readcif, lattice_params_to_matrix_torch, read_cif_bonds, compute_image_flag, get_atomic_graph, compute_distance_matrix, frac2cart
from mofdiff.common.data_utils import frac_to_cart_coords
from mofdiff.common.constants import COVALENT_RADII

def has_no_overlapping_atoms(cif_path, threshold=0.7):
    """
    Check if the given CIF file has any overlapping atoms. If no overlapping atoms, return True; otherwise, return False.

    :param cif_path: Path to the CIF file
    :param threshold: The threshold to determine if atoms are overlapping, default is 0.7
    :return: True if there are no overlapping atoms, False if there are overlapping atoms
    """
    print(f"Checking {cif_path} for overlapping atoms.")
    obConversion = ob.OBConversion()
    obConversion.SetInFormat("cif")
    mol = ob.OBMol()

    if not obConversion.ReadFile(mol, cif_path):
        print(f"Failed to read {cif_path} file.")
        return False

    # Separate all connected fragments in the molecule
    fragments = mol.Separate()

    for frag in fragments:
        frag_mol = ob.OBMol(frag)
        other_atoms = []

        # Check if any atoms in the molecule are overlapping
        for atom in ob.OBMolAtomIter(frag_mol):
            pos = np.array([atom.GetX(), atom.GetY(), atom.GetZ()])
            e1 = atom.GetType()
            
            for other_atom in other_atoms:
                other_pos = np.array([other_atom.GetX(), other_atom.GetY(), other_atom.GetZ()])
                e2 = other_atom.GetType()
                
                # Remove numbers from atom types (e.g., H2 -> H)
                e1 = ''.join([i for i in e1 if not i.isdigit()])
                e2 = ''.join([i for i in e2 if not i.isdigit()])
                # Calculate the minimum covalent radius based on atom types
                try:
                    min_threshold = min(COVALENT_RADII[e1], COVALENT_RADII[e2])
                except KeyError as e:
                    continue  # Skip unrecognized atom types
                if np.linalg.norm(pos - other_pos) < threshold * min_threshold:
                    return False  # Found overlapping atoms, return False

            other_atoms.append(atom)

    return True  # No overlapping atoms, return True

def merge_cif_files(file_paths):
    # Initialize an empty list to store all atom data
    atom_data = []
    # Get the base path from the first file in the list
    base_path = Path(file_paths[0]).parent
    hofid = base_path.name
    print(f"Making HOF: {hofid}")
    
    # Initialize variables for storing cell data and atom headers
    cell_data = ""
    atom_header = ""
    
    # Flag to process cell data and atom headers only for the first file
    first_file = True
    
    # Iterate over all provided CIF file paths
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            # Flag to start collecting atom data
            start_collecting_atoms = False
            
            for line in lines:
                if first_file:
                    # Process cell and atom headers only for the first file
                    if "CIF file" in line:
                        cell_data += line
                    elif "data_I" in line:
                        cell_data += line
                    elif "_chemical" in line:
                        cell_data += line
                    elif "_cell_length_a" in line or "_cell_length_b" in line or "_cell_length_c" in line:
                        cell_data += line
                    elif "_cell_angle_alpha" in line or "_cell_angle_beta" in line or "_cell_angle_gamma" in line:
                        cell_data += line
                    elif "_atom_site_label" in line:
                        atom_header = line
                        start_collecting_atoms = True  # Start collecting atom data
                    elif start_collecting_atoms:
                        if line.strip() and not line.startswith('loop_'):
                            atom_data.append(line)
                else:
                    # For subsequent files, only collect atom data
                    if "_atom_site_occupancy" in line:
                        start_collecting_atoms = True  # Confirm start of atom data collection
                    elif start_collecting_atoms:
                        if line.strip() and not line.startswith('loop_'):
                            atom_data.append(line)

        # Set the flag to False after processing the first file
        if first_file:
            first_file = False

    # Combine the cell data, atom header, and atom data into a single CIF content
    merged_cif_content = cell_data + 'loop_\n' + atom_header + ''.join(atom_data)
    directory = os.path.dirname(file_paths[0])
    
    # Extract numbers from the file names, assuming the file name format is "molecule_X.cif"
    numbers = [os.path.basename(path).split('.')[0].split('_')[-1] for path in file_paths]
    # Sort the numbers and generate a new file name
    sorted_numbers = sorted(numbers, key=int)
    new_filename = hofid + '.cif'
    merged_file_path = os.path.join(directory, new_filename)
    
    # Write the merged CIF file
    with open(merged_file_path, 'w') as output_file:
        output_file.write(merged_cif_content)
    
    # Remove overlapping atoms from the merged CIF file
    remove_overlapping_atoms_from_cif(merged_file_path, merged_file_path)
    print(f"Merged CIF file created as {merged_file_path}")

def make_combined_cif_files(base_path):
    # Regex pattern to match "molecule_{i}.cif" file names
    cif_pattern = re.compile(r"^molecule.*\.cif$")
    
    # List to store non-conforming CIF files
    non_conforming_cifs = []
    hid = Path(base_path).parts[-1]
    
    # Iterate through all files in the base directory
    for file in os.listdir(base_path):
        if file.endswith('.cif') and not cif_pattern.match(file) and file != f'{hid}.cif':
            bb_path = os.path.join(base_path, file)
            if has_no_overlapping_atoms(bb_path):
                print(f"{bb_path} has no overlapping")
                non_conforming_cifs.append(os.path.join(base_path, file))
    
    if len(non_conforming_cifs) == 0:
        print("No non-conforming .cif files in the directory.")
        print(base_path)
        return
    merge_cif_files(non_conforming_cifs)

def remove_overlapping_atoms_from_cif(input_cif, output_cif, threshold=0.001):
    """
    Read a .cif file, remove atoms that are completely overlapping, and save the processed .cif file.
    
    Parameters:
    - input_cif (str): Path to the input .cif file.
    - output_cif (str): Path to the output .cif file.
    - threshold (float): The distance threshold for determining overlapping atoms, default is 0.001 Å.
    """
    
    # Read the CIF file
    try:
        structure = read(input_cif)
    except:
        print(f"Failed to read {input_cif} file.")
        return
    
    # Remove overlapping atoms
    positions = structure.get_positions()
    unique_indices = []
    unique_positions = []

    for i, pos in enumerate(positions):
        if all(np.linalg.norm(pos - other) > threshold for other in unique_positions):
            unique_indices.append(i)  # Store the index
            unique_positions.append(pos)

    # Create a new structure with unique atoms
    new_structure = structure[unique_indices]
    
    # Save the deduplicated structure to a new CIF file
    write(output_cif, new_structure)
    
    # Use Pybel to ensure compatibility with CIF format
    mol = next(pybel.readfile("cif", output_cif))
    mol.write("cif", output_cif, overwrite=True)

def process_folders(root_path):
    for sub_folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, sub_folder)
        # Skip folders with more than 100 CIF files or containing bb_* files
        if len(os.listdir(folder_path)) > 100 or contains_bb_cif_files(folder_path):
            continue
        print("Processing folder:", folder_path)
        s = time.time()
        cif_groups = get_cif_groups(folder_path)
        for group in cif_groups:
            print(group)
            merge_cif_files(group)
        e = time.time()
        print(f"Total time taken: {e - s} seconds")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Merge CIF files from different directories")
    parser.add_argument(
        "--root_path",
        type=str,
        required=True,
        help="Root path where subfolders with CIF files are located",
    )
    args = parser.parse_args()

    # Call the processing function with the provided root path
    process_folders(args.root_path)
