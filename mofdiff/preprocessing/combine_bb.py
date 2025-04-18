import argparse
import lmdb
import pickle
from pathlib import Path
from tqdm import tqdm
import os
import re
import time
from collections import defaultdict

# Merge CIF files into a single CIF file
def merge_cif_files(file_paths):
    # Initialize an empty list to store all atom data
    atom_data = []
    
    # Variables for storing cell data and atom header
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
    
    # Check existing files in the directory
    existing_files = os.listdir(directory)
    # Filter for files that match the "bb_{i}.cif" pattern and extract numbers
    bb_files = [f for f in existing_files if re.match(r'bb_\d+\.cif', f)]
    existing_numbers = [int(re.search(r'bb_(\d+)\.cif', f).group(1)) for f in bb_files]
    max_number = max(existing_numbers, default=0)  # Get the maximum number, defaulting to 0 if none

    # Generate the new filename
    new_filename = f"bb_{max_number + 1}.cif"
    merged_file_path = os.path.join(directory, new_filename)
    with open(merged_file_path, 'w') as output_file:
        output_file.write(merged_cif_content)
    print(f"Merged CIF file created as {merged_file_path}")

def build_graph(files):
    n = len(files)
    graph = defaultdict(list)
    for i in range(n):
        has_connection = False
        for j in range(i + 1, n):
            if cal_cif_combined(files[i], files[j]):
                graph[files[i]].append(files[j])
                graph[files[j]].append(files[i])
                has_connection = True
        if not has_connection:
            graph[files[i]]
    return graph

def find_connected_components(graph):
    visited = set()
    components = []

    def dfs(node, component):
        stack = [node]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                component.append(node)
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        stack.append(neighbor)

    for node in graph:
        if node not in visited:
            component = []
            dfs(node, component)
            components.append(component)
        elif node not in graph:  # Check if the node has no neighbors
            components.append([node])
    
    # Check for any nodes that may not be in the graph keys but exist in the values
    all_nodes = set(graph.keys()).union(set(n for neighbors in graph.values() for n in neighbors))
    isolated_nodes = all_nodes - set(graph.keys())
    for node in isolated_nodes:
        if node not in visited:
            components.append([node])

    return components

def get_cif_groups(folder_path):
    print("Processing folder:", folder_path)
    files = [f for f in os.listdir(folder_path) if f.endswith('.cif') and 'molecule' in f]
    files = [os.path.join(folder_path, f) for f in files]
    graph = build_graph(files)
    components = find_connected_components(graph)
    return components

def extract_number_from_path(path):
    match = re.search(r'molecule_(\d+)', path)
    return int(match.group(1)) if match else None

def contains_bb_cif_files(folder_path):
    for file_name in os.listdir(folder_path):
        if re.match(r'bb_\d+\.cif', file_name):
            return True
    return False

# Main process to process each subfolder within the specified directory
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
