import os
import re
import shutil
import argparse
import subprocess
import torch
import numpy as np
from pathlib import Path

class HydrogenBondChecker:
    def __init__(self, cifs_path):
        self.cifs_path = cifs_path

    def get_Hbond_tensor(self, cif_id):
        donors = []
        lis_path = os.path.join(self.cifs_path, f"{cif_id}.lis")
        if not os.path.exists(lis_path):
            return None
        with open(lis_path, 'r') as file:
            content = file.read()
            match = re.search(r"(Nr Typ Res Donor.*?)(?=\n[A-Z])", content, re.DOTALL | re.MULTILINE)
        if match:
            for line in match.group(0).splitlines():
                if "?" in line:
                    continue
                line = re.sub(r'[<>]|Intra|\d\*|_[a-z0-9*]', ' ', line)
                columns = line.split()
                if len(columns) > 4 and (columns[0].isdigit() or columns[0].startswith('**')):
                    donor = re.search(r'[A-Za-z]+\d+[A-Z]*$', columns[2])
                    if donor and not donor.group().startswith('C'):
                        donors.append(donor.group())
        return torch.tensor([1 if donors else 0])

def run_platon_on_folder(folder):
    for file in os.listdir(folder):
        if file.endswith(".cif"):
            cif_path = os.path.join(folder, file)
            print(f"Running Platon on {cif_path}...")
            process = subprocess.Popen(['../PLATON/platon', cif_path], stdin=subprocess.PIPE, text=True)
            process.stdin.write("CALC HBONDS\n")
            process.stdin.write("exit\n")
            process.stdin.close()
            process.wait()

def copy_hbond_cifs(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    checker = HydrogenBondChecker(input_dir)
    for file in os.listdir(input_dir):
        if file.endswith(".lis"):
            cif_id = Path(file).stem
            hbond_tensor = checker.get_Hbond_tensor(cif_id)
            if hbond_tensor is not None and torch.any(hbond_tensor == 1):
                src_path = os.path.join(input_dir, f"{cif_id}.cif")
                dst_path = os.path.join(output_dir, f"{cif_id}.cif")
                if os.path.exists(src_path):
                    shutil.copy(src_path, dst_path)
                    print(f"Copied {cif_id}.cif to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Run Platon on CIFs and copy those with hydrogen bonds.")
    parser.add_argument("--input_dir", type=str, required=True, help="Input folder containing .cif files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output folder for filtered .cif files")
    args = parser.parse_args()

    run_platon_on_folder(args.input_dir)
    copy_hbond_cifs(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
