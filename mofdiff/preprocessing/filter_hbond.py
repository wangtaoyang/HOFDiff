import os
import shutil
import torch
import numpy as np
import re


class HydrogenBondChecker:
    def __init__(self, cifs_path):
        self.cifs_path = cifs_path

    def get_Hbond_lists(self, cif_id):
        donors, hs, acceptors = [], [], []
        lis_path = os.path.join(self.cifs_path, f"{cif_id}.lis")
        if not os.path.exists(lis_path):
            print(f"No LIS file found for CIF ID {cif_id}.")
            return donors, hs, acceptors
        with open(lis_path, 'r') as file:
            content = file.read()
            print("cif", cif_id)
            # print("content:", content)
            data_block_match = re.search(r"(Nr Typ Res Donor.*?)(?=\n[A-Z])", content, re.DOTALL | re.MULTILINE)
        if data_block_match:
            data_block = data_block_match.group(0)
            lines = data_block.splitlines()
            for line in lines:
                if "?" in line:
                    continue
                line = re.sub(r'Intra', ' ', line)
                line = re.sub(r'\d\*', '1 ', line)
                line = re.sub(r'_[a-z*]', ' ', line)
                line = re.sub(r'_[0-9*]', ' ', line)
                line = re.sub(r'_', ' ', line)
                line = re.sub(r'>', ' ', line)
                line = re.sub(r'<', ' ', line)
                columns = line.split()
                if len(columns) > 1 and (columns[0].isdigit() or columns[0].startswith('**')) and columns[1].isdigit():
                    donor = re.search(r'[A-Za-z]+\d+[A-Z]*$', columns[2])
                    h = re.search(r'[A-Za-z]+\d+[A-Z]*$', columns[3])
                    acceptor = re.search(r'[A-Za-z]+\d+[A-Z]*$', columns[4])
                    if donor and not donor.group().startswith('C'):
                        donors.append(donor.group())
                        if h:
                            hs.append(h.group())
                        if acceptor:
                            acceptors.append(acceptor.group())
        # print(donors, hs, acceptors)
        return donors, hs, acceptors

    @staticmethod
    def read_cif_extract_block(file_path):
        with open(file_path, 'r') as file:
            content = file.read()
        start = content.find('_atom_site_type_symbol')
        if start == -1:
            return None, 0
        data_block = content[start:].split('\n')[1:]
        return data_block, len(data_block)

    @staticmethod
    def extract_atom_labels(data_block):
        atom_labels = []
        for line in data_block:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            atom_labels.append(parts[0])
        return atom_labels

    @staticmethod
    def classify_atoms(atom_labels, donors, hs, acceptors):
        atom_classification = []
        for label in atom_labels:
            if label in donors:
                atom_classification.append(1)
            elif label in hs:
                atom_classification.append(2)
            elif label in acceptors:
                atom_classification.append(3)
            else:
                atom_classification.append(0)
        return atom_classification

    def get_Hbond(self, cif_id):
        donors, hs, acceptors = self.get_Hbond_lists(cif_id)
        file_path = os.path.join(self.cifs_path, f"{cif_id}.cif")
        data_block, _ = self.read_cif_extract_block(file_path)
        if data_block:
            atom_labels = self.extract_atom_labels(data_block)
            atom_classification = self.classify_atoms(atom_labels, donors, hs, acceptors)
            return torch.LongTensor(np.array(atom_classification, dtype=np.int8))
        else:
            return None


def find_and_copy_cif_files(src_folder, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    checker = HydrogenBondChecker(src_folder)
    for file in os.listdir(src_folder):
        if file.endswith(".lis"):
            cif_id = os.path.splitext(file)[0]
            hbond_tensor = checker.get_Hbond(cif_id)
            print("hbond_tensor:", hbond_tensor)
            if hbond_tensor is not None and torch.any(hbond_tensor == 1):  # 检查是否存在氢键
                cif_path = os.path.join(src_folder, f"{cif_id}.cif")
                if os.path.exists(cif_path):
                    shutil.copy(cif_path, dest_folder)
                    print(f"Copied {cif_id}.cif to {dest_folder}")
                else:
                    print(f"No CIF file found for LIS ID {cif_id}.")


# 示例用法
src_folder = "/data/user2/wty/HOF/MOFDiff/mofdiff/data/mof_models/mof_models/bwdb_hoff/temp_no_overlap"  # 替换为实际的源文件夹路径
dest_folder = "/data/user2/wty/HOF/MOFDiff/mofdiff/data/mof_models/mof_models/bwdb_hoff/temp_hbond"  # 替换为实际的目标文件夹路径

find_and_copy_cif_files(src_folder, dest_folder)
