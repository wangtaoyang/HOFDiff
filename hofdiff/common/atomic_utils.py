import itertools
import numpy as np
import torch
import os
import re

from ase.data import chemical_symbols
from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import StructureGraph, MoleculeGraph
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data

import sys
import os

from hofdiff.common.io_utils import readcif, writecif
from hofdiff.common.data_utils import (
    lattice_params_to_matrix_torch,
    cart_to_frac_coords,
)
from hofdiff.common.constants import (
    COVALENT_RADII,
    lanthanides,
    alkali,
    metals,
    INDEX2BTYPE,
)

from openbabel import pybel
from openbabel import openbabel as ob
from pymatgen.io.babel import BabelMolAdaptor
from pymatgen.io.cif import CifParser, CifWriter
    
SUPERCELLS = torch.FloatTensor(list(itertools.product((-2, -1, 0, 1, 2), repeat=3)))


class ConnectedComponents(MessagePassing):
    def __init__(self):
        super().__init__(aggr="max")

    def forward(self, n_node, edge_index):
        if len(edge_index) == 0:
            return torch.arange(n_node)

        x = torch.arange(n_node).view(-1, 1).to(edge_index.device)
        last_x = torch.zeros_like(x)

        while not x.equal(last_x):
            last_x = x.clone()
            x = self.propagate(edge_index, x=x)
            x = torch.max(x, last_x)

        _, perm = torch.unique(x, return_inverse=True)
        perm = perm.view(-1).to(edge_index.device)
        return perm

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out


connected_components = ConnectedComponents()


def frac2cart(fcoord, cell):
    return fcoord @ cell


def cart2frac(coord, cell):
    # pylint: disable=E1102
    invcell = torch.linalg.inv(cell)
    return coord @ invcell


def compute_distance_matrix(cell, cart_coords, num_cells=1):
    pos = torch.arange(-num_cells, num_cells + 1, 1).to(cell.device)
    combos = (
        torch.stack(torch.meshgrid(pos, pos, pos, indexing="xy"))
        .permute(3, 2, 1, 0)
        .reshape(-1, 3)
        .to(cell.device)
    )
    shifts = torch.sum(cell.unsqueeze(0) * combos.unsqueeze(-1), dim=1)
    shifted = cart_coords.unsqueeze(1) + shifts.unsqueeze(0)
    dist = cart_coords.unsqueeze(1).unsqueeze(1) - shifted.unsqueeze(0)
    # +eps to avoid nan in differentiation
    dist = (dist.pow(2).sum(dim=-1) + 1e-32).sqrt()
    distance_matrix = dist.min(dim=-1)[0]
    return distance_matrix

def compute_distance_matrix_offset(cell, cart_coords, num_cells=1):
    pos = torch.arange(-num_cells, num_cells + 1, 1).to(cell.device)
    combos = (
        torch.stack(torch.meshgrid(pos, pos, pos, indexing="xy"))
        .permute(3, 2, 1, 0)
        .reshape(-1, 3)
        .to(cell.device)
    )
    shifts = torch.sum(cell.unsqueeze(0) * combos.unsqueeze(-1), dim=1)
    shifted = cart_coords.unsqueeze(1) + shifts.unsqueeze(0)
    dist = cart_coords.unsqueeze(1).unsqueeze(1) - shifted.unsqueeze(0)
    # +eps to avoid nan in differentiation
    dist = (dist.pow(2).sum(dim=-1) + 1e-32).sqrt()
    min_dist, min_indices = dist.min(dim=-1)  # min_dist: (N x N), min_indices: (N x N)
    image_offsets = combos[min_indices]  # (N x N x 3)
    return min_dist, image_offsets



def compute_image_flag(cell, fcoord1, fcoord2):
    supercells = SUPERCELLS.to(cell.device)
    fcoords = fcoord2[:, None] + supercells
    coords = fcoords @ cell
    coord1 = fcoord1 @ cell
    dists = torch.cdist(coord1[:, None], coords).squeeze()
    image = dists.argmin(dim=-1)
    return supercells[image].long()

def merge_cif_files(file1, file2, output_file):
    def read_cif(file):
        with open(file, 'r') as f:
            lines = f.readlines()
        return lines

    def extract_atom_data(lines):
        atom_data = []
        atom_section = False
        for line in lines:
            if line.strip().startswith('loop_'):
                atom_section = True
            if atom_section:
                atom_data.append(line)
        return atom_data

    lines1 = read_cif(file1)
    lines2 = read_cif(file2)

    atom_data1 = extract_atom_data(lines1)
    atom_data2 = extract_atom_data(lines2)

    merged_atom_data = atom_data1[:6] + atom_data1[6:] + atom_data2[7:]

    with open(output_file, 'w') as f:
        f.writelines(lines1[:lines1.index(atom_data1[0])])
        f.writelines(merged_atom_data)


def cal_cif_combined(cif1, cif2, Hbond=False):
    cif1_bond_len = cal_bonds_len(cif1, Hbond = Hbond)
    cif2_bond_len = cal_bonds_len(cif2, Hbond = Hbond)
    # 合并cif1 和 cif2 
    base_path = '/'.join(cif1.split('/')[:-1])
    
    cif1_name = cif1.split('/')[-1].split('.')[0]
    cif2_name = cif2.split('/')[-1].split('.')[0]
    output_file = base_path + "/" + cif1_name + cif2_name + '_combined.cif'
    merge_cif_files(cif1, cif2, output_file)
    combined_bond_len = cal_bonds_len(output_file, Hbond = Hbond)
    os.remove(output_file)
    if cif1_bond_len + cif2_bond_len < combined_bond_len:
        return True
    else :
        return False
    
def cal_bonds_len(cif, Hbond = False):
    lattice_parameters, atom_types, frac_coords, _ = readcif(cif)
    frac_coords = torch.FloatTensor(frac_coords)
    atom_types = torch.LongTensor(atom_types)
    num_atoms = len(atom_types)
    lengths = torch.FloatTensor(lattice_parameters[:3]).view(1, -1)
    angles = torch.FloatTensor(lattice_parameters[3:]).view(1, -1)
    cell = lattice_params_to_matrix_torch(lengths, angles).squeeze()
    cart_coords = frac2cart(frac_coords, cell)
    dist_mat = compute_distance_matrix(cell, cart_coords)
    allatomtypes = [chemical_symbols[i] for i in atom_types]
    all_bonds = []
    for i, e1 in enumerate(allatomtypes[:-1]):
        for j, e2 in enumerate(allatomtypes[i + 1 :]):
            elements = set([e1, e2])
            # if MOF, use two line in the following.
            # if elements < metals:
            #     continue
            rad = COVALENT_RADII[e1] + COVALENT_RADII[e2]
            dist = dist_mat[i, i + j + 1]
            
            scale_factor = 0.9
            # probably a better way to fix these kinds of issues..
            if (set("F") < elements) and (elements & metals):
                scale_factor = 0.8
            if (set("C") < elements) and (elements & metals):
                scale_factor = 0.95
            if (
                (set("H") < elements)
                and (elements & metals)
                and (not elements & alkali)
            ):
                scale_factor = 0.75
            if (set("O") < elements) and (elements & metals):
                scale_factor = 0.85
            if (set("N") < elements) and (elements & metals):
                scale_factor = 0.82
            # fix for water particle recognition.
            if set(["O", "H"]) <= elements:
                scale_factor = 0.8
            # very specific fix for Michelle's amine appended MOF
            if set(["N", "H"]) <= elements:
                scale_factor = 0.67
            if set(["Mg", "N"]) <= elements:
                scale_factor = 0.80
            if set(["C", "H"]) <= elements:
                scale_factor = 0.80
            if set(["K"]) <= elements:
                scale_factor = 0.95
            if lanthanides & elements:
                scale_factor = 0.95
            if elements == set(["C"]):
                scale_factor = 0.85
            if dist * scale_factor < rad:  
                all_bonds.append([i, i + j + 1])
                all_bonds.append([i + j + 1, i])
    return len(all_bonds)

def read_cif_extract_block(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    start = content.find('_atom_site_occupancy')
    if start == -1:
        return None  # 没有找到相应的标签
    data_block = content[start:].split('\n')[1:]  
    return data_block, len(data_block)

def extract_atom_labels(data_block):
    atom_labels = []
    for line in data_block:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        atom_labels.append(parts[0].replace('?', ''))  
    return atom_labels

def get_Hbond_pairs(atom_labels, hs, acceptors):
    Hbond_pairs = []
    if len(hs) != len(acceptors):
        print("Error: Donor, H, Acceptor lists have different lengths.")
        return Hbond_pairs
    for i in range(len(hs)):
        if hs[i] not in atom_labels:
            continue
        index_hs = atom_labels.index(hs[i])
        index_acceptor = atom_labels.index(acceptors[i])
        Hbond_pairs.append([index_hs, index_acceptor])
        Hbond_pairs.append([index_acceptor, index_hs])
    return Hbond_pairs

def get_Hbond(cif):
    hbond_pairs = []
    donors, hs, acceptors = get_Hbond_lists(cif)
    data_block, data_len = read_cif_extract_block(cif)
    if data_block:
        atom_labels = extract_atom_labels(data_block)
        print(hs)
        print(acceptors)
        hbond_pairs = get_Hbond_pairs(atom_labels, hs, acceptors)
        if len(hbond_pairs) == 0:
            print(f"No H-bond pairs found in {cif}.")
        return hbond_pairs       
    else:
        print("No data block found in the CIF file.")
        return hbond_pairs    

def get_Hbond_lists(cif):
    donors, hs, acceptors = [], [], []
    # If there is no lis file, return an empty list directly
    lis = str(cif).replace('.cif', '.lis')
    if not os.path.exists(cif):
        print(f"No LIS file found for CIF ID {lis}.")
        return donors, hs, acceptors
    with open(lis, 'r') as file:
        content = file.read()
        # Find the data block between "H....Acceptor" and "Translation of ARU-Code to CIF and Equivalent Position Code"
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
        # If the lengths of the three lists are not equal, output cif_id and print an error message
        if len(donors) != len(acceptors):
            print('donors:', donors)
            print('hs:', hs)
            print('acceptors:', acceptors)
            print(f"Error in {cif}: Donor, H, Acceptor lists have different lengths.")
    return donors, hs, acceptors  

def compute_bonds(distance_mat, atomic_numbers, Hbond = False, cif = None):
    allatomtypes = [chemical_symbols[i] for i in atomic_numbers]
    all_bonds = []
    all_hbonds = []
    if Hbond:
        all_hbonds = get_Hbond(cif)
        print("all_hbonds:", all_hbonds)
        all_bonds.extend(all_hbonds)
            
    for i, e1 in enumerate(allatomtypes[:-1]):
        for j, e2 in enumerate(allatomtypes[i + 1 :]):
            elements = set([e1, e2])
            rad = COVALENT_RADII[e1] + COVALENT_RADII[e2]
            dist = distance_mat[i, i + j + 1]
            
            scale_factor = 0.9
            # probably a better way to fix these kinds of issues..
            if (set("F") < elements) and (elements & metals):
                scale_factor = 0.8
            if (set("C") < elements) and (elements & metals):
                scale_factor = 0.95
            if (
                (set("H") < elements)
                and (elements & metals)
                and (not elements & alkali)
            ):
                scale_factor = 0.75
            if (set("O") < elements) and (elements & metals):
                scale_factor = 0.85
            if (set("N") < elements) and (elements & metals):
                scale_factor = 0.82
            # fix for water particle recognition.
            if set(["O", "H"]) <= elements:
                scale_factor = 0.8
            # very specific fix for Michelle's amine appended MOF
            if set(["N", "H"]) <= elements:
                scale_factor = 0.67
            if set(["Mg", "N"]) <= elements:
                scale_factor = 0.80
            if set(["C", "H"]) <= elements:
                scale_factor = 0.80
            if set(["K"]) <= elements:
                scale_factor = 0.95
            if lanthanides & elements:
                scale_factor = 0.95
            if elements == set(["C"]):
                scale_factor = 0.85
            if dist * scale_factor < rad:  # and not (alkali & elements):
                all_bonds.append([i, i + j + 1])
                all_bonds.append([i + j + 1, i])
    return torch.LongTensor(all_bonds)

def get_atomic_graph(frac_coords, atomic_numbers, cell, Hbond = False, cif = None):
    """
    Input: torch_geometry.data.Data object. graph nodes must atoms.
    """
    cart_coords = frac2cart(frac_coords, cell)
    dist_mat = compute_distance_matrix(cell, cart_coords)
    edge_index = compute_bonds(dist_mat, atomic_numbers, Hbond = Hbond, cif = cif).T
    if len(edge_index) == 0:
        to_jimages = torch.FloatTensor([])
    else:
        to_jimages = compute_image_flag(
            cell, frac_coords[edge_index[0]], frac_coords[edge_index[1]]
        )
    return edge_index, to_jimages


def remap_values(remapping, x):
    index = torch.bucketize(x.ravel().contiguous(), remapping[0])
    return remapping[1][index].reshape(x.shape)


def extract_mol(frac_coords, atom_types, edge_index, to_jimages, component, mol_idx):
    num_atoms = atom_types.shape[0]
    mol_atomic_numbers = atom_types[component == mol_idx]
    node_index = torch.arange(num_atoms)[component == mol_idx].to(frac_coords.device)
    mol_frac_coords = frac_coords[node_index]
    mol_num_atoms = len(node_index)

    if len(edge_index) > 0:
        edge_mask = (
            ((edge_index[0]).view(-1, 1) == node_index)
            .any(dim=1)
            .to(mol_atomic_numbers.device)
        )
        mol_edge_index = edge_index[:, edge_mask]
        remapping = node_index, torch.arange(mol_num_atoms).to(
            mol_atomic_numbers.device
        )
        mol_edge_index = remap_values(remapping, mol_edge_index)
        mol_tojimages = to_jimages[edge_mask]
    else:
        mol_edge_index = edge_index
        mol_tojimages = to_jimages
    return mol_frac_coords, mol_edge_index, mol_atomic_numbers, mol_tojimages


def get_xyz_connected(frac_coords, edge_index, cell):
    """
    remap frac_coords to get the full piece of a connected component without wrapping.
    assumes the input is a connected graph and does BFS.
    """
    n_nodes = frac_coords.shape[0]
    assert (
        connected_components(n_nodes, edge_index) == 0
    ).all(), "input is not a connected graph."
    shifted_frac_coords = frac_coords.clone()
    visited = [0]
    counter = 0
    while len(visited) < n_nodes:
        current_node = visited[counter]
        for sender, receiver in edge_index.T.tolist():
            if sender == current_node and (int(receiver) not in visited):
                shifted_frac_coords[receiver] += compute_image_flag(
                    cell,
                    shifted_frac_coords[sender][None, :],
                    shifted_frac_coords[receiver][None, :],
                )
                visited.append(int(receiver))
        counter += 1
    return shifted_frac_coords


def process_cif_jimage(string):
    if string == ".":
        return [0, 0, 0]
    else:
        return [int(string[2]) - 5, int(string[3]) - 5, int(string[4]) - 5]


def get_digits(string):
    return int("".join([c for c in string if c.isdigit()]))


def read_cif_bonds(cif):
    with open(cif, "r") as f:
        lines = f.read().splitlines()

    # No bonds
    if sum(["_geom_bond_distance" in lin for lin in lines]) == 0:
        return [], [], []

    bond_start = (
        int(np.nonzero(["_geom_bond_distance" in lin for lin in lines])[0]) + 1
    )
    from_index = []
    to_index = []
    to_jimage = []
    for idx in range(bond_start, len(lines)):
        v1, v2, jimage = [x for x in lines[idx].split(" ") if x]
        v1 = get_digits(v1) - 1
        v2 = get_digits(v2) - 1
        jimage = process_cif_jimage(jimage)
        from_index.append(v1)
        to_index.append(v2)
        to_jimage.append(jimage)
    return from_index, to_index, to_jimage


def pyg_graph_from_cif(cif, graph_provided=False, Hbond=False):
    lattice_parameters, atom_types, frac_coords, atom_symbols = readcif(cif)
    frac_coords = torch.FloatTensor(frac_coords)
    atom_types = torch.LongTensor(atom_types)
    num_atoms = len(atom_types)
    lengths = torch.FloatTensor(lattice_parameters[:3]).view(1, -1)
    angles = torch.FloatTensor(lattice_parameters[3:]).view(1, -1)
    cell = lattice_params_to_matrix_torch(lengths, angles).squeeze()
    scaled_lattice = torch.cat(
        [lengths / float(frac_coords.shape[0]) ** (1 / 3), angles], dim=1
    )

    if graph_provided:        
        from_index, to_index, _ = read_cif_bonds(cif)
        edge_index = np.stack([from_index, to_index]).T
        if len(edge_index) > 0:
            edge_index = torch.LongTensor(edge_index).T.contiguous()
            reverse_edge_index = torch.stack([edge_index[1], edge_index[0]])
            edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
            to_jimages = compute_image_flag(
                cell, frac_coords[edge_index[0]], frac_coords[edge_index[1]]
            )
        else:
            to_jimages = torch.FloatTensor([])
    else:
        edge_index, to_jimages = get_atomic_graph(frac_coords, atom_types, cell, Hbond = Hbond, cif = cif)
    num_bonds = len(edge_index.T)

    return Data(
        frac_coords=frac_coords,
        atom_types=atom_types,
        lengths=lengths,
        angles=angles,
        edge_index=edge_index,
        to_jimages=to_jimages,
        num_atoms=num_atoms,
        num_bonds=num_bonds,
        num_nodes=num_atoms,
        cell=cell,
        scaled_lattice=scaled_lattice,
    )

# pyg_graph_from_cif('/opt/wty/hof_synthesis/HOF/MOFDiff/hofdiff/data/hof_data/48_1846107_19/48_1846107_19.cif', Hbond=True)

def get_all_bbs(data, use_cg_anchors=False):
    cell = lattice_params_to_matrix_torch(data.lengths, data.angles).squeeze()
    edge_index = data.edge_index
    all_sbbs = []
    for mol_idx, mol in enumerate(data.pyg_mols):
        mask = (data.component.view(-1, 1) == mol_idx).any(dim=1)
        node_index = mask.nonzero().view(-1)
        boundary_edge_mask = torch.logical_and(
            ((edge_index[0]).view(-1, 1) == node_index).any(dim=1),
            ((edge_index[1]).view(-1, 1) != node_index).all(dim=1),
        )
        boundary_edges = torch.unique(edge_index[:, boundary_edge_mask].T, dim=0).T
        dockers = boundary_edges[0]
        anchors = boundary_edges[1]
        anchor_atom_types = data.atom_types[anchors]
        anchor_components = data.component[anchors]

        remapping = node_index, torch.arange(len(node_index)).to(node_index.device)
        dockers_mol_index = remap_values(remapping, dockers)
        docker_frac_coords = mol.frac_coords[dockers_mol_index]
        anchor_images = compute_image_flag(
            cell, docker_frac_coords, data.frac_coords[anchors]
        )

        anchor_shifted_frac_coords = data.frac_coords[anchors] + anchor_images
        anchor_shifted_frac_coords = (
            docker_frac_coords + anchor_shifted_frac_coords
        ) / 2

        if use_cg_anchors:
            anchor_cg_images = compute_image_flag(
                cell, anchor_shifted_frac_coords, data.cg_frac_coords[anchor_components]
            )
            if len(anchor_cg_images.shape) == 1:
                anchor_cg_images = anchor_cg_images.unsqueeze(0)
            anchor_signatures = torch.cat(
                [anchor_cg_images, anchor_components.unsqueeze(1)], dim=-1
            )
            _, anchor_group = torch.unique(
                anchor_signatures, dim=0, return_inverse=True
            )
            anchor_shifted_frac_coords = scatter(
                anchor_shifted_frac_coords, anchor_group, dim=0, reduce="mean"
            )
            num_anchors = anchor_shifted_frac_coords.shape[0]
            anchor_atom_types = 2 * torch.ones(num_anchors).long().to(node_index.device)
            docking_edges = torch.stack(
                [dockers_mol_index, anchor_group + mol.num_atoms]
            )
        else:
            num_anchors = anchor_shifted_frac_coords.shape[0]
            docking_edges = torch.stack(
                [
                    dockers_mol_index,
                    torch.arange(num_anchors).to(dockers_mol_index.device)
                    + mol.num_atoms,
                ]
            )

        frac_coords = torch.cat([mol.frac_coords, anchor_shifted_frac_coords], dim=0)
        atom_types = torch.cat([mol.atom_types, anchor_atom_types])

        mol_edge_index = mol.edge_index
        if not isinstance(mol_edge_index, torch.Tensor):
            mol_edge_index = torch.LongTensor(mol_edge_index).to(atom_types.device)
        if len(docking_edges) > 0:
            
            if mol_edge_index.shape[0] > 0:
                mol_edge_index = torch.cat(
                    [
                        mol_edge_index,
                        docking_edges,
                        torch.stack([docking_edges[1], docking_edges[0]]),
                    ],
                    dim=1,
                )
            else:
                mol_edge_index = torch.cat(
                    [docking_edges, torch.stack([docking_edges[1], docking_edges[0]])],
                    dim=1,
                )
        if mol_edge_index.numel() > 0:      
            mol_edge_index = torch.unique(mol_edge_index, dim=0)

        is_anchor = torch.zeros(mol.num_atoms + num_anchors).bool()
        if num_anchors != 0:
            is_anchor[-num_anchors:] = True
        else:
            is_anchor[-num_anchors:] = False
        mol_sbb = Data(
            frac_coords=frac_coords,
            atom_types=atom_types,
            edge_index=mol_edge_index,
            lengths=data.lengths,
            angles=data.angles,
            num_atoms=atom_types.shape[0],
            cell=cell,
            is_anchor=is_anchor,
            num_cps=mol.num_cps,
        )
        mol_sbb.atom_types[mol_sbb.is_anchor] = 2
        # get fingerprint
        mol_sbb.fp = get_bb_fingerprint(mol_sbb)
        all_sbbs.append(mol_sbb)
    return all_sbbs

def assemble_local_struct(building_blocks, mof_asr=None, device='cpu'):
    """
    get a pyg data object from the MOFid decompostion.
    mof_asr: use the graph constructed by MOFid, stored in <mof_asr.cif> and loaded as a pyg object already.
    """
    for bb in building_blocks:
        bb.to(device)
    if mof_asr is not None:
        mof_asr = mof_asr.to(device)

    lengths = building_blocks[0].lengths
    angles = building_blocks[0].angles
    cell = lattice_params_to_matrix_torch(lengths, angles).squeeze()
    frac_coords = []
    atom_types = []
    component = []
    is_linker = []
    is_hbond_candidate = []
    n_components = torch.zeros(1).long().to(lengths.device)
    pyg_mols = []
    
    # extract each individual building block without the connection points.
    is_ls = [True] * len(building_blocks)
    for is_l, comp in zip(is_ls, building_blocks):
        if comp.num_nodes > 0:
            frac_coords.append(comp.frac_coords)
            atom_types.append(comp.atom_types)

            hbond_mask = (comp.atom_types == 1) | (comp.atom_types == 8) | \
                         (comp.atom_types == 16) | (comp.atom_types == 7)
            is_hbond_candidate.append(hbond_mask)

            perm = connected_components(comp.num_atoms, comp.edge_index).to(
                lengths.device
            )
            n_mols = perm.max() + 1
            is_linker.append(torch.BoolTensor([is_l] * n_mols))
            component.append(perm + n_components)
            n_components += n_mols

            # compose mols.
            for mol_idx in range(n_mols):
                (
                    mol_frac_coords,
                    mol_edge_index,
                    mol_atomic_numbers,
                    mol_tojimages,
                ) = extract_mol(
                    comp.frac_coords,
                    comp.atom_types,
                    comp.edge_index,
                    comp.to_jimages,
                    perm,
                    mol_idx,
                )

                shifted_frac_coords = get_xyz_connected(
                    mol_frac_coords, mol_edge_index, cell
                )
                if len(mol_edge_index) > 0:
                    shifted_jimages = compute_image_flag(
                        cell,
                        shifted_frac_coords[mol_edge_index[0]],
                        shifted_frac_coords[mol_edge_index[1]],
                    )
                else:
                    shifted_jimages = torch.FloatTensor([]).to(mol_frac_coords.device)

                pyg_mols.append(
                    Data(
                        frac_coords=shifted_frac_coords,
                        atom_types=mol_atomic_numbers,
                        edge_index=mol_edge_index,
                        to_jimages=shifted_jimages,
                        lengths=lengths,
                        angles=angles,
                        num_atoms=mol_atomic_numbers.shape[0],
                        cell=cell,
                    )
                )

    if len(frac_coords) == 0:
        return None

    frac_coords = torch.cat(frac_coords, dim=0)
    atom_types = torch.cat(atom_types, dim=0)
    component = torch.cat(component, dim=0)
    is_linker = torch.cat(is_linker, dim=0)
    scaled_lattice = torch.cat(
        [lengths / float(frac_coords.shape[0]) ** (1 / 3), angles], dim=1
    )

    # use mof_asr atomic graph if provided.
    if mof_asr is not None:
        sort_index = []
        for coords in mof_asr.frac_coords:
            sort_index.append(int((frac_coords == coords).all(axis=1).nonzero()[0]))
        sort_index = torch.LongTensor(sort_index).to(frac_coords.device)
        # print(mof_asr.edge_index)
        edge_index = torch.stack(
            [sort_index[mof_asr.edge_index[0]], sort_index[mof_asr.edge_index[1]]]
        )
        reverse_edge_index = torch.stack([edge_index[1], edge_index[0]])
        edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
        to_jimages = compute_image_flag(
            cell, frac_coords[edge_index[0]], frac_coords[edge_index[1]]
        )
    else:
        edge_index, to_jimages = get_atomic_graph(frac_coords, atom_types, cell)

    data = Data(
        frac_coords=frac_coords,
        atom_types=atom_types,
        lengths=lengths,
        angles=angles,
        edge_index=edge_index,
        to_jimages=to_jimages,
        num_atoms=frac_coords.shape[0],
        num_bonds=edge_index.shape[1],
        num_nodes=frac_coords.shape[0],
        scaled_lattice=scaled_lattice,
        cell=cell,
        # CG info.
        component=component,
        is_linker=is_linker,
        is_hbond_candidate=is_hbond_candidate,
        num_components=n_components,
        pyg_mols=pyg_mols,
    )

    # get centroids of connection points as the position of the cg-bead.
    for mol_idx, mol in enumerate(data.pyg_mols):
        mask = (data.component.view(-1, 1) == mol_idx).any(dim=1)
        node_index = mask.nonzero().view(-1)
        boundary_edge_mask = torch.logical_and(
            ((edge_index[0]).view(-1, 1) == node_index).any(dim=1),
            ((edge_index[1]).view(-1, 1) != node_index).all(dim=1),
        )
        dockers = edge_index[0][boundary_edge_mask]
        anchors = edge_index[1][boundary_edge_mask]

        remapping = node_index, torch.arange(len(node_index)).to(node_index.device)
        dockers_mol_index = remap_values(remapping, dockers)
        docker_frac_coords = mol.frac_coords[dockers_mol_index]
        images = compute_image_flag(cell, docker_frac_coords, data.frac_coords[anchors])

        anchor_shifted_frac_coords = data.frac_coords[anchors] + images
        anchor_shifted_frac_coords = (
            docker_frac_coords + anchor_shifted_frac_coords
        ) / 2
        anchor_shifted_frac_coords, inverse_index = torch.unique(
            anchor_shifted_frac_coords, dim=0, return_inverse=True
        )
        num_anchors = anchor_shifted_frac_coords.shape[0]

        if len(docker_frac_coords) == 0:
            if mol.frac_coords.numel() > 0:
                anchor_centroid = mol.frac_coords.mean(dim=0)
            else:
                # 如果没有任何坐标点，可以设置一个默认值，如原点
                anchor_centroid = torch.zeros(3, device=mol.frac_coords.device)
            local_vectors = torch.zeros((0, 3), device=mol.frac_coords.device)
        else:
            anchor_centroid = anchor_shifted_frac_coords.mean(dim=0)
            local_vectors = anchor_shifted_frac_coords - anchor_centroid
            anchor_centroid = anchor_centroid % 1

        mol.num_cps = num_anchors
        mol.centroid = anchor_centroid
        mol.local_vectors = local_vectors

        target_comp = []
        for i in range(num_anchors):
            target_comp.append(component[anchors][inverse_index == i][0])
        if len(target_comp) > 0:
            target_comp = torch.stack(target_comp)
            mol.target_comp = target_comp
            mol.source_comp = torch.ones(num_anchors).long().to(frac_coords.device) * mol_idx
        else:
            # Handle the case where there is no join point, such as setting a default value or skipping
            continue  
        
    data.cg_frac_coords = torch.stack([mol.centroid for mol in data.pyg_mols])
    # print("data.cg_frac_coords:", data.cg_frac_coords)
    # get edge to_jimages. This process already counts for two directions for each edge.
    cg_edge_index = []
    cg_to_jimages = []
    for mol_idx, mol in enumerate(data.pyg_mols):
        if hasattr(mol, 'source_comp') and hasattr(mol, 'target_comp') and len(mol.target_comp) > 0:
            cg_edge_index.append(torch.stack([mol.source_comp, mol.target_comp]))
            new_anchor_locs = mol.centroid + mol.local_vectors
            cg_to_jimages.append(
                compute_image_flag(
                    cell, new_anchor_locs, data.cg_frac_coords[mol.target_comp]
                )
            )
        else:
            # Handle the case of no connection and avoid adding non-existent indexes
            continue
    if cg_edge_index:
        cg_edge_index = torch.cat(cg_edge_index, dim=1)
        cg_to_jimages = [x.unsqueeze(0) if len(x.shape) == 1 else x for x in cg_to_jimages]
        cg_to_jimages = torch.cat(cg_to_jimages, dim=0)

        cg_edge_and_jimages = torch.cat([cg_edge_index.T, cg_to_jimages], dim=1)
        cg_edge_and_jimages, cg_edge_group = torch.unique(
            cg_edge_and_jimages, dim=0, return_inverse=True
        )
        connection_types = scatter(
            torch.ones_like(cg_edge_group), cg_edge_group, dim=0, reduce="sum"
        )
        cg_edge_index = cg_edge_and_jimages[:, 0:2].T
        cg_to_jimages = cg_edge_and_jimages[:, 2:]
        data.cg_edge_index = cg_edge_index
        data.cg_to_jimages = cg_to_jimages
        data.cg_edge_types = connection_types

        data.num_cg_bonds = data.cg_edge_index.shape[1]
    else:
        # 处理完全没有边的情况
        data.cg_edge_index = torch.empty((2, 0), dtype=torch.long)
        data.cg_to_jimages = torch.empty((0, 3))
        data.cg_edge_types = torch.empty(0)
        data.num_cg_bonds = 0
    # get building block objects that augment the pyg_mol objects with connection points.
    data.bbs = get_all_bbs(data)

    return data.to("cpu")


def batch_adjusted_component(batch):
    node_offsets = torch.cumsum(batch.num_atoms, -1)
    component_offsets = torch.cumsum(batch.num_components, -1)
    component = batch.component.clone()
    for idx in range(len(component_offsets) - 1):
        component[node_offsets[idx] : node_offsets[idx + 1]] += component_offsets[idx]
    component_offsets = torch.cat(
        [torch.LongTensor([0]).to(component_offsets.device), component_offsets]
    )
    return component, component_offsets


def arrange_decoded_mofs(output, kdtree, all_data, load_traj=False):
    """
    retrive builing block idneities from a kdtree to get the all-atom MOF structure from 
    a sampled CG MOF structure.
    """
    natoms = output["num_atoms"]
    # print("natoms:", natoms)
    lengths = output["lengths"]
    angles = output["angles"]

    if "x" in output:
        # print("X in output")
        cart_coords = output["x"]
        # print("cart_coords:", cart_coords)
        bb_embedding = output["h"]
        frac_coords = cart_to_frac_coords(cart_coords, lengths, angles, natoms)
        # print("frac_coords:", frac_coords)
    else:
        # print("X not in output")
        frac_coords = output["frac_coords"]
        bb_embedding = output["atom_types"]
    
    if load_traj:
        key = "all_x" if "all_x" in output else "all_noise_cart"
        if key == "all_x":
            frac_traj = []
            for i in range(len(output[key])):
                frac_traj.append(
                    cart_to_frac_coords(
                        output["all_x"][i].to(lengths.device), lengths, angles, natoms
                    )
                )
            frac_traj = torch.stack(frac_traj)
        else:
            frac_traj = output["all_noise_cart"]

    all_mofs = []
    offsets = torch.cat(
        [torch.LongTensor([0]).to(natoms.device), torch.cumsum(natoms, dim=0)]
    )
    for i in range(len(natoms)):
        cry_rec = Data(
            frac_coords=frac_coords[offsets[i] : offsets[i + 1]],
            atom_types=torch.ones(natoms[i]).long(),
            bb_embedding=bb_embedding[offsets[i] : offsets[i + 1]],
            num_atoms=natoms[i],
            num_components=natoms[i],
            lengths=output["lengths"][i, None],
            angles=output["angles"][i, None],
            edge_index=[],
            to_jimages=[],
            all_x=frac_traj[:, offsets[i] : offsets[i + 1]] if load_traj else None,
            all_h=output["all_h"][:, offsets[i] : offsets[i + 1]]
            if load_traj
            else None,
        )

        ret_bbs = []
        cry_rec.all_atom_coords = []
        cry_rec.building_block_ids = []
        cry_rec.all_atom_types = []
        for j in range(cry_rec.num_atoms):
            bb_emb = cry_rec.bb_embedding[j]
            # nan may appear during sampling and the code will bug out here if that happens.
            if bb_emb.isnan().any():
                print("bb_emb is nan")
                ret_bb = None
            else:
                ret_bb = all_data[kdtree.query(bb_emb.cpu())[1]].cpu().clone()
                cry_rec.building_block_ids.append(kdtree.query(bb_emb.cpu())[1])
                ret_bb.centroid = cry_rec.frac_coords[j].cpu()
                if ret_bb.is_anchor.sum() == 0:
                    # 没有锚点，使用几何中心替代
                    anchor_center = ret_bb.frac_coords.mean(dim=0)
                else:
                    # 有锚点，使用锚点几何中心
                    anchor_center = ret_bb.frac_coords[ret_bb.is_anchor].mean(dim=0)

                ret_bb.frac_coords = (
                    ret_bb.frac_coords
                    - anchor_center
                    + ret_bb.centroid
                )
                ret_bb.local_vectors = (
                    ret_bb.frac_coords[ret_bb.is_anchor] - ret_bb.centroid
                )
                del ret_bb.fp

                # 保存全原子坐标信息
                cry_rec.all_atom_coords.append(ret_bb.frac_coords.clone())  # 保存每个构建块的全原子坐标
                cry_rec.all_atom_types.append(ret_bb.atom_types)  # 保存原子类型

            ret_bbs.append(ret_bb)
        # print(ret_bbs)
        cry_rec.all_atom_coords = torch.cat(cry_rec.all_atom_coords, dim=0)
        cry_rec.all_atom_types = torch.cat(cry_rec.all_atom_types, dim=0)
        # cry_rec.building_block_ids = torch.cat(cry_rec.building_block_ids, dim=0)
        cry_rec.bbs = ret_bbs
        cry_rec.cell = lattice_params_to_matrix_torch(
            cry_rec.lengths, cry_rec.angles
        ).squeeze()
        cry_rec.num_components = cry_rec.num_atoms

        all_mofs.append(cry_rec)

    return all_mofs


def mof2cif_with_bonds(mof, fname):
    cellprm = torch.cat([mof.lengths, mof.angles]).numpy().flatten()
    fcoords = mof.frac_coords.numpy()
    atom_labels = [chemical_symbols[i] for i in mof.atom_types]
    edge_index = mof.edge_index.T.numpy()
    
    distances = compute_distance_matrix(
        mof.cell.squeeze(), frac2cart(mof.frac_coords, mof.cell.squeeze())
    )[edge_index[:, 0], edge_index[:, 1]].numpy()
    # print("card_coords", frac2cart(mof.frac_coords, mof.cell.squeeze()))
    to_jimages = mof.to_jimages.numpy()
    if "bond_types" in mof:
        bond_types = [INDEX2BTYPE[i] for i in mof.bond_types]
    else:
        bond_types = None
    writecif(
        fname,
        cellprm,
        fcoords,
        atom_labels,
        edge_index,
        distances,
        to_jimages,
        bond_types,
    )


def get_structure(cifpath, primitive=False):
    return CifParser(cifpath).get_structures(primitive)[0]


def get_primitive(datapath, writepath):
    s = CifParser(datapath, occupancy_tolerance=1).get_structures()[0]
    sprim = s.get_primitive_structure()
    CifWriter(sprim).write_file(str(writepath))


def graph_from_cif(cif, bond=False):
    if bond:
        from_index, to_index, to_jimage = read_cif_bonds(cif)
    else:
        lattice_parameters, atom_types, frac_coords = readcif(cif)
        frac_coords = torch.FloatTensor(frac_coords)
        atom_types = torch.LongTensor(atom_types)
        lengths = torch.FloatTensor(lattice_parameters[:3]).view(1, -1)
        angles = torch.FloatTensor(lattice_parameters[3:]).view(1, -1)
        cell = lattice_params_to_matrix_torch(lengths, angles).squeeze()
        edge_index, to_jimage = get_atomic_graph(frac_coords, atom_types, cell)
        # print("edge_index:", edge_index)
        from_index = edge_index[0,:]
        to_index = edge_index[1,:]
        # print("from_index:", from_index)
        # print("to_index:", to_index)
    try:
        parsed = CifParser(cif, occupancy_tolerance=1000)
    except ZeroDivisionError:
        return None
    structure = parsed.get_structures(primitive=False)[0]

    # find sort index through coords.
    crys_xyz = structure.frac_coords
    data = next(iter(parsed._cif.data.values()))
    data_xyz = np.concatenate(
        [
            np.array(data["_atom_site_fract_x"]).astype("float")[:, None],
            np.array(data["_atom_site_fract_y"]).astype("float")[:, None],
            np.array(data["_atom_site_fract_z"]).astype("float")[:, None],
        ],
        axis=1,
    )
    ri_x = (data_xyz[:, 0] == 1).nonzero()[0]
    ri_y = (data_xyz[:, 1] == 1).nonzero()[0]
    ri_z = (data_xyz[:, 2] == 1).nonzero()[0]
    data_xyz[data_xyz == 1] = 0

    # remap index
    sort_index = []
    for xyz in data_xyz:
        sort_index.append(int(np.isclose(crys_xyz, xyz).all(axis=1).nonzero()[0]))
    # print("sort_index:",sort_index)
    from_index = np.array(sort_index)[from_index]
    to_index = np.array(sort_index)[to_index]
    ri_x = np.array(sort_index)[ri_x]
    ri_y = np.array(sort_index)[ri_y]
    ri_z = np.array(sort_index)[ri_z]
    rounded_index = [ri_x, ri_y, ri_z]

    # return rounded_index
    for i in range(len(to_jimage)):
        for axis, ri in enumerate(rounded_index):
            if (from_index[i] in ri) != (to_index[i] in ri):
                to_jimage[i][axis] -= 1
    edges = {}
    for i in range(len(from_index)):
        if (
            from_index[i],
            to_index[i],
            (0, 0, 0),
            tuple(to_jimage[i]),
        ) not in edges.keys() and (
            to_index[i],
            from_index[i],
            (0, 0, 0),
            tuple([-x for x in to_jimage[i]]),
        ) not in edges.keys():
            edges[(from_index[i], to_index[i], (0, 0, 0), tuple(to_jimage[i]))] = {}
    # print("graph edges:",edges)
    graph = StructureGraph.with_edges(structure, edges)
    # 查看graph 的边信息
    return graph


def get_smiles(mol):
    return mol.write("smi").split("\t")[0]


def get_xyz_string(cart_coords, atomic_numbers):
    xyz_string = []
    atoms = [chemical_symbols[i] for i in atomic_numbers]
    xyz_string.append("%i\n\n" % len(atoms))
    for i, cart_coord in enumerate(cart_coords):
        s = "%10.2f %10.2f %10.2f" % (cart_coord[0], cart_coord[1], cart_coord[2])
        xyz_string.append("%s %s\n" % (atoms[i], s))
    return "".join(xyz_string)


def get_pybel_mol(frac_coords, atomic_numbers, edge_index, cell):
    """
    Input: torch_geometry.data.Data object. graph nodes must atoms.
    Return: pybel.Molecule object.
    """
    shifted_frac_coords = get_xyz_connected(frac_coords, edge_index, cell)
    cart_coords = frac2cart(shifted_frac_coords, cell)
    mol = pybel.readstring("xyz", get_xyz_string(cart_coords, atomic_numbers))
    return mol


def pygmol2pybel(mol):
    if "cell" in mol:
        cell = mol.cell
    else:
        cell = lattice_params_to_matrix_torch(mol.lengths, mol.angles).squeeze()
    return get_pybel_mol(mol.frac_coords, mol.atom_types, mol.edge_index, cell)


def pymatgenmol2pybel(molgraph):
    return BabelMolAdaptor.from_molecule_graph(molgraph).pybel_mol


def get_mol_xyz_w_bond(sbb):
    sbb_pybel = pygmol2pybel(sbb)
    anchors = sbb.is_anchor.nonzero().flatten()
    bonds = []
    for bond in ob.OBMolBondIter(sbb_pybel.OBMol):
        if bond.IsAromatic():
            btype = "A"
        elif bond.GetBondOrder() == 1:
            btype = "S"
        elif bond.GetBondOrder() == 2:
            btype = "D"
        elif bond.GetBondOrder() == 3:
            btype = "T"
        else:
            raise NotImplementedError
        begin, end = bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1
        if begin in anchors and end in anchors:
            continue
        bonds.append(f"   {begin}    {end} {btype}")
    sbb.atom_types[sbb.is_anchor] = 118
    sbb_pybel = pygmol2pybel(sbb)
    xyz = sbb_pybel.write("xyz").replace("Og", "X ")
    return xyz + "\n".join(bonds)


def to_onehot(bits):
    x = torch.zeros(4096, dtype=torch.bool)
    x[bits] = 1
    return x


def pyg2molecule(data):
    if "cart_coords" in data:
        cart_coords = data.cart_coords
    else:
        lattice_matrix = lattice_params_to_matrix_torch(data.lengths, data.angles)
        cart_coords = frac2cart(data.frac_coords, lattice_matrix.squeeze())
    cart_coords = cart_coords - cart_coords.mean(dim=0)
    atom_types = data.atom_types
    sites = []

    num_atoms = cart_coords.shape[0]
    for i in range(num_atoms):
        symbol = chemical_symbols[atom_types[i]]
        sites.append(
            {
                "species": [{"element": symbol, "occu": 1.0}],
                "xyz": cart_coords[i].tolist(),
                "properties": {},
            }
        )
    pymatgen_dict = {
        "@module": "pymatgen.core.structure",
        "@class": "Molecule",
        "charge": 0.0,
        "spin_multiplicity": 0,
        "sites": sites,
    }
    mol = Molecule.from_dict(pymatgen_dict)
    edge_index = data.edge_index
    if len(edge_index) > 0:
        remove_dir_mask = edge_index[0] <= edge_index[1]
        edge_index = data.edge_index[:, remove_dir_mask]
        edges = {
            (int(edge_index[0, i]), int(edge_index[1, i])): {}
            for i in range(edge_index.shape[1])
        }
    else:
        edges = {}
    graph = MoleculeGraph.with_edges(mol, edges)
    return graph


def get_bb_fingerprint(bb):
    return to_onehot(
        [x - 1 for x in pymatgenmol2pybel(pyg2molecule(bb)).calcfp("ecfp4").bits]
    )
