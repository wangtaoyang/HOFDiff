"""
Assembly algorithm.
"""
import numpy as np
import torch
import random

from scipy.optimize import minimize, linear_sum_assignment
from torch_geometric.data import Data
from openbabel import openbabel as ob
from ase.data import covalent_radii

from mofdiff.common.atomic_utils import (
    compute_distance_matrix,
    compute_distance_matrix_offset,
    frac2cart,
    cart2frac,
    remap_values,
    compute_image_flag,
)
from mofdiff.common.constants import METALS
from mofdiff.common.data_utils import lattice_params_to_matrix_torch
from mofdiff.common.so3 import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    random_rotations,
)
from mofdiff.common.atomic_utils import pygmol2pybel

covalent_radii_tensor = torch.tensor(covalent_radii)
metal_atomic_numbers = torch.tensor(METALS)
non_metal_atomic_numbers = torch.tensor([x for x in np.arange(100) if x not in METALS])


def same_component_mask(component):
    """mask=1 for all pairs that belong to the same component."""
    mask = (component.unsqueeze(0) == component.unsqueeze(1)).to(torch.bool)
    return mask

def get_optimized_hbond_mask(hbond_potential_mask, dist_mat, num_hbonds_to_optimize):
    # 找到所有潜在氢键的实际索引
    hbond_indices = hbond_potential_mask.nonzero(as_tuple=True)
    
    # 从距离矩阵中获取所有潜在氢键的距离
    hbond_distances = dist_mat[hbond_indices]
    
    # 找到最短的氢键距离的索引
    _, top_indices = torch.topk(hbond_distances, num_hbonds_to_optimize, largest=False, sorted=True)
    
    # 获取对应的实际索引
    optimized_indices = (hbond_indices[0][top_indices], hbond_indices[1][top_indices])
    
    # 创建一个新的优化氢键掩码
    optimized_hbond_mask = torch.zeros_like(hbond_potential_mask, dtype=torch.bool)
    # print("optimized_indices:", optimized_indices)
    optimized_hbond_mask[optimized_indices] = True
    return optimized_hbond_mask


def hbond_loss_per_block(
    dist_mat,
    image_offsets,
    hbond_mask,
    building_block_mask,
    min_distance_threshold=2.4,
    target_distance=2.6,
    max_distance_threshold=3.5,  # 最大氢键距离阈值
    k=1,  # 每对 building block 考虑的氢键对数量，默认为 None
    weight=1.0
):
    """
    氢键距离损失函数：每对 building block 之间考虑 k 个氢键对，
    并且如果氢键对的最短距离大于 max_distance_threshold，则不计算该对氢键的损失。
    并确保每个 building block 至少与一个其他 building block 形成氢键。
    """
    unique_blocks = torch.unique(building_block_mask)  # 所有 building block 的唯一标识
    device = dist_mat.device
    hbond_quality_loss = torch.zeros(1, dtype=torch.float64, device=device, requires_grad=True)
    connectivity_loss = torch.zeros(1, dtype=torch.float64, device=device, requires_grad=True)
    target_distance = torch.tensor(target_distance, dtype=torch.float64, device=device)
    # print("unique_blocks", unique_blocks)
    # 遍历每个 building block
    for block_id in unique_blocks:
        # 当前 block 的原子掩码
        current_block_mask = building_block_mask == block_id
        # print("current_block_mask", current_block_mask)
        current_indices = torch.nonzero(current_block_mask).view(-1)  # 当前 block 的原子索引

        # 非当前 block 的原子掩码
        other_blocks_mask = ~current_block_mask
        other_indices = torch.nonzero(other_blocks_mask).view(-1)

        current_hbond_mask_other = hbond_mask[current_block_mask][:, other_blocks_mask]
        current_dist_mat_other = dist_mat[current_block_mask][:, other_blocks_mask]

        # 计算潜在氢键距离
        potential_hbond_distances_other = current_dist_mat_other[current_hbond_mask_other]
        # print(current_dist_mat_other.shape)
        # print(current_hbond_mask_other)
        # 同一 block 的氢键掩码（与自身交互）
        # print("image_offsets:", image_offsets)
        current_image_offsets_self = image_offsets[current_block_mask][:, current_block_mask]
        current_hbond_mask_self = hbond_mask[current_block_mask][:, current_block_mask]
        current_dist_mat_self = dist_mat[current_block_mask][:, current_block_mask]

        non_zero_offsets_mask = torch.any(current_image_offsets_self != 0, dim=-1)  # 找到跨周期的偏移
        valid_hbond_mask_self = current_hbond_mask_self & non_zero_offsets_mask  # 同时满足氢键和跨周期条件
        potential_hbond_distances_self = current_dist_mat_self[valid_hbond_mask_self]
        # 筛选出potential_hbond_distances_self 中距离大于
        # 合并所有氢键距离，这段代码有问题，假如考虑到自连氢键，需要考虑到共价键和氢键的区别
        potential_hbond_distances = torch.cat(
            [potential_hbond_distances_other, potential_hbond_distances_self], dim=0
        )
        # potential_hbond_distances = potential_hbond_distances_other


        # 输出对应的距离和行列位置
        # for i, dist in enumerate(potential_hbond_distances_other):
        #     # print(rows_in_dist_mat)
        #     # print(cols_in_dist_mat)
        #     row = rows_in_dist_mat[i]
        #     col = cols_in_dist_mat[i]
        #     # print(f"Distance: {dist}, Row: {row}, Col: {col}")

        # 过滤掉大于最大距离阈值的氢键对
        potential_hbond_distances = potential_hbond_distances_other[
            (potential_hbond_distances_other >= min_distance_threshold)
            & (potential_hbond_distances_other <= max_distance_threshold)
        ]
        if potential_hbond_distances.numel() > 0:
            hbond_quality_loss = hbond_quality_loss +  ((potential_hbond_distances - target_distance) ** 2).sum()
            print("hbond_quality_loss:", hbond_quality_loss)
        # 检查是否有至少一个氢键与其他 block 连接（增加惩罚项）
        # print("potential_hbond_distances", potential_hbond_distances)
        if potential_hbond_distances.numel() == 0:
            block_size = torch.tensor(current_block_mask.sum().item(), dtype=torch.float64, device=device)
            connectivity_loss = connectivity_loss + block_size  # 使用张量更新
            print("connectivity_loss:", connectivity_loss)

    # 总损失是氢键损失和连接性损失的加权和
    total_loss = weight * hbond_quality_loss + connectivity_loss
 
    return total_loss


# def gaussian_ball_overlap_loss(
#     cart_coords,
#     all_atom_rotated_coords,
#     all_atom_types,
#     building_block_mask,
#     component,
#     cell,
#     sigma=10.0,
#     max_neighbors=30,
#     cp_atom_types=None,
#     type_mask=True,
#     num_hbonds_to_optimize=1,  # 每个 building block 优化的氢键数量
#     safe_distance=0.8,
#     buffer_distance=0.5,
#     hbond_target_distance=2.8,
#     hbond_weight=1.0
# ):
#     """
#     损失函数：考虑晶体周期性，计算高斯球重叠损失和氢键距离损失。
#     """

#     # 计算全局距离矩阵，考虑周期性边界条件 (PBC)
#     # print("all_atom_rotated_coords:", all_atom_rotated_coords)
#     frac_coords = cart2frac(all_atom_rotated_coords, cell) % 1
#     all_dist_mat_pbc, image_offsets = compute_distance_matrix_offset(cell.squeeze(), frac2cart(frac_coords, cell.squeeze()))
#     building_block_dist_mask = building_block_mask.view(-1, 1) != building_block_mask.view(1, -1)
#     # 处理本周期和其他周期的 inter-block 距离
#     inter_block_distances_pbc = all_dist_mat_pbc[building_block_dist_mask]
#     # print(inter_block_distances_pbc)
#     # 1. 距离安全性惩罚：避免距离小于 (safe_distance + buffer_distance)
#     # safe_threshold = safe_distance + buffer_distance
#     # distance_loss = torch.relu(safe_threshold - inter_block_distances_pbc).pow(2).sum()

#     # # 2. 全局分布平滑：鼓励合理的分布（包括周期性镜像）
#     # smoothness_loss = -torch.exp(-(all_dist_mat_pbc - sigma).pow(2) / sigma**2).sum()

#     # 3. 计算氢键损失
#     all_h_donor = all_atom_types == 1  # 判断是否为氢键供体
#     all_h_acceptor = (all_atom_types == 8) | (all_atom_types == 7) | (all_atom_types == 16)  # 判断是否为受体
#     all_hbond_potential_mask = (
#         (all_h_donor.view(-1, 1) & all_h_acceptor.view(1, -1))
#         | (all_h_acceptor.view(-1, 1) & all_h_donor.view(1, -1))
#     )

#     hbond_loss = hbond_loss_per_block(
#         dist_mat=all_dist_mat_pbc,
#         image_offsets=image_offsets,
#         hbond_mask=all_hbond_potential_mask,
#         building_block_mask=building_block_mask,
#         target_distance=hbond_target_distance,
#         k=num_hbonds_to_optimize,
#         weight=hbond_weight
#     )

#     # 总损失权重
#     alpha = 0.1  # 权重因子
#     beta = 0.1   # 权重因子
#     # total_loss = alpha * distance_loss + smoothness_loss * beta + hbond_loss
#     total_loss = hbond_loss
#     # total_loss = alpha * distance_loss
#     # print("distance_loss:", distance_loss * alpha)
#     # print("smoothness_loss:", smoothness_loss * beta)
#     # print("hbond_loss:", hbond_loss)
#     return total_loss


# def gaussian_ball_overlap_loss(
#     cart_coords,
#     component,
#     cell,
#     sigma=10.0,
#     max_neighbors=30,
#     cp_atom_types=None,
#     type_mask=True,
# ):
#     """loss for gaussian ball overlap at connection points."""
#     n = cart_coords.shape[0]
#     dist_mat = compute_distance_matrix(cell, cart_coords).squeeze()

#     comp_mask = same_component_mask(component)
#     dist_mat[comp_mask] = 1000.0

#     if type_mask:
#         assert cp_atom_types is not None
#         # metal, metal mask -- bond cannot be formed between two metal atoms
#         # non-metal, non-metal mask -- bond cannot be formed between two non-metal atoms
#         cp_atom_is_metal = (cp_atom_types.view(-1, 1) == metal_atomic_numbers).any(
#             dim=-1
#         )
#         cp_atom_is_not_metal = (
#             cp_atom_types.view(-1, 1) == non_metal_atomic_numbers
#         ).any(dim=-1)
#         metal_mask = torch.logical_and(
#             cp_atom_is_metal.view(-1, 1), cp_atom_is_metal.view(1, -1)
#         )
#         non_metal_mask = torch.logical_and(
#             cp_atom_is_not_metal.view(-1, 1), cp_atom_is_not_metal.view(1, -1)
#         )
#         dist_mat[metal_mask] = 1000.0
#         dist_mat[non_metal_mask] = 1000.0

#     sorted_dist_mat = dist_mat.sort(dim=-1)[0]
#     kNN_same_mask = (sorted_dist_mat >= 1000.0)[:, : min(max_neighbors, n - 1)]

#     kNN_dists = sorted_dist_mat[:, : min(max_neighbors, n - 1)][~kNN_same_mask]
#     return -torch.exp(-kNN_dists / sigma).sum() / n



def get_cp_coords(vecs, cg_frac_coords, all_atom_coords, bb_local_vectors, cell, building_block_mask, bbs_cells, bbs_centers, remove_cg_coords=True, scale_factor=1.0):
    # print("vecs:", vecs)
    vecs = vecs.view(-1, 3)
    cg_frac_coords *= scale_factor
    # print("cg_frac_coords:",cg_frac_coords)
    r = axis_angle_to_matrix(vecs).to(cg_frac_coords.dtype).to(cg_frac_coords.device)
    # print("r:", r)
    frac_coords = cg_frac_coords
    num_atoms = cg_frac_coords.shape[0]
    num_nodes = num_atoms

    all_atom_rotated_coords = []  # 存储旋转后的全原子坐标

    # print("local_vectors:", bb_local_vectors)
    for i, local_vectors in enumerate(bb_local_vectors):
        # print("local_vectors:", local_vectors)

        local_vectors_r = local_vectors @ r[i]
        cp_frac_coords = cg_frac_coords[i] + local_vectors_r
        frac_coords = torch.cat([frac_coords, cp_frac_coords % 1], dim=0)

        # 旋转全原子坐标
        mask_indices = (building_block_mask == i)  # 获取当前 building block 的索引
        if mask_indices.any():
            current_block_coords = all_atom_coords[mask_indices]  # 当前 building block 的原子坐标
            current_block_coords = frac2cart(current_block_coords, bbs_cells[i].double())
            # print(f"i: {i}, current_block_coords: {current_block_coords}, r[i]: {r[i]}")
            rotated_coords = (current_block_coords - bbs_centers[i]) @ r[i]
            rotated_coords += bbs_centers[i]
            all_atom_rotated_coords.append(rotated_coords)
            
        num_cps = cp_frac_coords.shape[0]
        num_nodes = num_nodes + num_cps

    cart_coords = frac2cart(frac_coords, cell)
    # print("cart_coords:", cart_coords)
    atom_node = torch.zeros(num_nodes, dtype=torch.long, device=r.device)
    atom_node[:num_atoms] = 1

    if remove_cg_coords:
        frac_coords = frac_coords[~atom_node.bool()]
        cart_coords = cart_coords[~atom_node.bool()]

    # 将所有旋转后的原子坐标合并
    if all_atom_rotated_coords:
        all_atom_rotated_coords = torch.cat(all_atom_rotated_coords, dim=0)

    # print("all_atom_rotated_coords:", all_atom_rotated_coords.shape)
    return cart_coords, frac_coords, all_atom_rotated_coords


def fun(x, arg_dict):
    """
    Params:
        optimizable vars:
          vecs: num_bb x 3
          frac_coords: num_bb x 3
          lengths: 1 x 3
          angles: 1 x 3
        bb info:
          bb_local_vectors: List[Tensor], length=num_bb
          bb_atom_local_vectors: List[Tensor], length=num_bb
          bb_atom_types: List[Tensor], length=num_bb
          connecting_atom_index: Tensor, length=num_atoms
        opt params:
          sigma: gaussian overlap loss ball radius
          max_neighbor: maximum number of nearby connection points to compute for overlapping volume
    """

    # gather variables from arg_dict.
    vecs = arg_dict["vecs"]
    cg_frac_coords = arg_dict["cg_frac_coords"]
    lengths = arg_dict["lengths"]
    angles = arg_dict["angles"]
    bb_local_vectors = arg_dict["bb_local_vectors"]
    cp_components = arg_dict["cp_components"]
    connecting_atom_types = arg_dict["connecting_atom_types"]
    bbs_cells = arg_dict["bbs_cells"]

    building_block_mask = arg_dict["building_block_mask"]
    all_atom_coords = arg_dict["all_atom_coords"]
    all_atom_types = arg_dict["all_atom_types"]
    bbs_centers = arg_dict["bbs_centers"]

    sigma = arg_dict.get("sigma", 1.0)
    max_neighbors = arg_dict.get("max_neighbors", 30)

    N_bb = len(bb_local_vectors)
    vecs = x.view(N_bb, 3)

    cell = lattice_params_to_matrix_torch(
        lengths.view(1, -1), angles.view(1, -1)
    ).squeeze()
    cp_cart_coords, _, all_atom_rotated_coords  = get_cp_coords(vecs, cg_frac_coords, all_atom_coords, bb_local_vectors, cell, building_block_mask, bbs_cells, bbs_centers)
    cp_loss = gaussian_ball_overlap_loss(
        cp_cart_coords, all_atom_rotated_coords, all_atom_types, building_block_mask, cp_components, cell, sigma, max_neighbors, connecting_atom_types
    ).view(-1)

    return cp_loss


def grad_fun(x, arg_dict):
    grad = torch.autograd.grad(sum(fun(x, arg_dict)), x)[0]
    return grad


def fun_apply(x, arg_dict):
    return fun(torch.tensor(x, dtype=torch.float64), arg_dict).detach().cpu().numpy()


def grad_apply(x, arg_dict):
    return (
        grad_fun(torch.tensor(x, requires_grad=True, dtype=torch.float64), arg_dict)
        .detach()
        .numpy()
        .flatten()
    )


def layout_optimization(
    x0, args, bounds=None, return_traj=False, maxiter=100, tol=None
):
    x_traj = []
    f_traj = []

    def callback(x):
        x = torch.tensor(x)
        x_traj.append(x)
        cp_loss = fun(x, args)
        f_traj.append(cp_loss)

    result = minimize(
        x0=x0,
        args=args,
        fun=fun_apply,
        jac=grad_apply,
        method="L-BFGS-B",
        bounds=bounds,
        callback=callback if return_traj else None,
        tol=tol,
        options={"maxiter": maxiter, "disp": False},
    )

    result = dict(result)
    if return_traj:
        result.update({"x_traj": x_traj, "f_traj": f_traj})

    return result


def prepare_optimization_variables(mof, device='cuda'):
    # fix these keys...
    # print("mof:", mof)
    device = mof.num_atoms.device if device is None else device
    cg_frac_coords = mof.cg_frac_coords if "cg_frac_coords" in mof else mof.frac_coords
    key = "pyg_mols" if "pyg_mols" in mof else "bbs"
    # for bb in mof['bb_embedding']:
    #     print("bb:",bb)
    bb_local_vectors = [bb.local_vectors.to(device).double() for bb in mof[key]]
    bb_atom_types = [bb.atom_types.to(device).double() for bb in mof[key]]

    cp_components = torch.cat(
        [
            torch.ones(bb_local_vectors[i].shape[0]) * i
            for i in range(len(bb_local_vectors))
        ],
        dim=0,
    ).long()

    atom_types = []
    atom_components = []
    building_block_mask = []
    atom_count = 0  # 用于跟踪当前原子的总数

    for i in range(len(bb_atom_types)):
        atom_types.append(bb_atom_types[i])
        atom_components.append(i * torch.ones(bb_atom_types[i].shape[0]).long())

         # 生成 mask
        current_count = bb_atom_types[i].shape[0]
        building_block_mask.append(torch.full((current_count,), i, dtype=torch.long))  # 创建当前 block 的 mask
        atom_count += current_count

    atom_types = torch.cat(atom_types, dim=0).long()
    atom_components = torch.cat(atom_components, dim=0).long()

    # 生成最终的 building block mask
    building_block_mask = torch.cat(building_block_mask, dim=0)
    # print("building_block_mask:", building_block_mask)
    all_atoms_types = mof.all_atom_types.to(device).long()
    all_atom_coords = mof.all_atom_coords.to(device).double()

    connecting_atom_index = get_connecting_atom_index(mof[key])
    connecting_atom_types = atom_types[connecting_atom_index]
    # print("cp_components:", cp_components)
    bbs_cells = torch.stack([bb.cell for bb in mof.bbs]).double()
    bbs_centers = torch.stack([bb.centroid for bb in mof.bbs]).double()

    return {
        "cg_frac_coords": cg_frac_coords.to(device).double(),
        "lengths": mof.lengths.to(device).double(),
        "angles": mof.angles.to(device).double(),
        "bbs_cells": bbs_cells.to(device),
        "bb_local_vectors": bb_local_vectors,
        "cp_components": cp_components,
        "connecting_atom_types": connecting_atom_types,
        "building_block_mask": building_block_mask,  # 返回 mask
        'all_atom_types': all_atoms_types,
        'all_atom_coords': all_atom_coords,
        'bbs_centers': bbs_centers.to(device)
    }


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def annealed_optimization(
    mof,
    seed,
    optimize=True,
    sigma_schedule=np.linspace(3, 0.6, 10),
    max_neighbors_schedule=np.arange(25, 5, -2),
    return_traj=False,
    maxiter=100,
    tol=None,
    print_freq=1,
    verbose=False,
):
    set_seed(seed)
    N_bb = totaln = (
        mof.num_components.sum() if "pyg_mols" in mof else mof.num_atoms.sum()
    )
    vec_dim = N_bb * 3
    # init BB rotations
    r = random_rotations(totaln, device=mof.num_atoms.device).transpose(-1, -2)
    vecs = matrix_to_axis_angle(r).double()

    # get initial x and args
    args = prepare_optimization_variables(mof)
    args.update({"vecs": vecs})
    x0 = vecs.flatten().numpy()
    bounds = [(-100.0, 100.0)] * (N_bb * 3)

    if not optimize:
        results = {"x": x0, "total_iters": 0}
        v = 0  # or some other value representing the initial state
        return results, v

    print("in annealed_optimization")
    x_traj = []
    n_steps = len(sigma_schedule)
    total_iters = 0
    for i in range(n_steps):
        args.update(
            {
                "vecs": vecs,
                "sigma": sigma_schedule[i],
                "max_neighbors": max_neighbors_schedule[i],
            }
        )

        results = layout_optimization(
            x0, args, bounds=bounds, return_traj=return_traj, maxiter=maxiter, tol=tol
        )
        x0 = results["x"]
        args["vecs"] = torch.from_numpy(x0[:vec_dim]).view(N_bb, 3)
        v = results["fun"]
        n_iter = results["nit"]
        sigma = args["sigma"]
        maxn = args["max_neighbors"]
        total_iters += n_iter

        if verbose and (i % print_freq == 0 or i == n_steps - 1):
            print(
                f"[{i+1}/{n_steps}] total iter: {total_iters}, sigma: {sigma:.2f}, max_neighbors: {maxn:.2f},"
                f"v: {v:.4f}"
            )
        if return_traj:
            x_traj.extend(results["x_traj"])

    results["total_iters"] = total_iters
    if return_traj:
        results["x_traj"] = x_traj

    return results, v


# def match_cps(
#     vecs,
#     cg_frac_coords,
#     bb_local_vectors,
#     lengths,
#     angles,
#     cp_atom_types=None,
#     type_mask=True,
# ):
#     cell = lattice_params_to_matrix_torch(
#         lengths.view(1, -1), angles.view(1, -1)
#     ).squeeze()
#     cart_coords, _ = get_cp_coords(vecs, cg_frac_coords, bb_local_vectors, cell)
#     dist_mat = compute_distance_matrix(cell, cart_coords).squeeze()

#     cp_components = torch.cat(
#         [
#             torch.ones(bb_local_vectors[i].shape[0]) * i
#             for i in range(len(bb_local_vectors))
#         ],
#         dim=0,
#     ).long()
#     mask = same_component_mask(cp_components)
#     dist_mat[mask] = 1000.0

#     if type_mask:
#         assert cp_atom_types is not None
#         # metal, metal mask -- bond cannot be formed between two metal atoms
#         # non-metal, non-metal mask -- bond cannot be formed between two non-metal atoms
#         cp_atom_is_metal = (cp_atom_types.view(-1, 1) == metal_atomic_numbers).any(
#             dim=-1
#         )
#         cp_atom_is_not_metal = (
#             cp_atom_types.view(-1, 1) == non_metal_atomic_numbers
#         ).any(dim=-1)
#         metal_mask = torch.logical_and(
#             cp_atom_is_metal.view(-1, 1), cp_atom_is_metal.view(1, -1)
#         )
#         non_metal_mask = torch.logical_and(
#             cp_atom_is_not_metal.view(-1, 1), cp_atom_is_not_metal.view(1, -1)
#         )
#         dist_mat[metal_mask] = 1000.0
#         dist_mat[non_metal_mask] = 1000.0

#     row, col = linear_sum_assignment(dist_mat.cpu().numpy())
#     cost = dist_mat[row, col]
#     return {"row": torch.from_numpy(row), "col": torch.from_numpy(col), "cost": cost}

# def match_cps(
#     vecs,
#     cg_frac_coords,
#     bb_local_vectors,
#     lengths,
#     angles,
#     cp_atom_types=None,
#     type_mask=True,
# ):
#     cell = lattice_params_to_matrix_torch(
#         lengths.view(1, -1), angles.view(1, -1)
#     ).squeeze()
#     cart_coords, _ = get_cp_coords(vecs, cg_frac_coords, bb_local_vectors, cell)
#     dist_mat = compute_distance_matrix(cell, cart_coords).squeeze()

#     cp_components = torch.cat(
#         [
#             torch.ones(bb_local_vectors[i].shape[0]) * i
#             for i in range(len(bb_local_vectors))
#         ],
#         dim=0,
#     ).long()
#     mask = same_component_mask(cp_components)
#     dist_mat[mask] = 1000.0

#     if type_mask:
#         assert cp_atom_types is not None
#         # metal, metal mask -- bond cannot be formed between two metal atoms
#         # non-metal, non-metal mask -- bond cannot be formed between two non-metal atoms
#         cp_atom_is_not_h = ~(cp_atom_types.view(-1, 1) == torch.tensor([1])).any(dim=-1)
#         cp_atom_is_not_hbond = ~(
#             cp_atom_types.view(-1, 1) == torch.tensor([7, 8, 9, 15, 16, 17, 35, 53])
#         ).any(dim=-1)
#         # hbond mask
#         hbond_mask = torch.logical_and((cp_atom_is_not_h).view(-1, 1), (cp_atom_is_not_hbond).view(1, -1))
#         dist_mat[hbond_mask] = 1000.0

#     row, col = linear_sum_assignment(dist_mat.cpu().numpy())
#     cost = dist_mat[row, col]
#     return {"row": torch.from_numpy(row), "col": torch.from_numpy(col), "cost": cost}

def get_mol_bonds(sbb):
    bonds = []
    bond_types = []
    # print("sbb:", sbb)
    sbb_pybel = pygmol2pybel(sbb)
    mol = sbb_pybel.OBMol
    for bond in ob.OBMolBondIter(mol):
        if bond.IsAromatic():
            btype = 0
        elif bond.GetBondOrder() == 1:
            btype = 1
        elif bond.GetBondOrder() == 2:
            btype = 2
        elif bond.GetBondOrder() == 3:
            btype = 3
        else:
            raise NotImplementedError
        begin, end = bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1
        bonds.append([begin, end])
        bond_types.append(btype)
    bonds = torch.tensor(bonds).T
    bond_types = torch.tensor(bond_types).long()
    return bonds, bond_types


def get_bb_bond_and_types(bb):
    new_bb = bb.clone()
    edge_index = new_bb.edge_index
    atom_index = (~bb.is_anchor).nonzero().flatten()
    anchor_index = bb.is_anchor.nonzero().T
    anchor_s_mask = (edge_index[0].view(-1, 1) == anchor_index).any(dim=1)
    anchor_t_mask = (edge_index[1].view(-1, 1) == anchor_index).any(dim=1)
    anchor_e_mask = torch.logical_or(anchor_s_mask, anchor_t_mask)
    edge_index = edge_index[:, ~anchor_e_mask]
    remapping = atom_index, torch.arange(len(atom_index))
    edge_index = remap_values(remapping, edge_index)
    edge_index = torch.unique(edge_index, dim=1)
    new_bb.edge_index = edge_index
    new_bb.frac_coords = bb.frac_coords[atom_index]
    new_bb.atom_types = bb.atom_types[atom_index]
    new_bb.num_nodes = len(new_bb.atom_types)
    new_bb.is_anchor = torch.zeros(new_bb.num_nodes).bool()

    # only apply get_mol_bonds to organic building blocks.
    # for metal nodes, use original MOFid bonds and bond type single.
    has_metal = len(np.intersect1d(bb.atom_types.numpy(), METALS)) > 0
    # print("edge_index:", edge_index)
    # print("len edge_index:", len(edge_index))
    if edge_index.numel() == 0:
        edge_index = torch.FloatTensor([])
        bond_types = torch.FloatTensor([])
    elif has_metal:
        bond_types = torch.ones(len(edge_index[0])).long()
    else:
        edge_index, bond_types = get_mol_bonds(new_bb)

    if edge_index.numel() == 0:
        edge_index = torch.FloatTensor([])
        bond_types = torch.FloatTensor([])
    else:
        remapping = torch.arange(len(atom_index)), atom_index
        edge_index = remap_values(remapping, edge_index)
        rev_edge_index = torch.stack([edge_index[1], edge_index[0]])
        edge_index = torch.cat([edge_index, rev_edge_index], dim=1)
        bond_types = bond_types.repeat(2)
    return anchor_e_mask, edge_index, bond_types


def get_unique_and_index(x, dim=0):
    # https://github.com/pytorch/pytorch/issues/36748
    unique, inverse, counts = torch.unique(
        x, dim=dim, sorted=True, return_inverse=True, return_counts=True
    )
    decimals = torch.arange(inverse.numel(), device=inverse.device) / inverse.numel()
    inv_sorted = (inverse + decimals).argsort()
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    index = inv_sorted[tot_counts]
    index = index.sort().values
    return unique, index


def get_connecting_atom_index(bbs):
    all_connecting_atoms = []
    offset = 0
    for bb in bbs:
        # relying on: cp_index is sorted, edges are double-directed
        cp_index = (bb.atom_types == 2).nonzero().flatten()
        connecting_atom_index = (
            bb.edge_index[1, (bb.edge_index[0].view(-1, 1) == cp_index).any(dim=-1)]
            + offset
        )
        offset += bb.num_atoms
        all_connecting_atoms.append(connecting_atom_index)
    all_connecting_atoms = torch.cat(all_connecting_atoms)
    return all_connecting_atoms

def assemble_mof(
    cg_mof,
    vecs=None,
    bb_local_vectors=None,
    bbs=None,
):
    if vecs is not None:
        final_R = axis_angle_to_matrix(vecs)
        cg_mof.R = final_R.float()
    # print("cg_mof:", cg_mof)
    # print("cg_mof.all_atom_coords:", cg_mof.all_atom_coords)
    cell = lattice_params_to_matrix_torch(cg_mof.lengths, cg_mof.angles).squeeze()
    device = cell.device
    cg_cart_coords = frac2cart(cg_mof.frac_coords, cell)
    # print("cg_cart_coords:", cg_cart_coords)
    bbs = bbs if bbs is not None else cg_mof.bbs
    bbs = [bb.to(device) for bb in bbs]

    cart_coords, atom_types, component, edge_index, to_jimages = [], [], [], [], []
    natom_offset = 0
    natom = cg_mof.num_components if "num_components" in cg_mof else cg_mof.num_atoms
    bond_types = []
    
    # print("all_atom_coords:", cg_mof.all_atom_coords)
    # 遍历每个 building block，生成 cartesian 坐标、原子类型、边等
    for i in range(natom):
        bb = bbs[i]
        # print("bb centroid:", bb.centroid)
        bb_cart_coords = frac2cart(bb.frac_coords, bb.cell)
        # print("bb_cart_coords:", bb_cart_coords)
        bb_center = bb.centroid

        # 以bb_center 为中心旋转坐标
        bb_cart_coords = (bb_cart_coords  - bb_center) @ cg_mof.R[i]  # 应用旋转矩阵
        bb_cart_coords += bb_center

        cart_coords.append(bb_cart_coords)
        # print("bb.atom_types:", bb.atom_types)
        atom_types.append(bb.atom_types)
        component.append(torch.ones(bb_cart_coords.shape[0]) * i)

        anchor_e_mask, bb_edge_index, bb_bond_types = get_bb_bond_and_types(bb)
        anchor_edges = bb.edge_index[:, anchor_e_mask]
        bb_edge_index = torch.cat([anchor_edges, bb_edge_index], dim=1)
        bb_bond_types = torch.cat([torch.ones(anchor_e_mask.sum()), bb_bond_types])
        edge_index.append(bb_edge_index + natom_offset)
        bond_types.append(bb_bond_types)
        natom_offset += bb_cart_coords.shape[0]


    # 将所有原子坐标、类型和边连接信息拼接在一起
    cart_coords = torch.cat(cart_coords, dim=0)
    atom_types = torch.cat(atom_types, dim=0)
    frac_coords = cart2frac(cart_coords, cell) % 1  # 转换为分数坐标
    atom_node = atom_types != 2
    frac_coords = frac_coords[atom_node]
    atom_types = atom_types[atom_node]
    num_atoms = atom_node.sum()
    atom_index = atom_node.nonzero().flatten()

    # resolving edges after removing CPs
    edge_index = torch.cat(edge_index, dim=1)
    bond_types = torch.cat(bond_types, dim=0)
    mask_keep_edges = (atom_node[edge_index[0]] & atom_node[edge_index[1]])
    edge_index = edge_index[:, mask_keep_edges]

    remapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(atom_index)}
    edge_index = torch.tensor(
        [[remapping[idx.item()] for idx in row] for row in edge_index],
        dtype=torch.long,
    )

    # 获取唯一的边连接索引，并更新 bond_types
    edge_index, unique_index = get_unique_and_index(edge_index, dim=1)
    bond_types = bond_types[unique_index].long()

    # 计算图的 j-images
    to_jimages = compute_image_flag(
        cell, frac_coords[edge_index[0]], frac_coords[edge_index[1]]
    )

    # 返回组装好的 Data 对象
    return Data(
        frac_coords=frac_coords,
        atom_types=atom_types,
        num_atoms=num_atoms,
        cell=cell,
        lengths=cg_mof.lengths,
        angles=cg_mof.angles,
        edge_index=edge_index,
        to_jimages=to_jimages,
        bond_types=bond_types,
    )



# def assemble_mof(
#     cg_mof,
#     vecs=None,
#     bb_local_vectors=None,
#     bbs=None,
# ):
#     if vecs is not None:
#         final_R = axis_angle_to_matrix(vecs)
#         cg_mof.R = final_R.float()

#     cell = lattice_params_to_matrix_torch(cg_mof.lengths, cg_mof.angles).squeeze()
#     device = cell.device
#     cg_cart_coords = frac2cart(cg_mof.frac_coords, cell)
#     bbs = bbs if bbs is not None else cg_mof.bbs
#     bbs = [bb.to(device) for bb in bbs]

#     cart_coords, atom_types, component, edge_index, to_jimages = [], [], [], [], []
#     natom_offset = 0
#     natom = cg_mof.num_components if "num_components" in cg_mof else cg_mof.num_atoms
#     bond_types = []
#     for i in range(natom):
#         bb = bbs[i]
#         bb_cart_coords = frac2cart(bb.frac_coords, bb.cell)
#         bb_cart_coords = bb_cart_coords - bb_cart_coords[bb.is_anchor].mean(dim=0)
#         bb_cart_coords = bb_cart_coords @ cg_mof.R[i]
#         bb_cart_coords = cg_cart_coords[i] + bb_cart_coords

#         cart_coords.append(bb_cart_coords)
#         atom_types.append(bb.atom_types)
#         component.append(torch.ones(bb_cart_coords.shape[0]) * i)

#         anchor_e_mask, bb_edge_index, bb_bond_types = get_bb_bond_and_types(bb)
#         anchor_edges = bb.edge_index[:, anchor_e_mask]
#         bb_edge_index = torch.cat([anchor_edges, bb_edge_index], dim=1)
#         bb_bond_types = torch.cat([torch.ones(anchor_e_mask.sum()), bb_bond_types])
#         edge_index.append(bb_edge_index + natom_offset)
#         bond_types.append(bb_bond_types)
#         natom_offset += bb_cart_coords.shape[0]

#     # remove CPs
#     cart_coords = torch.cat(cart_coords, dim=0)
#     atom_types = torch.cat(atom_types, dim=0)
#     frac_coords = cart2frac(cart_coords, cell) % 1
#     atom_node = atom_types != 2
#     num_atoms = atom_node.sum()
#     atom_index = atom_node.nonzero().flatten()
#     connecting_atom_index = get_connecting_atom_index(bbs)
#     cp_atom_types = atom_types[connecting_atom_index]

#     # cp_frac_coords = frac_coords[~atom_node]
#     frac_coords = frac_coords[atom_node]
#     atom_types = atom_types[atom_node]

#     # resolving edges after removing CPs
#     edge_index = torch.cat(edge_index, dim=1)
#     bond_types = torch.cat(bond_types, dim=0)
#     is_anchors = torch.cat([bb.is_anchor for bb in bbs])
#     anchor_index = is_anchors.nonzero().T
#     anchor_s_mask = (edge_index[0].view(-1, 1) == anchor_index).any(dim=1)
#     anchor_neighs = torch.unique(edge_index[:, anchor_s_mask], dim=1)[1]
#     cp_match = match_cps(
#         vecs,
#         cg_mof.frac_coords,
#         bb_local_vectors,
#         cg_mof.lengths,
#         cg_mof.angles,
#         cp_atom_types,
#     )
#     row, col = cp_match["row"], cp_match["col"]
#     inter_BB_edges = torch.cat(
#         [
#             torch.stack([anchor_neighs[row], anchor_neighs[col]]),
#             torch.stack([anchor_neighs[col], anchor_neighs[row]]),
#         ],
#         dim=1,
#     )
#     anchor_t_mask = (edge_index[1].view(-1, 1) == anchor_index).any(dim=1)
#     anchor_e_mask = torch.logical_or(anchor_s_mask, anchor_t_mask)
#     edge_index = edge_index[:, ~anchor_e_mask]
#     bond_types = bond_types[~anchor_e_mask]

#     edge_index = torch.cat([edge_index, inter_BB_edges], dim=1)
#     bond_types = torch.cat(
#         [bond_types, torch.ones(inter_BB_edges.shape[1])], dim=0
#     ).long()
#     remapping = atom_index, torch.arange(num_atoms)
#     edge_index = remap_values(remapping, edge_index)
#     edge_index, unique_index = get_unique_and_index(edge_index, dim=1)
#     bond_types = bond_types[unique_index]
#     to_jimages = compute_image_flag(
#         cell, frac_coords[edge_index[0]], frac_coords[edge_index[1]]
#     )

#     return Data(
#         frac_coords=frac_coords,
#         atom_types=atom_types,
#         num_atoms=num_atoms,
#         cell=cell,
#         lengths=cg_mof.lengths,
#         angles=cg_mof.angles,
#         edge_index=edge_index,
#         to_jimages=to_jimages,
#         bond_types=bond_types,
#     )


def feasibility_check(cg_mof):
    """
    check the matched connection criterion.
    """
    if cg_mof is not None and cg_mof.bbs[0] is not None:
        atom_types = torch.cat([x.atom_types for x in cg_mof.bbs])
        connecting_atom_index = get_connecting_atom_index(cg_mof.bbs)
        connecting_atom_types = atom_types[connecting_atom_index]
        n_metal = (
            (connecting_atom_types.view(-1, 1) == metal_atomic_numbers)
            .any(dim=-1)
            .sum()
        )
        n_nonmetal = (
            (connecting_atom_types.view(-1, 1) == non_metal_atomic_numbers)
            .any(dim=-1)
            .sum()
        )
        return n_metal == n_nonmetal
    else:
        return False
