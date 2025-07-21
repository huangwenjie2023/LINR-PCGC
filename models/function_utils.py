from copy import deepcopy
import os
import numpy as np
import torch
import MinkowskiEngine as ME
from typing import List
import MinkowskiEngine as ME
import torch
from sort_functions import *

pruning = ME.MinkowskiPruning()

def generate_sparse(xyz, feature, tensor_stride=1):
    xyz_, feature_ = xyz.int(), feature.float()
    frame_C, frame_F = ME.utils.sparse_collate([xyz_], [feature_])
    frame_data = ME.SparseTensor(features=frame_F, coordinates=frame_C,tensor_stride=tensor_stride, device=xyz.device)
    assert (frame_data.C[:, 1:] != xyz).sum() == 0
    return frame_data


def unique_by_reval_number_np(coords, return_idx=False, step=None, min_c=None, max_c=None):
    ravel_num = get_reval_number_np(coords, step, min_c, max_c)
    # 获取unique的索引
    _, idx = np.unique(ravel_num, return_index=True)
    coords_new = coords[idx]
    if return_idx:
        return coords_new, idx
    else:
        return coords_new

def get_target_by_sp_tensor(out, target_sp_tensor):
    with torch.no_grad():
        def ravel_multi_index(coords, step):
            coords = coords.long()
            step = step.long()
            coords_sum = coords[:, 3] \
                         + coords[:, 2] * step \
                         + coords[:, 1] * step * step \
                         + coords[:, 0] * step * step * step
            return coords_sum

        step = max(out.C.max(), target_sp_tensor.C.max()) + 1

        out_sp_tensor_coords_1d = ravel_multi_index(out.C, step)
        in_sp_tensor_coords_1d = ravel_multi_index(target_sp_tensor.C, step)

        # test whether each element of a 1-D array is also present in a second array.
        target = torch.isin(out_sp_tensor_coords_1d, in_sp_tensor_coords_1d)

    return target


def prune_by_sp_tensor(out: ME.SparseTensor, target_sp_tensor: ME.SparseTensor):
    mask = get_target_by_sp_tensor(out, target_sp_tensor)
    pruned_out = pruning(out, mask)
    return pruned_out

def merge_two_frames(f1: ME.SparseTensor, f2: ME.SparseTensor, stride=None):
    if stride is None:
        stride = f1.tensor_stride
    f1_shape = f1.F.shape
    f2_shape = f2.F.shape
    f1_ = ME.SparseTensor(torch.cat([f1.F, torch.zeros((f1_shape[0],f2_shape[1]),device=f1.device)], dim=-1), coordinates=f1.C,
                        device=f1.device)
    f2_ = ME.SparseTensor(torch.cat([torch.zeros((f2_shape[0],f1_shape[1]),device=f1.device), f2.F], dim=-1), coordinates=f2.C,
                          coordinate_manager=f1_.coordinate_manager, device=f1.device)
    merged_f = f1_ + f2_
    merged_f = ME.SparseTensor(merged_f.F, coordinates=merged_f.C,coordinate_manager=f1.coordinate_manager, tensor_stride=stride, device=merged_f.device)
    return merged_f

def keep_adaptive(out: ME.SparseTensor, coords_nums: List[int], rho: float = 1.0):
    """
    :param out:带有多个batch的元素
    :param coords_nums:是一个列表，代表着不同batch中需要保留的数量，比如coords_nums[i]代表着第i个batch中需要保留的点数的基数
    :param rho: 需要保留基数乘以因子，获取最终需要保留的点数k
    :return: 后续修剪需要的蒙版
    对多个batch同时操作，对于第i个batch只选取前coords_num[i]*rho个元素，获取修剪蒙版，为后面修剪提供条件
    """
    with torch.no_grad():
        keep = torch.zeros(len(out), dtype=torch.bool, device=out.device)
        #  get row indices per batch.
        # row_indices_per_batch = out.coords_man.get_row_indices_per_batch(out.coordinate_map_key)
        row_indices_per_batch = out._batchwise_row_indices

        for row_indices, ori_coords_num in zip(row_indices_per_batch, coords_nums):
            coords_num = min(len(row_indices), ori_coords_num * rho)  # select top k points.
            values, indices = torch.topk(out.F[row_indices].squeeze(), int(coords_num))
            keep[row_indices[indices]] = True
    return keep


def set_tensor_stride(x: ME.SparseTensor, stride):
    return ME.SparseTensor(features=x.F, coordinates=x.C, tensor_stride=stride, coordinate_manager=x.coordinate_manager, device=x.device)


def reset_tensor_stride(f: ME.SparseTensor):
    stride = f.tensor_stride[0]
    coord = f.C
    ret_coord = deepcopy(coord)
    ret_coord[:,1:] = coord[:,1:] // stride
    new_f = ME.SparseTensor(f.F, coordinates=ret_coord, tensor_stride=1, coordinate_manager=f.coordinate_manager, device=f.device)
    return new_f

def set_feature_value(x: ME.SparseTensor, value):
    feature = torch.ones((x.F.shape[0],1),device=x.device,dtype=torch.float32)*value
    return ME.SparseTensor(features=feature,coordinates=x.C,coordinate_manager=x.coordinate_manager, tensor_stride=x.tensor_stride,device=x.device)


def pack_bitstream(bitstream_list, dtype='uint32'):
    bitstream_all = np.array(len(bitstream_list), dtype=dtype).tobytes()
    bitstream_all += np.array([len(bitstream) for bitstream in bitstream_list], dtype=dtype).tobytes()
    for bitstream in bitstream_list:
        assert len(bitstream) < 2 ** 32 - 1
        bitstream_all += bitstream

    return bitstream_all


def unpack_bitstream(bitstream_all, dtype='uint32'):
    s = 0
    num = np.frombuffer(bitstream_all[s: s + 1 * 4], dtype=dtype)[0]
    s += 1 * 4
    lengths = np.frombuffer(bitstream_all[s:s + num * 4], dtype=dtype)
    s += num * 4
    # print('DBG!!!', num, lengths)
    bitstream_list = []
    for l in lengths:
        bitstream = bitstream_all[s: s + l]
        bitstream_list.append(bitstream)
        s += l

    return bitstream_list


def ae_pack_bitstream(shape, min_v, max_v, strings, dtype='int32'):
    bitstream = np.array(shape, dtype=dtype).tobytes()
    bitstream += np.array(min_v, dtype=dtype).tobytes()
    bitstream += np.array(max_v, dtype=dtype).tobytes()
    bitstream += strings

    return bitstream


def ae_unpack_bitstream(bitstream, dtype='int32'):
    s = 0
    shape = np.frombuffer(bitstream[s:s+2*4], dtype=dtype)
    s += 2*4
    min_v = np.frombuffer(bitstream[s:s+1*4], dtype=dtype)
    s += 1*4
    max_v = np.frombuffer(bitstream[s:s+1*4], dtype=dtype)
    s += 1*4
    strings = bitstream[s:]

    return shape, min_v, max_v, strings

def copy_sparse_tensor(x: ME.SparseTensor):
    return ME.SparseTensor(features=x.F, coordinates=x.C, tensor_stride=x.tensor_stride, device=x.device)



def quantize_sparse_tensor(x, factor, quant_mode='floor'):
    # assert factor <= 1
    if factor==1: return x

    coords = x.C[:,1:]
    precise = 1 / factor
    coordsQ = coords / precise
    if quant_mode=='round': coordsQ = torch.round(coordsQ)
    elif quant_mode=='floor': coordsQ = torch.floor(coordsQ)
    coordsQ = coordsQ.int()
    new_coord = torch.cat([x.C[:,0].unsqueeze(1), coordsQ], dim=1)
    out = ME.SparseTensor(features=x.F, coordinates=new_coord, tensor_stride=x.tensor_stride, device=x.device) 
    return out

def num2bits(coords, dtype='int16'):
    min_v = coords.min(axis=0)
    coords = coords - min_v
    bitstream = np.array(min_v, dtype=dtype).tobytes()
    bitstream += coords.tobytes()
    return bitstream

def bit2num(bitstream, dtype='int16'):
    s = 0
    min_v = np.frombuffer(bitstream[s:s+3*2], dtype=dtype).reshape(-1, 3)
    s += 3*2
    coords = np.frombuffer(bitstream[s:], dtype=dtype).reshape(-1, 3)
    coords = coords + min_v
    return coords




def array2vector(array, step):
    """ravel 2D array with multi-channel to one 1D vector by sum each channel with different step.
    """
    array, step = array.long(), step.long()
    vector = sum([array[:,i]*(step**i) for i in range(array.shape[-1])])

    return vector

def isin(data, ground_truth):
    """ Input data and ground_truth are torch tensor of shape [N, D].
    Returns a boolean vector of the same length as `data` that is True
    where an element of `data` is in `ground_truth` and False otherwise.
    """
    device = data.device
    if len(ground_truth)==0:
        return torch.zeros([len(data)]).bool().to(device)
    step = torch.max(data.max(), ground_truth.max()) + 1
    data = array2vector(data, step)
    ground_truth = array2vector(ground_truth, step)
    mask = torch.isin(data.to(device), ground_truth.to(device))

    return mask

def istopk(data, nums, rho=1.0):
    """ Input data is sparse tensor and nums is a list of shape [batch_size].
    Returns a boolean vector of the same length as `data` that is True
    where an element of `data` is the top k (=nums*rho) value and False otherwise.
    """
    mask = torch.zeros(len(data), dtype=torch.bool)
    row_indices_per_batch = data._batchwise_row_indices
    for row_indices, N in zip(row_indices_per_batch, nums):
        k = int(min(len(row_indices), N*rho))
        _, indices = torch.topk(data.F[row_indices].squeeze().detach().cpu(), k)# must CPU.
        mask[row_indices[indices]]=True

    return mask.bool().to(data.device)

def create_new_sparse_tensor(coordinates, features, tensor_stride, dimension, device):
    sparse_tensor = ME.SparseTensor(features=features, 
                                coordinates=coordinates,
                                tensor_stride=tensor_stride,
                                device=device)
    # manager = ME.CoordinateManager(D=dimension)
    # key, _ = manager.insert_and_map(coordinates.to(device), tensor_stride)
    # sparse_tensor = ME.SparseTensor(features=features, 
    #                                 coordinate_map_key=key, 
    #                                 coordinate_manager=manager, 
    #                                 device=device)

    return sparse_tensor




###########################3
def istopk_new(data, nums):
    """
    """
    # local: top-k (k=1) in each 8-voxel set
    prob = torch.sigmoid(data.F)
    mask = torch.zeros(len(prob), dtype=torch.bool)
    _, indices1 = torch.topk(prob.reshape(-1, 8), 1)
    indices1 += (torch.arange(0, len(indices1)) * 8).reshape(-1, 1).to(indices1.device)
    indices1 = indices1.reshape(-1)
    mask[indices1] = True
    prob[torch.where(mask)[0]] = 1
    # global: top-k (k=num) in all voxels
    _, indices2 = torch.topk(prob.squeeze(), nums[1])
    mask[indices2] = True

    return mask.bool().to(data.device)


def istopk_local(data, k=1):
    """input data is probability
        select top-k voxels in each 8-voxels set
    """
    mask = torch.zeros(len(data), dtype=torch.bool)
    _, indices = torch.topk(data.reshape(-1, 8), k)
    indices += (torch.arange(0, len(indices)) * 8).reshape(-1, 1).to(indices.device)
    indices = indices.reshape(-1)
    mask[indices] = True

    return mask.bool().to(data.device)


def istopk_global(data, k):
    """input data is probability
        select top-k voxel in all voxels
    """
    mask = torch.zeros(len(data), dtype=torch.bool)
    _, indices = torch.topk(data.squeeze(), k)
    mask[indices] = True

    return mask.bool().to(data.device)
