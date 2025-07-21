import MinkowskiEngine as ME
import torch
import numpy as np
def sort_by_coor_sum(xyz, feature, return_coor_sum=False):
    maximum = xyz.max() + 1
    xyz, maximum = xyz.long(), maximum.long()
    coor_sum = xyz[:, 0] * maximum * maximum \
               + xyz[:, 1] * maximum \
               + xyz[:, 2]
    coor_sum_, idx = coor_sum.sort()
    xyz_, feature_ = xyz[idx], feature[idx]
    if return_coor_sum:
        return xyz_, feature_, coor_sum_
    else:
        return xyz_, feature_
    
def sort_by_coord_sum_c(xyz,onlycoord=True):
    maximum = xyz.max() + 1
    minimum = xyz.min() - 1
    step = maximum - minimum
    xyz_T = xyz - minimum
    xyz_T, step = xyz_T.long(), step.long()
    coor_sum = xyz_T[:, 0] * (step * step) \
               + xyz_T[:, 1] * step \
               + xyz_T[:, 2]
    coor_sum_, idx = coor_sum.sort()
    xyz_ = xyz[idx]
    if onlycoord:
        return xyz_
    return xyz_, coor_sum_, minimum, step

def sort_by_coord_sum_c_with_feat(xyz,feature):
    maximum = xyz.max() + 1
    minimum = xyz.min() - 1
    step = maximum - minimum
    xyz_T = xyz - minimum
    xyz_T, step = xyz_T.long(), step.long()
    coor_sum = xyz_T[:, 0] * (step * step) \
               + xyz_T[:, 1] * step \
               + xyz_T[:, 2]
    coor_sum_, idx = coor_sum.sort()
    xyz_ = xyz[idx]
    feature_ = feature[idx]
    return xyz_, feature_

def sort_by_coor_sum_detail(xyz, feature):
    maximum = xyz.max() + 1
    minimum = xyz.min() - 1
    step = maximum - minimum
    
    xyz_T = xyz - minimum
    
    xyz_T, step = xyz_T.long(), step.long()
    coor_sum = xyz_T[:, 0] * (step * step) \
               + xyz_T[:, 1] * step \
               + xyz_T[:, 2]
    coor_sum_, idx = coor_sum.sort()
    xyz_, feature_ = xyz[idx], feature[idx]

    return xyz_, feature_, coor_sum_, minimum, step




def get_reval_number_np(coords, step=None, min_c=None, max_c=None):
    if step is None or min_c is None or max_c is None:
        min_c = np.min(coords)
        max_c = np.max(coords)
        step = max_c - min_c + 1
    dim = coords.shape[1]
    coords = coords - min_c
    coords = coords.astype('int64')
    step = step.astype('int64')
    coords_sum = 0
    for i in range(dim):
        coords_sum = coords_sum * step + coords[:, i]
    
    return coords_sum

def get_reval_number(coords, step=None, min_c=None, max_c=None):
    if step is None or min_c is None or max_c is None:
        min_c = coords.min()
        max_c = coords.max()
        step = max_c - min_c + 1
    
    coords = coords - min_c
    coords = coords.long()
    step = step.long()
    dim = coords.shape[1]
    coords_sum = 0
    for i in range(dim):
        coords_sum = coords_sum * step + coords[:, i]
    return coords_sum

def sort_sparse_tensor(f:ME.SparseTensor, stride=None):
    if stride is None:
        stride = f.tensor_stride[0]
    xyz, feature = f.C, f.F
    coor_sum = get_reval_number(xyz)
    _, idx = coor_sum.sort()
    xyz_, feature_ = xyz[idx].to(torch.int32), feature[idx]
    f_ = ME.SparseTensor(feature_, coordinates=xyz_, tensor_stride=stride, device=f.device)
    return f_


def sort_coord(coords, return_idx=False):
    reval_num = get_reval_number_np(coords)
    idx = reval_num.argsort()
    if return_idx:
        return coords[idx], idx
    else:
        return coords[idx]