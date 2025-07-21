import torchac

import torch
from torch import nn
import MinkowskiEngine as ME
from function_utils import *
from quantize_functions import *
class BinaryArithmeticCoding():
    """arithmetic coding for occupancy status.
    """
    def _get_cdf(self, prob):
        zeros = torch.zeros(prob.shape, dtype=prob.dtype, device=prob.device)
        ones = torch.ones(prob.shape, dtype=prob.dtype, device=prob.device)
        cdf = torch.cat([zeros, 1-prob, ones], dim=-1)

        return cdf

    def estimate_bitrate(self, prob, occupancy):
        pmf = torch.cat([1-prob, prob], axis=-1)
        prob_true = pmf[torch.arange(0, len(occupancy)).tolist(), occupancy.tolist()]
        entropy = -torch.log2(prob_true)
        bits = torch.sum(entropy).tolist()

        return bits

    def encode(self, prob, occupancy):
        cdf = self._get_cdf(prob)
        bitstream = torchac.encode_float_cdf(cdf,occupancy)
        # with open(filename+'.bin', 'wb') as fout:
        #     fout.write(bitstream)

        return bitstream

    def decode(self, prob, bitstream):
        cdf = self._get_cdf(prob)
        # with open(filename+'.bin', 'rb') as fin:
        #     bitstream =fin.read()
        occupancy = torchac.decode_float_cdf(cdf, bitstream)

        return occupancy

def get_and_init_FC_layer(din, dout, init_bias='zeros'):
    """Get a fully-connected layer."""

    li = nn.Linear(din, dout)
    #init weights/bias
    # nn.init.normal_(convnet.weight, mean=0.0, std=0.05)
    nn.init.xavier_uniform_(li.weight.data, gain=nn.init.calculate_gain('relu'))
    # nn.init.normal_(li.weight, mean=0.0, std=0.05)
    if init_bias == 'uniform':
        nn.init.uniform_(li.bias)
    elif init_bias == 'zeros':
        li.bias.data.fill_(0.)
    else:
        raise 'Unknown init ' + init_bias
    return li


def get_MLP_layers(dims, doLastRelu, init_bias='zeros'):
    """Get a series of MLP layers."""

    layers = []
    for i in range(1, len(dims)):
        layers.append(get_and_init_FC_layer(dims[i-1], dims[i], init_bias=init_bias))
        if i==len(dims)-1 and not doLastRelu:
            continue
        layers.append(nn.ReLU())
    return layers


class PointwiseMLP(nn.Sequential):
    """PointwiseMLP layers.

    Args:
        dims: dimensions of the channels
        doLastRelu: do the last Relu (nonlinear activation) or not.
        Nxdin ->Nxd1->Nxd2->...-> Nxdout
    """
    def __init__(self, dims, doLastRelu=False, init_bias='zeros'):
        layers = get_MLP_layers(dims, doLastRelu, init_bias)
        super(PointwiseMLP, self).__init__(*layers)




class octree_level(nn.Module):
    def __init__(self):
        super(octree_level, self).__init__()
        # self.ref_conv = ME.MinkowskiConvolution(in_channels=1, out_channels=1, kernel_size=2, stride=2,
        #                                      bias=True,
        #                                      dimension=3)
        self.ref_conv = lambda x: quantize(x, 2)
        self.offsets = torch.tensor([[i, j, k] for i in range(2) for j in range(2) for k in range(2)], dtype=torch.float32).cuda()

        
        
    def forward(self, leaf, qsc):
        
        N, C = leaf.size()
        device = leaf.device
        # leaf_ = torch.ones(N, 1, dtype=torch.float32, device=device)
        
        parent_C = self.ref_conv(leaf)
        # parent_C = parent.C
        N_p = parent_C.size()[0]
        occupancy = torch.zeros(N_p, 8, dtype=torch.float32, device=device)
        for i in range(8):
            offset_parent = parent_C.to(torch.float32) * 2
            offset_parent += self.offsets[i] 
            occupancy[:, i:i+1] = qsc.search(offset_parent)
            
        # recon_coord = self.upper_layer(parent_C, occupancy)
        # assert (recon_coord != leaf).sum() == 0
        # level = ME.SparseTensor(occupancy, coordinate_map_key=parent.coordinate_map_key, coordinate_manager=parent.coordinate_manager)
        return parent_C, occupancy

    def upper_layer(self, parent_C, occupancy):
        children = []
        # leaf_stride = stride // 2
        parent_C = dequatize(parent_C, 2)
        for i in range(8):
            offset_parent = parent_C[occupancy[:, i] == 1]
            offset_parent += self.offsets[i].to(torch.int32)
            children.append(offset_parent)
        children = torch.cat(children, dim=0)
        children = sort_by_coord_sum_c(children)
        return children
    
    def upper_layer_with_feature(self, parent_C, occupancy, feature):
        children = []
        # leaf_stride = stride // 2
        parent_C = dequatize(parent_C, 2)
        
        children_features_lst = []
        
        
        
        for i in range(8):
            derive_idx = occupancy[:, i] == 1
            
            offset_parent = parent_C[derive_idx]
            offset_parent += self.offsets[i].to(torch.int32)
            children.append(offset_parent)
            
            ch_feat_all = feature[:,i]
            children_features_lst.append(ch_feat_all[derive_idx])
            
        children = torch.cat(children, dim=0)
        children_features = torch.cat(children_features_lst, dim=0)
        
        children, children_features = sort_by_coord_sum_c_with_feat(children,children_features)
        return children, children_features.view(-1,1)

octree_level_obj = octree_level()
class qscTensor:
    def __init__(self, coord, feat=None):
        self.qsc = QuickSearchCoord(coord, feat)
        self.coord = self.qsc.coord
        self.feat = self.qsc.feat
        
        
        self.parent_C = None
        self.occupancy = None
        self.offset_sp = None
        self.offset_tensor = None
        self.neigbor_idxs = None

        

    def get_coord(self):
        return self.coord
    
    def search(self, coord_in):
        return self.qsc.search(coord_in)
    
    def search_coord_idx(self, coord_in):
        return self.qsc.search_coord_idx(coord_in)
    
    
    def get_sparse(self):
        return generate_sparse(self.coord, self.feat)
    
    
    def set_oct_level(self):
        self.parent_C, self.occupancy = octree_level_obj(self.coord, self.qsc)
    
    def get_oct_level(self):
        return self.parent_C, self.occupancy
    
    
    def upper_layer(self, parent, occupancy):
        return octree_level_obj.upper_layer(parent, occupancy)
    
    
    def upper_layer_with_feature(self, parent, occupancy, feature):
        return octree_level_obj.upper_layer_with_feature(parent, occupancy, feature)
    
    
    def set_offset_sparse(self, offsets):
        # 在10_0中废弃
        features = [self.search(self.coord + offset) for offset in offsets]
        feat = torch.concat(features, dim=-1)
        coord = self.coord
        out_sp = generate_sparse(coord, feat)
        self.offset_sp = out_sp
    
    def get_offset_sparse(self):
        return self.offset_sp   

    def set_offset_tensor(self, offsets):
        features = [self.search(self.coord + offset) for offset in offsets]
        feat = torch.concat(features, dim=-1)
        self.offset_tensor = feat
        
    def get_offset_tensor(self):
        return self.offset_tensor
    
    def set_neigber_idxs(self, offset_of_neigbor):
        neigbor_idxs = [self.search_coord_idx(self.coord + offset) for offset in offset_of_neigbor]
        neigbor_idxs = torch.stack(neigbor_idxs, dim=-1)
        self.neigbor_idxs = neigbor_idxs
        
    def get_neigbor_idxs(self):
        return self.neigbor_idxs

class Linear(nn.Module):
    def __init__(self, input, output, activation='relu'):
        super(Linear, self).__init__()
        self.linear = nn.Sequential(nn.Linear(input, output))
        if activation is None:
            pass
        elif activation == 'relu':
            self.linear.add_module('activation', nn.ReLU(inplace=True))
        elif activation == 'lrelu':
            self.linear.add_module('activation', nn.LeakyReLU(inplace=True))
        elif activation == 'sigmoid':
            self.linear.add_module('activation', nn.Sigmoid())
        elif activation == 'softmax':
            self.linear.add_module('activation', nn.Softmax(dim=-2))
        elif activation == 'tanh':
            self.linear.add_module('activation', nn.Tanh())

    def forward(self, x):
        return self.linear(x)
    
class QuickSearchCoord:
    def __init__(self, coord, feat=None):
        coord = torch.unique(coord, dim=0)
        if feat is None:
            feat = torch.ones((coord.shape[0], 1), dtype=torch.float32, device=coord.device)
        coord, feat, coor_sum, minimum, step = sort_by_coor_sum_detail(coord, feat)
        self.coord = coord
        self.feat = feat
        self.coor_sum = coor_sum
        self.minimum = minimum
        self.step = step
        
        # self.feat_col = feat.shape[1]
        
    def search(self, coord):
        coord = coord - self.minimum
        
        # 减去最小值后，如果有负数，说明该行坐标不在范围内
        idx_invalid = (coord < 0).any(dim=1)
        
        coord = coord.long()
        coor_sum_in = coord[:, 0] * (self.step * self.step) \
                   + coord[:, 1] * self.step \
                   + coord[:, 2]
                   
        sear_idx = self.search_idx_(coor_sum_in)
        idx_invalid = idx_invalid | (sear_idx == -1)
        got_feat = self.feat[sear_idx]
        got_feat[idx_invalid,:] = 0
        return got_feat
        
    def search_coord_idx(self, coord):
        coord = coord - self.minimum
        
        coord = coord.long()
        coor_sum_in = coord[:, 0] * (self.step * self.step) \
                   + coord[:, 1] * self.step \
                   + coord[:, 2]
                   
        sear_idx = self.search_idx_(coor_sum_in)
        return sear_idx
    def search_idx_(self, coord):
        sorted_tensor = self.coor_sum
        values = coord
        # sorted_tensor = torch.tensor([1, 3, 5, 7, 9])

        # # 要查找的值
        # values = torch.tensor([0, 3, 6, 7, 10])

        # 使用 searchsorted 查找插入点
        indices = torch.searchsorted(sorted_tensor, values)

        # 用最朴素的方法查找values在不在sorted_tensor中
        
        # exist_ret = sum(indices == -1)
        # exist_ret_ori = (values in sorted_tensor).sum()
        # exist_ret_ori = torch.isin(values, sorted_tensor).sum()
        
        
        # 创建一个与 values 相同大小的输出张量，初始化为 -1
        result_indices = torch.full_like(values, -1)

        # 确保 indices 在有效范围内
        valid_indices = indices[indices < len(sorted_tensor)]

        # 仅在有效索引中检查相等
        result_indices[indices < len(sorted_tensor)] = torch.where(sorted_tensor[valid_indices] == values[indices < len(sorted_tensor)], valid_indices, -1)

        # exist_ret = (result_indices != -1).sum()
        # assert exist_ret == exist_ret_ori
        # 输出结果
        # print(result_indices)  # 输出：tensor([-1,  1, -1,  3, -1])
        return result_indices
    
    