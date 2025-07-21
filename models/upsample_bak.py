import torch
import MinkowskiEngine as ME
import numpy as np
from resnet import ResNetBlock
import torchac
from torch import nn
from torch.nn import functional as F
from module_utils import PointwiseMLP
from function_utils import *
from module_utils import BinaryArithmeticCoding

BAC = BinaryArithmeticCoding()
class ConvWithPrune(torch.nn.Module):
    def __init__(self,in_channels, out_channels,
                kernel_size, stride=1, bias=True, dimension=3):
        super().__init__()
        self.conv = ME.MinkowskiConvolution(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, bias=bias, dimension=dimension)
    def forward(self, x, coord):
        out = self.conv(x, coord)
        out = set_tensor_stride(out, x.tensor_stride)
        return out
def get_stage_lst(stage):
    if stage == 8:
        stage_list = [[0], [1], [2], [3], [4], [5], [6], [7]]
    elif stage==4:
        stage_list = [[0,1],[2,3],[4,5],[6,7]]
    elif stage == 3:
        stage_list = [[0, 1], [6, 7], [2, 3, 4, 5]]
    elif stage == 2:
        stage_list = [[0, 1, 6, 7], [2, 3, 4, 5]]
    elif stage == 1:
        stage_list = [[0, 1, 2, 3, 4, 5, 6, 7]]
    return stage_list
############################################################
class CNP(torch.nn.Module):  ### upsampling is in block_in
    def __init__(self, in_channels=1, channels=12, kernel_size=3, block_layers=2, outstage=8, instage=1):
        super().__init__()
        self.outstage = outstage
        self.instage = instage
        
        self.block_in = self.make_block(in_channels=in_channels, channels=channels, out_channels=channels, kernel_size=kernel_size, block_layers=1)
        outstage_lst = get_stage_lst(outstage)
        
        _channels = [len(outstage_lst[out_idx]) for out_idx in range(outstage)]
        cum_channels = np.cumsum(_channels)
        
        self.inner_mlps = nn.ModuleList(
            [
                nn.ModuleList([
                    PointwiseMLP([channels, 24, _channels[out_idx]])
                for in_idx in range(instage)])
            for out_idx in range(outstage)]
        )
        
        self.inner_blocks = nn.ModuleList(
            [
                nn.ModuleList([
                    self.make_block(in_channels=_channels[out_idx], channels=channels, out_channels=channels, kernel_size=kernel_size, block_layers=1) 
                for in_idx in range(instage-1)]) 
            for out_idx in range(outstage)]
        )
        self.prune_blocks = nn.ModuleList(
            [
                nn.ModuleList([
                    ConvWithPrune(in_channels=channels, out_channels=channels,kernel_size=kernel_size, stride=1, bias=True, dimension=3)
                for in_idx in range(instage)])
            for out_idx in range(outstage)]
        )
        
        self.outter_blocks = nn.ModuleList(
            [
                self.make_block(in_channels=cum_channels[out_idx], channels=channels, out_channels=channels, kernel_size=kernel_size, block_layers=1)
            for out_idx in range(outstage-1)]
        )
        
        self.sigmoid = torch.nn.Sigmoid()
        
        # self.classifier = Classifier(channels)
        self.pruning = ME.MinkowskiPruning()
        self.relu_sp = ME.MinkowskiReLU(inplace=True)
        self.relu = torch.nn.ReLU(inplace=True)

        
        
    
    def make_block(self, in_channels=32, channels=32, out_channels=32, kernel_size=3, block_layers=3):
        return torch.nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_channels, out_channels=channels,
                kernel_size=kernel_size, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=True),
            ResNetBlock(block_layers=block_layers, channels=channels, kernel_size=kernel_size),
            ME.MinkowskiConvolution(
                in_channels=channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=1, bias=True, dimension=3))
        
    def split_mask(self, x):
        device = x.device
        mask_list = []
        octant = torch.sum((torch.div(x.C[:, 1:], x.tensor_stride[0], rounding_mode='floor')
                            %torch.tensor(2).to(device))
                           * torch.tensor([1, 2, 4]).to(device), axis=1)
        offset_list = get_stage_lst(self.instage)
        for _, offsets in enumerate(offset_list):
            mask = torch.sum(torch.stack([octant == a for a in offsets]), axis=0).bool()
            mask_list.append(mask)
        return mask_list
    
    def mask_datas(self, datas, mask_list):
        out_lst = []
        for data in datas:
            data_lst = []
            for mask in mask_list:
                data_lst.append(self.pruning(data, mask))
            out_lst.append(data_lst)
        return out_lst
    
    def mask_data(self, data, mask_list):
        data_lst = []
        for mask in mask_list:
            data_lst.append(self.pruning(data, mask))
        return data_lst
    
    def concat(self, x0, x1):
        feats = torch.cat([x0.F, x1.F], dim=0)
        coords = torch.cat([x0.C, x1.C], dim=0)
        out = create_new_sparse_tensor(
            features=feats,
            coordinates=coords,
            tensor_stride=x0.tensor_stride,
            dimension=x0.D, device=x0.device)

        return out

    def basic_module(self, siblings, prior, out_idx, in_idx):
        
        if siblings is None:
            inputs = prior
        else:
            assert in_idx > 0
            inner_block = self.inner_blocks[out_idx][in_idx-1]
            siblings = inner_block(siblings)
            inputs = self.concat(siblings, prior)
        # convetail = self.inner_tail[out_idx][in_idx]
        conveprune = self.prune_blocks[out_idx][in_idx]
        
        final_mlp = self.inner_mlps[out_idx][in_idx]
        
        # out = convetail(inputs)
        # out = self.relu_sp(out)
        out = conveprune(inputs, prior.C)
        
        # out = conv_tail(out)
        coord = out.C
        out_F = out.F
        # out_F = self.relu(out_F)
        out_F = final_mlp(out_F)
        out_cls = self.sigmoid(out_F)
        return coord, out_cls
        
    def forward(self, x_low, x_high_args):
        # 为x_low分组，方便重建
        x_occ_lst = x_high_args['x_occ_lst']
        
        
        out_cls_list = []
        ground_truth_list = []
        # 提取全局特征
        x_glob = self.block_in(x_low)
        mask_lst = self.split_mask(x_low)
        
        # 分组
        
        x_glob_split = self.mask_data(x_glob, mask_list=mask_lst)
        
        out_in_split = self.mask_datas(x_occ_lst, mask_list=mask_lst)
        
        
        out_glob_split = x_glob_split
        outsiblings = None
        for out_idx in range(self.outstage):
            in_split = out_in_split[out_idx]
            insiblings = None
            for in_idx in range(self.instage):
                curr_gt = in_split[in_idx]
                out_glob = out_glob_split[in_idx]
                coord, out_cls = self.basic_module(insiblings, out_glob, out_idx, in_idx)
                
                assert (coord != curr_gt.C).sum() == 0
                
                if insiblings is None:
                    insiblings = curr_gt
                else:
                    insiblings = self.concat(insiblings, curr_gt)
                out_cls_list.append(out_cls)
                ground_truth_list.append(curr_gt.F)
            
            if out_idx == self.outstage-1:
                break
            if outsiblings is None:
                outsiblings = x_occ_lst[out_idx]
            else:
                outsiblings = merge_two_frames(outsiblings, x_occ_lst[out_idx])
            out_glob_new = self.outter_blocks[out_idx](outsiblings)
            # new_split = self.mask_data(out_glob_new, mask_list=mask_lst)
            out_glob_new = ME.SparseTensor(out_glob_new.F, coordinate_map_key=x_glob.coordinate_map_key, coordinate_manager=x_glob.coordinate_manager, device=x_glob.device)
            glob_new = x_glob + out_glob_new
            out_glob_split = self.mask_data(glob_new, mask_list=mask_lst)
                
        return {'out_cls_list': out_cls_list,
                'ground_truth_list': ground_truth_list}

    @torch.no_grad()
    def encode(self, x_low, x_high_args, DBG=True):
        
        out_set = self.forward(x_low, x_high_args)
        bitstream_list = []
        for out_cls, ground_truth in zip(out_set['out_cls_list'], out_set['ground_truth_list']):

            prob = out_cls.view(-1,1).detach().cpu()
            print(prob)
            occupancy = ground_truth.view(-1).detach().cpu()
            # prob = torch.round(prob*1e3)/1e3
            occupancy = occupancy.to(torch.int16)
            
            bitstream = BAC.encode(prob, occupancy)
            print(bitstream)
            bitstream_list.append(bitstream)
            if DBG:
                occupancy_dec = BAC.decode(prob, bitstream)
                assert (occupancy == occupancy_dec).all()

        bitstream = pack_bitstream(bitstream_list)
        bits = len(bitstream)*8
        return {'enc_bytes': bitstream, 'bits': bits, 'x_low': x_low}


    @torch.no_grad()
    def decode(self, enc_out):
        bitstream = enc_out['enc_bytes']
        x_low = enc_out['x_low']
        x_glob = self.block_in(x_low)
        mask_lst = self.split_mask(x_low)
        
        # 分组
        x_glob_split = self.mask_data(x_glob, mask_list=mask_lst)
        
        bitstream_list = unpack_bitstream(bitstream)
        
        out_glob_split = x_glob_split
        outsiblings = None
        
        
        out_cls_list = []
        dec_gt_list = []
        for out_idx in range(self.outstage):
            insiblings = None
            for in_idx in range(self.instage):
                bitstream = bitstream_list.pop(0)
        
                out_glob = out_glob_split[in_idx]
                coord, out_cls, occupancy = self.basic_decode_module(insiblings, out_glob, out_idx, in_idx, bitstream)
                
                dec_gt = ME.SparseTensor(occupancy, coord, device=x_low.device)
                if insiblings is None:
                    insiblings = dec_gt
                else:
                    insiblings = self.concat(insiblings, dec_gt)
                out_cls_list.append(out_cls)
                dec_gt_list.append(occupancy)
            if out_idx == self.outstage-1:
                break
            if outsiblings is None:
                outsiblings = dec_gt_list[out_idx]
            else:
                outsiblings = merge_two_frames(outsiblings, dec_gt_list[out_idx])
            out_glob_new = self.outter_blocks[out_idx](outsiblings)
            out_glob_new = ME.SparseTensor(out_glob_new.F, coordinate_map_key=x_glob.coordinate_map_key, coordinate_manager=x_glob.coordinate_manager, device=x_glob.device)
            glob_new = x_glob + out_glob_new
            out_glob_split = self.mask_data(glob_new, mask_list=mask_lst)
        return dec_gt_list
                

        


    @torch.no_grad()
    def basic_decode_module(self, siblings, prior, out_idx, in_idx, bitstream):
        if siblings is None:
            inputs = prior
        else:
            assert in_idx > 0
            inner_block = self.inner_blocks[out_idx][in_idx-1]
            siblings = inner_block(siblings)
            inputs = self.concat(siblings, prior)
        # convetail = self.inner_tail[out_idx][in_idx]
        conveprune = self.prune_blocks[out_idx][in_idx]
        
        final_mlp = self.inner_mlps[out_idx][in_idx]
        
        # out = convetail(inputs)
        # out = self.relu_sp(out)
        out = conveprune(inputs, prior.C)
        
        # out = conv_tail(out)
        coord = out.C
        out_F = out.F
        # out_F = self.relu(out_F)
        out_F = final_mlp(out_F)
        out_cls = self.sigmoid(out_F)
        
        occupancy = BAC.decode(out_cls.detach().cpu(), bitstream)
        occupancy = occupancy.to(torch.float32).view(-1,1)
        return coord, out_cls, occupancy
         

    @torch.no_grad()
    def test(self, x_low, x_high):
        # real bitstream
        bitstream = self.encode(x_low, x_high)
        bits = len(bitstream)*8
        x_high_rec = self.decode(x_low, bitstream)
        assert (sort_sparse_tensor(x_high).C==sort_sparse_tensor(x_high_rec).C).all()
        return bits






