
import time
import torch
from torch import nn
# from inout import *
# from nerfutils import *
# from pointutils import *
from module_utils import PointwiseMLP
from upsample import CNP
from module_utils import BinaryArithmeticCoding
from function_utils import generate_sparse, pack_bitstream, unpack_bitstream

from models.module_utils import qscTensor, octree_level_obj
bce = nn.BCELoss(reduction='sum')
device = torch.device("cuda")



class LINR_PCGC_Model(torch.nn.Module):
    def __init__(self, inargs):
        super(LINR_PCGC_Model, self).__init__()
        scale_num = inargs['scale_num']
        self.scale_num = scale_num
        in_channel = inargs['in_channel']
        hidden_channel_conv = inargs['hidden_channel_conv']
        block_layers = inargs['block_layers']
        outstage = inargs['outstage']
        instage = inargs['instage']
        kernel_size = inargs['kernel_size']

        self.scale_emb = nn.Embedding(scale_num, 8)
        self.scale_mlp = nn.ModuleList(
            [PointwiseMLP([8+in_channel, 16, 8]) for i in range(scale_num)]
        )
        self.upsampler = CNP(in_channels=8,channels=hidden_channel_conv,kernel_size=kernel_size,block_layers=block_layers, outstage=outstage, instage=instage)
        
        self.sigmoid = nn.Sigmoid()
    def logic_core(self, in_args):
        # 'occupancy_0':occupancy_0,
        # 'occupancy_1':occupancy_1,
        coord = in_args['coord']
        # ground_truth = in_args['ground_truth']

        occ_lst = in_args['occ_lst']
        offset_tensor = in_args['offset_tensor']
        scale_idx = in_args['scale_idx']
        
        scale_em = self.scale_emb(torch.tensor(scale_idx,device=device))
        N = len(coord)
        mix_data = scale_em.repeat(N, 1)
        mix_data = torch.cat([mix_data, offset_tensor], dim=-1)
        
        intensor = self.scale_mlp[scale_idx](mix_data)
        
        # 复制scale_em1为N份
        
        # coord_double = coord * 2

        x_low = generate_sparse(coord, intensor, 1)
        
        x_occ_lst = []
        for occ in occ_lst:
            x_occ_lst.append(generate_sparse(coord, occ, 1))
        
        
        upout = self.upsampler(x_low, {'x_occ_lst':x_occ_lst})
        
        return upout
        
        
    
    def forward(self, inargs):
        core_out = self.logic_core(inargs)
        out_cls_list = core_out['out_cls_list']
        ground_truth_list = core_out['ground_truth_list']
        bits_t = 0
        for out_cls, ground_truth in zip(out_cls_list, ground_truth_list):
            
            bits_tmp = bce(out_cls, ground_truth) / torch.log(torch.tensor(2.0))
            bits_t = bits_t + bits_tmp
        return bits_t
    
    
    @torch.no_grad()
    def codec_with_point(self, inargs):
        st1 = time.time()
        
        core_out = self.logic_core(inargs)
        out_cls_list = core_out['out_cls_list']
        ground_truth_list = core_out['ground_truth_list']
        
        derive_bit_pc = inargs['derive_bit_pc']
        
        bits_t = 0
        for out_cls, ground_truth in zip(out_cls_list, ground_truth_list):
            bits_tmp = bce(out_cls.detach(), ground_truth) / torch.log(torch.tensor(2.0))
            bits_t = bits_t + bits_tmp
        bits_t = bits_t.item()
        
        # pov_f = pov_s.F
        # 将out_cls_list拉成一个长条
        pov_lst = []
        for out_cls in out_cls_list:
            pov_lst.append(out_cls.view(-1))
        pov_f = torch.cat(pov_lst, dim=0)
        
        povs_t_2 = pov_f.view(-1,1).detach().cpu()
        # 给出分布 1-povs_t_2, 1, 0
        # pov_t_to_use = torch.cat([1-povs_t_2, torch.ones((len(povs_t_2),1),device=povs.device), torch.ones((len(povs_t_2),1),device=povs.device)], dim=-1).detach().cpu()
        # 把ground_truth拉成长条
        
        BAC = BinaryArithmeticCoding()
        
        st2 = time.time()
        
        # 中间的时间用来生成点云热力图
        all_stage_bit_pc = []
        if derive_bit_pc:
            coord = inargs['coord'] 
            all_stage_bit_pc.append(coord)
            
            gt_state = []
            bit_state = []
            for out_cls, ground_truth in zip(out_cls_list, ground_truth_list):
                # gt=1,p=out_cls, gt=0,p=1-out_cls
                stage_bit_pc=1-(out_cls.detach()-ground_truth).abs()
                gt_state.append(ground_truth)
                bit_state.append(stage_bit_pc)
            
            gt_state = torch.cat(gt_state, dim=-1)
            bit_state = torch.cat(bit_state, dim=-1)
            all_stage_bit_pc.append(gt_state)
            all_stage_bit_pc.append(bit_state)
                
        
        # 进行编码
        st_m1 = time.time()
        mask_lst = []
        for ground_truth in ground_truth_list:
            mask_lst.append(ground_truth.view(-1))
        mask = torch.cat(mask_lst, dim=0)
        
        ground_truth_2 = mask.to(torch.int16).view(-1).detach().cpu()
        enc_bytes = BAC.encode(povs_t_2, ground_truth_2)
        
        st3 = time.time()
        
        recon_gt = BAC.decode(povs_t_2, enc_bytes)
        st4 = time.time()
        
        
        # # 进行编码
        # enc_bytes = torchac.encode_float_cdf(pov_t_to_use, ground_truth_2, True, True)
        
        # recon_gt = torchac.decode_float_cdf(pov_t_to_use, enc_bytes)
        
        assert (recon_gt != ground_truth_2).sum() == 0
        
        bits = len(enc_bytes) * 8
        
        # print('bits_t:', bits_t)
        # print('bits:', bits)
        
        enc_time = st2-st1 + st3-st_m1
        dec_time = st2 -st1 + st4 - st3
        # memory = torch.cuda.max_memory_allocated() / 1024 ** 3
        return {'bits':bits, 'enc_bytes':enc_bytes, 'enc_time':enc_time, 'dec_time':dec_time, 'bits_t': bits_t,'all_stage_bit_pc':all_stage_bit_pc}        
    
    @torch.no_grad()
    def codec(self, inargs):
        st1 = time.time()
        
        core_out = self.logic_core(inargs)
        out_cls_list = core_out['out_cls_list']
        ground_truth_list = core_out['ground_truth_list']
        bits_t = 0
        for out_cls, ground_truth in zip(out_cls_list, ground_truth_list):
            bits_tmp = bce(out_cls.detach(), ground_truth) / torch.log(torch.tensor(2.0))
            bits_t = bits_t + bits_tmp
        
        # pov_f = pov_s.F
        # 将out_cls_list拉成一个长条
        pov_lst = []
        for out_cls in out_cls_list:
            pov_lst.append(out_cls.view(-1))
        pov_f = torch.cat(pov_lst, dim=0)
        
        povs_t_2 = pov_f.view(-1,1).detach().cpu()
        # 给出分布 1-povs_t_2, 1, 0
        # pov_t_to_use = torch.cat([1-povs_t_2, torch.ones((len(povs_t_2),1),device=povs.device), torch.ones((len(povs_t_2),1),device=povs.device)], dim=-1).detach().cpu()
        # 把ground_truth拉成长条
        
        BAC = BinaryArithmeticCoding()
        
        st2 = time.time()
        # 进行编码
        
        mask_lst = []
        for ground_truth in ground_truth_list:
            mask_lst.append(ground_truth.view(-1))
        mask = torch.cat(mask_lst, dim=0)
        
        ground_truth_2 = mask.to(torch.int16).view(-1).detach().cpu()
        enc_bytes = BAC.encode(povs_t_2, ground_truth_2)
        
        st3 = time.time()
        
        recon_gt = BAC.decode(povs_t_2, enc_bytes)
        st4 = time.time()
        
        
        # # 进行编码
        # enc_bytes = torchac.encode_float_cdf(pov_t_to_use, ground_truth_2, True, True)
        
        # recon_gt = torchac.decode_float_cdf(pov_t_to_use, enc_bytes)
        
        assert (recon_gt != ground_truth_2).sum() == 0
        
        bits = len(enc_bytes) * 8
        
        # print('bits_t:', bits_t)
        # print('bits:', bits)
        
        enc_time = st2-st1 + st3-st2
        dec_time = st2 - st1 + st4 - st3
        
        return {'bits':bits, 'enc_bytes':enc_bytes, 'enc_time':enc_time, 'dec_time':dec_time, 'bits_t': bits_t}        
        
    # @torch.no_grad()
    # def test(self):
    #     enc_out = self.encode()
    #     dec_out = self.decode(enc_out)
    
    
    @torch.no_grad()
    def encode(self, in_args):
        # 'occupancy_0':occupancy_0,
        # 'occupancy_1':occupancy_1,
        coord = in_args['coord']
        # ground_truth = in_args['ground_truth']

        occ_lst = in_args['occ_lst']
        offset_tensor = in_args['offset_tensor']
        scale_idx = in_args['scale_idx']
        
        scale_em = self.scale_emb(torch.tensor(scale_idx,device=device))
        N = len(coord)
        mix_data = scale_em.repeat(N, 1)
        mix_data = torch.cat([mix_data, offset_tensor], dim=-1)
        
        intensor = self.scale_mlp[scale_idx](mix_data)
        
        # 复制scale_em1为N份
        
        # coord_double = coord * 2

        x_low = generate_sparse(coord, intensor, 1)
        
        x_occ_lst = []
        for occ in occ_lst:
            x_occ_lst.append(generate_sparse(coord, occ, 1))
        
        
        upencout = self.upsampler.encode(x_low, {'x_occ_lst':x_occ_lst})
        
        return upencout       
        
    @torch.no_grad()
    def decode(self, inagrs):
        enc_bytes = inagrs['enc_bytes']
        coord = inagrs['coord']
        offset_tensor = inagrs['offset_tensor']
        scale_idx = inagrs['scale_idx']
        scale_em = self.scale_emb(torch.tensor(scale_idx,device=device))
        
        N = len(coord)
        mix_data = scale_em.repeat(N, 1)
        mix_data = torch.cat([mix_data, offset_tensor], dim=-1)
        intensor = self.scale_mlp[scale_idx](mix_data)
        
        x_low = generate_sparse(coord, intensor, 1)
        
        # all_stage_bytes = unpack_bitstream(enc_bytes)
        
        occ_lst = self.upsampler.decode({'enc_bytes':enc_bytes, 'x_low':x_low})
        return occ_lst
        
        