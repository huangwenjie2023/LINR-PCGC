import json
import os
import torch
from model_compression.model_size_est import Model_Estimate
from test_utils import dec_all_frame_low_xyz
from models.module_utils import qscTensor
from glob_params import offsets_ini, offset_of_neigbor
from models.module_utils import octree_level_obj
from datautils.custom_dataset import MytestDataset, Read_Data_with_cache, MyDataset
from models.sort_functions import sort_by_coord_sum_c
from datautils.custom_dataset import write_ply_o3d, write_ply_ascii

# 对编码出来的结果进行解码，并验证是否是无损的
decompress_model = Model_Estimate().decompress_model
def decode(inargs):
    # result_dir, Gen_Model, 
    gop_names = inargs['gop_names']
    result_enc_dir = inargs['result_enc_dir']
    result_dec_dir = inargs['result_dec_dir']
    Gen_Model = inargs['Gen_Model']
    dataset = inargs['dataset']
    
    if not os.path.exists(result_dec_dir):
        os.makedirs(result_dec_dir)
    
    for g_idx, gop_name in enumerate(gop_names):
        
        
        
        gop_bound = [int(d) for d in gop_name.replace('gop_','').split('_')]
        
        gop_range = range(*gop_bound)
        reading_data = Read_Data_with_cache(dataset, gop_range)
        
        gop_size = gop_bound[1] - gop_bound[0] + 1
        
        gop_args = {}
        gop_args['gop_bound'] = gop_bound
        gop_args['frame_num'] = gop_size
        gop_args['result_enc_dir'] = result_enc_dir
        gop_args['result_dec_dir'] = result_dec_dir
        gop_args['Gen_Model'] = Gen_Model
        gop_args['gop_name'] = gop_name
        gop_args['reading_data'] = reading_data
        gop_args['write_flag'] = inargs['write_flag']
        
        decode_one_gop(gop_args)
        
    
def decode_one_gop(inargs):
    result_enc_dir = inargs['result_enc_dir']
    result_dec_dir = inargs['result_dec_dir']
    gop_name = inargs['gop_name']
    frame_num = inargs['frame_num']
    gop_bound = inargs['gop_bound']
    reading_data = inargs['reading_data']
    
    Gen_Model = inargs['Gen_Model']
    write_flag = inargs['write_flag']
    model_ori = Gen_Model()
    
    gop_enc_dir = os.path.join(result_enc_dir, gop_name)
    
    gop_enc_bin_dir = os.path.join(gop_enc_dir, 'bins')
    # 解码最低尺度的点云
    # all_xyz_enc_bytes = []
    # for frame_idx in range(frame_num):
    #     frame_idx_str = str(frame_idx).zfill(4)
    frame_enc_bin_dir = os.path.join(gop_enc_bin_dir, f'low_enc_bytes.bin')
    with open(frame_enc_bin_dir, 'rb') as f:
        xyz_enc_bytes = f.read()
        # all_xyz_enc_bytes.append({'enc_bytes': xyz_enc_bytes})
    low_xyz_ret = dec_all_frame_low_xyz(xyz_enc_bytes)
    all_xyz_low = low_xyz_ret['all_xyz_low']
    all_coord_data_min = low_xyz_ret['all_coord_data_min']
    
    # 解码模型
    side_info_path = os.path.join(gop_enc_dir, 'side_info.json')
    with open(side_info_path, 'r') as f:
        side_info = json.load(f)
    
    model_path = os.path.join(gop_enc_bin_dir, 'model.bin')
    with open(model_path, 'rb') as f:
        final_bytes = f.read()
    side_info['final_bytes'] = final_bytes
    model_esti = decompress_model(model_ori, side_info)
    
    scale_num = model_esti.scale_num
    
    start_idx = gop_bound[0]
    
    
    # 解码每一帧
    for frame_idx in range(frame_num):
        real_frame_idx = start_idx + frame_idx
        real_frame_idx_str = str(real_frame_idx).zfill(4)
        
        frame_idx_str = str(frame_idx).zfill(4)
        
        frame_enc_bytes = []
        for s_idx in range(scale_num):
            bin_path = os.path.join(gop_enc_bin_dir, f"frame{frame_idx_str}_scale{s_idx}.bin")
            with open(bin_path, 'rb') as f:
                bytes_t = f.read()
            frame_enc_bytes.append(bytes_t)
        
        
        
        
        gt_datas = reading_data[frame_idx]
        # print([len(b) for b in frame_enc_bytes])
        xyz_low = all_xyz_low[frame_idx]
        xyz_low = torch.tensor(xyz_low, device='cuda').to(torch.int32)
        # ref = gt_datas['all_input_info'][-1]['xyzqsc_t'].get_coord()
        # assert (xyz_low != ref).sum() == 0
        # dec_out = decode_one_frame(model_esti, frame_enc_bytes, gt_datas['all_input_info'][-1]['xyzqsc_t'].get_coord())
        dec_out = decode_one_frame(model_esti, frame_enc_bytes.copy(), xyz_low)
        dec_coord = dec_out['dec_coord']
        
        data_min = all_coord_data_min[frame_idx,:]
        
        dec_coord_final = dec_coord + data_min
        
        # ori = gt_datas['ori']
        assert (dec_coord_final != gt_datas).sum() == 0
        print(f"frame {frame_idx} is correct")
        
        # 将解码出来的点云写出去
        if write_flag:
            dec_coord_np = dec_coord_final.cpu().numpy()
            write_ply_ascii(os.path.join(result_dec_dir, f'frame{real_frame_idx_str}.ply'), dec_coord_np)
        
        
        
        
        
        
def decode_one_frame(model, frame_enc_bytes, xyz_low):
    scale_num = len(frame_enc_bytes)
    
    lowx = xyz_low
    # all_xs = []
    for s_idx in range(scale_num-1 , -1, -1):
        # print(lowx)
        # print(lowx.shape)
        bytes_t = frame_enc_bytes[s_idx]
        # all_xs.append(lowx)
        
        qsct = qscTensor(lowx)
        qsct.set_offset_tensor(offsets_ini)
        offset_tensor = qsct.get_offset_tensor()
        coord = qsct.get_coord()
        occ_lst = model.decode({'enc_bytes': bytes_t, 'coord': coord, 'offset_tensor': offset_tensor, 'scale_idx': s_idx})
        occupancy = torch.cat(occ_lst, dim=-1)
        
        highx = octree_level_obj.upper_layer(coord, occupancy)
        lowx = highx
    
    # all_xs.append(lowx)
    # return {'dec_coord': lowx, 'all_xs': all_xs}
    return {'dec_coord': lowx}
        
        


if __name__ == '__main__':
    from models.model_core import LINR_PCGC_Model
    from glob_params import offsets_ini, offset_of_neigbor
    
    
    outputdir = '/home/huangwenjie/python_files/PCAC/nerpcc_proj/tests/mytest_new13_8_1/output/basketballplayer'
    # gop_names = ['gop_0_31','gop_32_63','gop_64_95']
    gop_names = ['gop_0_31']
    Gen_Model =  lambda: LINR_PCGC_Model({'scale_num':7, 'in_channel':len(offsets_ini), 'hidden_channel_conv':8, 'block_layers':1, 'outstage':8, 'instage':1}).cuda()
    
    datapath = '/home/data/datasets/point_cloud/Owlii/Owlii_10bit_rgb/basketball_player'
    
    # dataset=MyDataset(datapath, handle_dir='tmp/basketball',ori_type='ply', stage=8, scale_num=7)
    dataset = MytestDataset(datapath, ori_type='ply')
    # dataset.set_prefix_data({'offsets_ini':offsets_ini, 'offset_of_neigbor':offset_of_neigbor,'min_point_num':64})
    result_enc_dir = 'result_enc/basketball'
    result_dec_dir = 'result_dec/basketball'

    
    dec_args = {'gop_names': gop_names, 'Gen_Model': Gen_Model, 'result_enc_dir': result_enc_dir,'result_dec_dir': result_dec_dir,'dataset':dataset, 'write_flag': True}
    decode(dec_args)