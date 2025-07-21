import json
import os
import torch
from model_compression.model_size_est import Model_Estimate
from test_utils import dec_all_frame_low_xyz
from models.module_utils import qscTensor
from glob_params import offsets_ini, offset_of_neigbor
from models.module_utils import octree_level_obj
# 对编码出来的结果进行解码，并验证是否是无损的
decompress_model = Model_Estimate().decompress_model
def decode(inargs):
    # result_dir, Gen_Model, 
    gop_names = inargs['gop_names']
    result_enc_dir = inargs['result_enc_dir']
    result_dec_dir = inargs['result_dec_dir']
    Gen_Model = inargs['Gen_Model']
    
    
    
    for g_idx, gop_name in enumerate(gop_names):
        gop_bound = [int(d) for d in gop_name.replace('gop_','').split('_')]
        gop_size = gop_bound[1] - gop_bound[0] + 1
        
        gop_args = {}
        gop_args['frame_num'] = gop_size
        gop_args['result_enc_dir'] = result_enc_dir
        gop_args['result_dec_dir'] = result_dec_dir
        gop_args['Gen_Model'] = Gen_Model
        gop_args['gop_name'] = gop_name
        
        decode_one_gop(gop_args)
        
    
def decode_one_gop(inargs):
    result_enc_dir = inargs['result_enc_dir']
    gop_name = inargs['gop_name']
    frame_num = inargs['frame_num']
    
    
    Gen_Model = inargs['Gen_Model']
    model_ori = Gen_Model()
    
    gop_enc_dir = os.path.join(result_enc_dir, gop_name)
    
    gop_enc_bin_dir = os.path.join(gop_enc_dir, 'bins')
    # 解码最低尺度的点云
    all_xyz_enc_bytes = []
    for frame_idx in range(frame_num):
        frame_idx_str = str(frame_idx).zfill(4)
        frame_enc_bin_dir = os.path.join(gop_enc_bin_dir, f'frame{frame_idx_str}_low.bin')
        with open(frame_enc_bin_dir, 'rb') as f:
            xyz_enc_bytes = f.read()
        all_xyz_enc_bytes.append({'enc_bytes': xyz_enc_bytes})
    all_xyz_low = dec_all_frame_low_xyz(all_xyz_enc_bytes)
    
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
    
    # 解码每一帧
    for frame_idx in range(frame_num):
        frame_idx_str = str(frame_idx).zfill(4)
        
        frame_enc_bytes = []
        for s_idx in range(scale_num):
            bin_path = os.path.join(gop_enc_bin_dir, f"frame{frame_idx_str}_scale{s_idx}.bin")
            with open(bin_path, 'rb') as f:
                bytes_t = f.read()
            frame_enc_bytes.append(bytes_t)
        xyz_low = all_xyz_low[frame_idx]
        dec_frame_coord = decode_one_frame(model_esti, frame_enc_bytes, xyz_low)
        
        
        
def decode_one_frame(model, frame_enc_bytes, xyz_low):
    scale_num = len(frame_enc_bytes)
    
    lowx = torch.tensor(xyz_low).cuda()
    for s_idx in range(scale_num-1 , -1, -1):
        bytes_t = frame_enc_bytes[s_idx]
        
        qsct = qscTensor(lowx)
        qsct.set_offset_tensor(offsets_ini)
        offset_tensor = qsct.get_offset_tensor()
        coord = qsct.get_coord()
        occ_lst = model.decode({'enc_bytes': bytes_t, 'coord': coord, 'offset_tensor': offset_tensor, 'scale_idx': s_idx})
        occupancy = torch.cat(occ_lst, dim=-1)
        
        highx = octree_level_obj.upper_layer(coord, occupancy)
        lowx = highx
    return lowx
        
        


if __name__ == '__main__':
    from models.model_core import LINR_PCGC_Model
    from glob_params import offsets_ini, offset_of_neigbor
    from datautils.custom_dataset import MyDataset
    
    outputdir = '/home/huangwenjie/python_files/PCAC/nerpcc_proj/tests/mytest_new13_8_1/output/basketballplayer'
    # gop_names = ['gop_0_31','gop_32_63','gop_64_95']
    gop_names = ['gop_0_31']
    Gen_Model =  lambda: LINR_PCGC_Model({'scale_num':7, 'in_channel':len(offsets_ini), 'hidden_channel_conv':8, 'block_layers':1, 'outstage':8, 'instage':1}).cuda()
    
    datapath = '/home/data/datasets/point_cloud/Owlii/Owlii_10bit_rgb/basketball_player'
    
    dataset=MyDataset(datapath, handle_dir='tmp/basketball',ori_type='ply', stage=8, scale_num=7)
    dataset.set_prefix_data({'offsets_ini':offsets_ini, 'offset_of_neigbor':offset_of_neigbor,'min_point_num':64})
    result_enc_dir = 'result_enc/basketball'
    result_dec_dir = 'result_dec/basketball'
    
    dec_args = {'gop_names': gop_names, 'Gen_Model': Gen_Model, 'result_enc_dir': result_enc_dir,'result_dec_dir': result_dec_dir}
    decode(dec_args)