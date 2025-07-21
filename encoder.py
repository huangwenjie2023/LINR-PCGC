# 训练结束后写一个真实的encoder
import os
import torch
import json
from model_compression.model_size_est import Model_Estimate
from datautils.custom_dataset import Read_Data_with_cache, MyDataset
from test_utils import enc_all_frame_low_xyz
from decoder import decode_one_frame
from models.module_utils import octree_level_obj

compress_model = Model_Estimate().compress_model

def write_bin_file(frame_idx, all_bytes, bins_dir):
    frame_idx_str = str(frame_idx).zfill(4)
    for idx, bytes_t in enumerate(all_bytes):
        bin_path = os.path.join(bins_dir, f"frame{frame_idx_str}_scale{idx}.bin")
        with open(bin_path, 'wb') as f:
            f.write(bytes_t)
            
def encode(enc_args):
    # 需要给出 outputdir, gop_names, Gen_Model, dataset, result_dir
    outputdir = enc_args['outputdir']
    gop_names = enc_args['gop_names']
    Gen_Model = enc_args['Gen_Model']
    dataset = enc_args['dataset']
    encode_dir = enc_args['encode_dir']
    
    if not os.path.exists(encode_dir):
        os.makedirs(encode_dir)
        
    for g_idx, gop_name in enumerate(gop_names):
        gop_arg = {}
        gop_arg['Gen_Model'] = Gen_Model
        
        gop_bound = [int(d) for d in gop_name.replace('gop_','').split('_')]
        gop_size = gop_bound[1] - gop_bound[0] + 1
        
        gop_arg['frame_num'] = gop_size
        
        model_path = os.path.join(outputdir, gop_name,'model.pth')
        gop_arg['model_path'] = model_path
        
        gop_result_dir = os.path.join(encode_dir, gop_name)
        gop_arg['result_dir'] = gop_result_dir
        
        
        gop_range = range(*gop_bound)
        reading_data = Read_Data_with_cache(dataset, gop_range)
        gop_arg['reading_data'] = reading_data
        
        low_enc_ret = enc_all_frame_low_xyz(reading_data, gop_size)
        gop_arg['low_enc_bytes'] = low_enc_ret
        
        encode_one_gop(gop_arg)


def encode_one_gop(inargs):
    model_path = inargs['model_path']
    Gen_Model = inargs['Gen_Model']

    ckpt = torch.load(model_path)
    estd_model = Gen_Model()
    estd_model.load_state_dict(ckpt['model'])
    
    
    bitdepth = ckpt.get('bitdepth')
    if bitdepth is None:
        bitdepth = 8

    
    model_ori = Gen_Model()
    frame_num = inargs['frame_num']
    


    result_dir = inargs['result_dir']
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    bins_dir = os.path.join(result_dir, "bins")
    if not os.path.exists(bins_dir):
        os.makedirs(bins_dir)
    
    low_enc_bytes = inargs.get('low_enc_bytes')
    if low_enc_bytes is None:
        raise ValueError('low_enc_bytes is None')
    with open(os.path.join(bins_dir, 'low_enc_bytes.bin'), 'wb') as f:
        f.write(low_enc_bytes)
        
    
    # all_xlow_info = enc_all_frame_low_xyz(inargs['reading_data'])
    # for frame_idx in range(frame_num):
    #     frame_xyzlow_ret = low_enc_ret[frame_idx]
    #     xyz_enc_bytes = frame_xyzlow_ret['enc_bytes']
    #     frame_idx_str = str(frame_idx).zfill(4)
    #     xyzlow_bin = os.path.join(bins_dir, f"frame{frame_idx_str}_low.bin")
    #     with open(xyzlow_bin, 'wb') as f:
    #         f.write(xyz_enc_bytes)
    
    compress_out_to_use = compress_model(estd_model, bitdepth, True, model_ori)

    model_to_use = compress_out_to_use['new_model']
    modelbit_use = compress_out_to_use['bit_real']
    enc_mode_to_use = compress_out_to_use['enc_mode']

    
    model_bytes = compress_out_to_use['final_bytes']
    model_bin = os.path.join(bins_dir, "model.bin")
    

    with open(model_bin, 'wb') as f:
        f.write(model_bytes)
    side_info = {'mu': compress_out_to_use['mu'].item(), 'b': compress_out_to_use['b'].item(), 'min_param': compress_out_to_use['min_param'].item(), 'max_param': compress_out_to_use['max_param'].item(),'enc_mode': enc_mode_to_use, 'bitdepth': bitdepth}
    # side_info = {}
    side_info_path = os.path.join(result_dir, "side_info.json")
    with open(side_info_path, 'w') as f:
        json.dump(side_info, f, indent=4)
        
    
    
    reading_data = inargs['reading_data']
    
    bits_test_frame = 0
    # bits_train_frame = 0
    
    point_test_frame = 0
    # xyzlow_test_frame = 0
    
    for frame_idx in range(frame_num):
        print('handle frame:', frame_idx)
        frame_data = reading_data[frame_idx]
        
        all_input_info = frame_data['all_input_info']
        point_num = frame_data['point_num']
        
        handle_out = encode_one_frame(model_to_use, all_input_info, frame_data['ori'])
        bits_t_tmp = handle_out['all_bit']
        # bits_train_tmp = handle_out['all_bit_t']
        # bits_train_frame += bits_train_tmp
        
        all_bytes = handle_out['all_bytes']

        

        write_bin_file(frame_idx, all_bytes, bins_dir)
        # print([len(bytes_t) for bytes_t in all_bytes])
        # bpp_t_tmp = bits_t_tmp / point_num
        bits_test_frame += bits_t_tmp
        point_test_frame += point_num

        # frame_xyzlow_ret = low_enc_ret[frame_idx]
        # xyzlow_test_frame += frame_xyzlow_ret['bits']
        # xyzlow_test_frame += frame_data['xyzQ_low_bits']
    
        torch.cuda.empty_cache()
    
def encode_one_frame(model, all_inargs, ori):
    # print(ori)
    # TODO: 这个地方要根据不同模型的变化而变化
    all_bit = 0
    all_bytes = []

    
    all_bit_t = 0
    for s_idx, inargs in enumerate(all_inargs):
        putin_args = {}
        putin_args.update(inargs)
        xyzqsc_t = inargs['xyzqsc_t']
        coord = xyzqsc_t.get_coord()
        offset_tensor = xyzqsc_t.get_offset_tensor()
        putin_args['offset_tensor'] = offset_tensor
        
        putin_args['coord'] = coord

        ret_out = model.encode(putin_args)
        
        scale_enc_byte = ret_out['enc_bytes']
        scale_bit = ret_out['bits']
        
        # all_bit_t += ret_out['bits_t']
        # enc_time += ret_out['enc_time']
        # dec_time += ret_out['dec_time']
        all_bit += scale_bit
        all_bytes.append(scale_enc_byte)
    # xyzlow = all_inargs[-1]['xyzqsc_t'].get_coord()
    # dec_out = decode_one_frame(model, all_bytes, xyzlow)
    
    # dec_all_xs = dec_out['all_xs'][::-1]
    # for s_idx, inargs in enumerate(all_inargs):
    #     putin_args = {}
    #     putin_args.update(inargs)
    #     xyzqsc_t = inargs['xyzqsc_t']
    #     coord = xyzqsc_t.get_coord()
        
    #     dec_x = dec_all_xs[s_idx+1]
    #     assert (dec_x!=coord).sum() == 0
    # print(1)
    
    # ori = all_inargs['ori']
    # assert (ori!=dec_out['dec_coord']).sum() == 0
    
    return {'all_bit': all_bit, 'all_bytes': all_bytes}



if __name__ == '__main__':
    from models.model_core import LINR_PCGC_Model
    from glob_params import offsets_ini, offset_of_neigbor
    outputdir = '/home/huangwenjie/python_files/PCAC/nerpcc_proj/tests/mytest_new13_8_1/output/basketballplayer'
    # gop_names = ['gop_0_31','gop_32_63','gop_64_95']
    gop_names = ['gop_0_31']
    Gen_Model =  lambda: LINR_PCGC_Model({'scale_num':7, 'in_channel':len(offsets_ini), 'hidden_channel_conv':8, 'block_layers':1, 'outstage':8, 'instage':1}).cuda()
    
    datapath = '/home/data/datasets/point_cloud/Owlii/Owlii_10bit_rgb/basketball_player'
    # 删除handle_dir
    handle_dir = 'tmp/basketball'
    # os.system(f'rm -rf {handle_dir}')
    dataset=MyDataset(datapath, handle_dir=handle_dir,ori_type='ply', stage=8, scale_num=7, derive_ori=True)
    dataset.set_prefix_data({'offsets_ini':offsets_ini, 'offset_of_neigbor':offset_of_neigbor,'min_point_num':64})
    encode_dir = 'result_enc/basketball'
    
    enc_args = {'outputdir': outputdir, 'gop_names': gop_names, 'Gen_Model': Gen_Model, 'dataset': dataset, 'encode_dir': encode_dir}
    encode(enc_args)
    