import torch
import os
import json
import numpy as np
import zlib
import time
from models.function_utils import pack_bitstream, unpack_bitstream
def write_bin_file(frame_idx, all_bytes, bins_dir):
    frame_idx_str = str(frame_idx).zfill(4)
    for idx, bytes_t in enumerate(all_bytes):
        bin_path = os.path.join(bins_dir, f"frame{frame_idx_str}_scale{idx}.bin")
        with open(bin_path, 'wb') as f:
            f.write(bytes_t)
        
        
def Test_one_gop(inargs):
    enc_time = 0
    dec_time = 0
    write_flag = inargs['write_flag']
    model_path = inargs['model_path']
    
    Gen_Model = inargs['Gen_Model']
    estd_model = Gen_Model()
    
    
    ckpt = torch.load(model_path)
    estd_model.load_state_dict(ckpt['model'])
    bitdepth = ckpt.get('bitdepth')
    if bitdepth is None:
        bitdepth = 8
    
    low_enc_ret = inargs.get('low_enc_ret')
    if low_enc_ret is None and write_flag:
        raise ValueError('low_enc_ret is None while write_flag is True')

    
    model_ori = Gen_Model()
    # embedding = inargs['embedding']
    frame_num = inargs['frame_num']
    
    compress_model_test = inargs['compress_model_test']
    
    
    result_dir = inargs['result_dir']
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    bins_dir = os.path.join(result_dir, "bins")
    if not os.path.exists(bins_dir):
        os.makedirs(bins_dir)
        
    
    # 写出最低分辨率的xyz信息
    xlow_enc_modes = []
    xlow_enc_flags = []
    if write_flag:
        # all_xlow_info = enc_all_frame_low_xyz(inargs['reading_data'])
        
        # frame_xyzlow_ret = low_enc_ret[frame_idx]
        # xyz_enc_bytes = frame_xyzlow_ret['enc_bytes']
        # xyz_enc_mode = frame_xyzlow_ret['enc_mode']
        # xlow_enc_flags.append(frame_xyzlow_ret['enc_flag'])
        # xlow_enc_modes.append(xyz_enc_mode)
        
        bin_path = os.path.join(bins_dir, f"low_enc_bytes.bin")
        with open(bin_path, 'wb') as f:
            f.write(low_enc_ret)

    # 'esti_compress_model': esti_compress_model,
    # esti_compress_model = inargs['esti_compress_model']
    # compress_out_to_use = esti_compress_model(estd_model, model_ori, bitdepth)
    # compress_out_to_use['enc_time'] = 0
    # compress_out_to_use['dec_time'] = 0
    # compress_out_to_use['final_byte'] = b''
    compress_out_to_use = compress_model_test(estd_model, model_ori, bitdepth)

    model_to_use = compress_out_to_use['new_model']
    
    
    modelbit_use = compress_out_to_use['bit_real']
    enc_mode_to_use = compress_out_to_use['enc_mode']
    
    model_compress_enc_time = compress_out_to_use['enc_time']
    model_compress_dec_time = compress_out_to_use['dec_time']
    # model_compress_enc_time = 0
    # model_compress_dec_time = 0
    enc_time += model_compress_enc_time
    dec_time += model_compress_dec_time
    
    model_bytes = compress_out_to_use['final_bytes']
    model_bin = os.path.join(bins_dir, "model.bin")
    
    if write_flag:
        with open(model_bin, 'wb') as f:
            f.write(model_bytes)
    side_info = {'mu': compress_out_to_use['mu'].item(), 'b': compress_out_to_use['b'].item(), 'min_param': compress_out_to_use['min_param'].item(), 'max_param': compress_out_to_use['max_param'].item(),'enc_mode': enc_mode_to_use}
    # side_info = {}
    side_info_path = os.path.join(result_dir, "side_info.json")
    # if write_flag:
    
    side_info['xlow_enc_flags'] = ','.join([str(f) for f in xlow_enc_flags])
    side_info['xlow_enc_modes'] = ','.join([str(m) for m in xlow_enc_modes])
    
    with open(side_info_path, 'w') as f:
        json.dump(side_info, f, indent=4)
        
    
    
    reading_data = inargs['reading_data']
    
    bits_test_frame = 0
    bits_train_frame = 0
    
    point_test_frame = 0
    xyzlow_test_frame = 0
    for frame_idx in range(frame_num):
        frame_data = reading_data[frame_idx]
            
        all_input_info = frame_data['all_input_info']
        point_num = frame_data['point_num']
        
        handle_out = test_one_frame(model_to_use, all_input_info)
        bits_t_tmp = handle_out['all_bit']
        bits_train_tmp = handle_out['all_bit_t']
        bits_train_frame += bits_train_tmp
        
        all_bytes = handle_out['all_bytes']
        
        enc_time += handle_out['enc_time']
        dec_time += handle_out['dec_time']
        
        if write_flag:
            write_bin_file(frame_idx, all_bytes, bins_dir)
        
        # bpp_t_tmp = bits_t_tmp / point_num
        bits_test_frame += bits_t_tmp
        point_test_frame += point_num

        # frame_xyzlow_ret = low_enc_ret[frame_idx]
        # xyzlow_test_frame += frame_xyzlow_ret['bits']
        # xyzlow_test_frame += frame_data['xyzQ_low_bits']
    
        torch.cuda.empty_cache()
    xyzlow_test_frame = len(low_enc_ret)*8
    point_bpp = bits_test_frame / point_test_frame
    point_bpp_val = bits_train_frame / point_test_frame
    # bpp_t和bpp_train应该是差不多的
    
    model_bpp = modelbit_use / point_test_frame
    

    xyzlow_bpp = xyzlow_test_frame / point_test_frame
    
    bpp_all = point_bpp + model_bpp + xyzlow_bpp
    
    result = {'bpp_all': bpp_all, 'point_bpp': point_bpp, 'point_bpp_val': point_bpp_val.item() ,'model_bpp': model_bpp, 'xyzlow_bpp': xyzlow_bpp, 'enc_mode': enc_mode_to_use, 'enc_time': enc_time/frame_num, 'dec_time': dec_time/frame_num}
    
    result_path = os.path.join(result_dir, "result.json")
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=4)
        
    return result


def test_one_frame(model, all_inargs):
    # TODO: 这个地方要根据不同模型的变化而变化
    all_bit = 0
    all_bytes = []
    enc_time = 0
    dec_time = 0
    
    all_bit_t = 0
    for s_idx, inargs in enumerate(all_inargs):
        putin_args = {}
        putin_args.update(inargs)
        xyzqsc_t = inargs['xyzqsc_t']
        coord = xyzqsc_t.get_coord()
        offset_tensor = xyzqsc_t.get_offset_tensor()
        putin_args['offset_tensor'] = offset_tensor
        
        putin_args['coord'] = coord

        ret_out = model.codec(putin_args)
        
        scale_enc_byte = ret_out['enc_bytes']
        scale_bit = ret_out['bits']
        
        all_bit_t += ret_out['bits_t']
        enc_time += ret_out['enc_time']
        dec_time += ret_out['dec_time']
        all_bit += scale_bit
        all_bytes.append(scale_enc_byte)
    return {'all_bit': all_bit,'all_bit_t': all_bit_t, 'all_bytes': all_bytes, 'enc_time': enc_time, 'dec_time': dec_time}




def enc_all_frame_low_xyz(reading_data,frame_num):
    all_xlow_info = []
    all_coord_data_min = []
    for frame_idx in range(frame_num):
        frame_data = reading_data[frame_idx]
        all_input_info = frame_data['all_input_info']

        coord_data_min = frame_data['coord_data_min']
        all_coord_data_min.append(coord_data_min)
        
        xyzlow = all_input_info[-1]['xyzqsc_t']

        xyzQ = xyzlow.get_coord()

        max_data = xyzQ.max().detach().cpu().numpy()

        # 表示一个数需要的bit数
        bitdepthQ = int(np.ceil(np.log2(max_data+1)))
        
        # 所有数都是正数，负数默认已经处理了
        xyzQ_enc = xyzQ.detach().cpu().numpy()

        assert bitdepthQ <= 8, 'downsampled xyzQ should be less than 8 bit'
        xyzQ_enc = xyzQ_enc.astype(np.uint8)

        xyz_enc_bytes = xyzQ_enc.tobytes()
        all_xlow_info.append(xyz_enc_bytes)
    all_coord_data_min = np.concatenate(all_coord_data_min, axis=0).astype(np.int32)
    all_coord_data_min_byte = all_coord_data_min.tobytes()
    all_xlow_info.append(all_coord_data_min_byte)
    low_byte = pack_bitstream(all_xlow_info)
    
    
    return low_byte
            

def part_xyz_low_detail(reading_data,frame_num):
    xyz_low_result = []
    for frame_idx in range(frame_num):
        
        frame_data = reading_data[frame_idx]
        all_input_info = frame_data['all_input_info']
        # st1_2 = time.time()
        xyzlow = all_input_info[-1]['xyzqsc_t']
        # st1_3 = time.time()
        xyzQ = xyzlow.get_coord()
        # st1_4 = time.time()
        max_data = xyzQ.max().detach().cpu().numpy()
        # st1_5 = time.time()
        # 表示一个数需要的bit数
        bitdepthQ = int(np.ceil(np.log2(max_data+1)))
        # 这个空间最多表示多少个点
        max_point_num = (2**bitdepthQ)**3
        
        # 数据有多少点
        point_num = xyzQ.shape[0]
        
        # enc_point

        

        
        xyz_byte = xyzQ.detach().cpu().numpy().tobytes()
        xyz_bits = len(xyz_byte)*8
        
        xyz_low_result.append({'xyz_bits': xyz_bits, 'xyz_num': point_num, 'xyz_inv_num': max_point_num-point_num})
        
    return xyz_low_result


def dec_all_frame_low_xyz(low_byte):
    all_xlow_info = unpack_bitstream(low_byte)
    all_coord_data_min_byte = all_xlow_info.pop()
    all_coord_data_min = np.frombuffer(all_coord_data_min_byte, dtype=np.int32)
    all_coord_data_min = all_coord_data_min.reshape(-1, 3)
    all_coord_data_min = torch.tensor(all_coord_data_min, device='cuda',dtype=torch.int32)
    dtyep = np.uint8
    all_xyzQ = []
    for frame_info in all_xlow_info:
        enc_bytes = frame_info
        xyzQ_enc = np.frombuffer(enc_bytes, dtype=dtyep)
        xyzQ = xyzQ_enc.reshape(-1, 3)
        all_xyzQ.append(xyzQ)
    return {'all_xyz_low': all_xyzQ, 'all_coord_data_min': all_coord_data_min}

    