import argparse
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import shutil
from datautils.custom_dataset import MyDataset, Read_Data, MytestDataset
import torch
import logging

from models.model_core import LINR_PCGC_Model
from test_utils import enc_all_frame_low_xyz, Test_one_gop
from model_compression.model_size_est import Model_Estimate, esti_model_size
import numpy as np
import time
import json
import random
from encoder import encode, encode_one_gop
from decoder import decode, decode_one_gop

esti_compress_model = Model_Estimate().estibits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

offsets_ini = [[0,0,0],[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]]
offset_of_neigbor = [[0,0,0],[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]]
offsets_ini = torch.tensor(offsets_ini).to(device)
offset_of_neigbor = torch.tensor(offset_of_neigbor, dtype=torch.float32).to(device)

def set_logger(logpath):
    # 创建日志记录器
    logger = logging.getLogger("simple_logger")
    logger.setLevel(logging.INFO)

    # 创建文件处理器
    file_handler = logging.FileHandler(logpath, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    # 设置简单的格式，只输出消息内容
    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)

    # 添加处理器到记录器
    logger.addHandler(file_handler)
    return logger



def setup_exp(args):
    ori_dir = args.ori_dir
    handle_dir = args.handle_dir
    output_dir = args.result_dir
    if os.path.exists(handle_dir):
        shutil.rmtree(handle_dir)
    
    if not os.path.exists(handle_dir):
        os.makedirs(handle_dir)
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    


def overfit_enc_dec(args):
    ori_dir = args.ori_dir
    handle_dir = args.handle_dir
    
    # construct dataset
    dataset = MyDataset(ori_dir, handle_dir, args.scale_num, args.ori_dtype, stage=8)
    dataset.set_prefix_data({'offsets_ini':offsets_ini, 'offset_of_neigbor':offset_of_neigbor,'min_point_num':args.min_point_num})
    # let dataset derive a frame data, to get the real scale_num
    dataset[0]
    args.scale_num = dataset.scale_num
    print('scale_num:', args.scale_num)
    
    all_frame_num = args.frame_num
    # # 将0,frame-1均分成若干长度args.gop_size的帧
    gop_size = args.gop_size
    all_groups = []
    for i in range(0, all_frame_num, gop_size):
        all_groups.append(list(range(i, min(i+gop_size, all_frame_num))))
    
    gop_names = [f'gop_{group[0]}_{group[-1]}' for group in all_groups]
    
    next_model_path = None
    
    
    if not os.path.exists(str(args.pretrain_path)):
        args.pretrain_path = None
    
    
    Gen_Model = lambda: LINR_PCGC_Model({'scale_num':args.scale_num, 'in_channel':len(offsets_ini), 'hidden_channel_conv':args.hidden_channel_conv, 'block_layers':args.block_layers, 'outstage':8, 'instage':1}).cuda()
    if args.overfit == 'True':
        for g_idx, group_range in enumerate(all_groups):
            # group_range = range(32,63)
            if g_idx == 0:
                last_model_pth = args.pretrain_path
                next_model_path = overfit_one_gop(args, dataset, group_range, args.first_epoch, last_model_pth)
            else:
                last_model_pth = next_model_path
                next_model_path = overfit_one_gop(args, dataset, group_range, args.others_epoch, last_model_pth)
            
    

    dataset_test = MytestDataset(ori_dir, ori_type='ply')
    if args.encode == 'True':
        enc_args = {'outputdir': args.result_dir, 'gop_names': gop_names, 'Gen_Model': Gen_Model, 'dataset': dataset_test, 'encode_dir': args.encode_dir}
        encode(enc_args)
    
    if args.decode == 'True':
        dec_args = {'gop_names': gop_names, 'Gen_Model': Gen_Model, 'result_enc_dir': args.encode_dir,'result_dec_dir': args.decode_dir,'dataset':dataset_test, 'write_flag': True}
        decode(dec_args)
    
    
    if args.delete_cache == 'True':
        shutil.rmtree(handle_dir)
    
    
    
    
def overfit_one_gop(args, dataset, group_range, epoch_num, last_model_pth):
     
    print(f'process_file: {group_range[0]} {group_range[-1]}')
    logger.info("="*40)
    logger.info(f'process_file: {group_range[0]} {group_range[-1]}')
    
    
    gop_result_dir = f'{args.result_dir}/gop_{group_range[0]}_{group_range[-1]}'
    if not os.path.exists(gop_result_dir):
        os.makedirs(gop_result_dir)
    
    json_ret_dir = os.path.join(gop_result_dir, 'result.json')    
    
    model_path = os.path.join(gop_result_dir, 'model.pth')
    

    # epoch_num = args.epoch_num
    step_size = args.step_size
    scale_num = args.scale_num
    gamma = args.gamma
    model_bitdepth = args.model_bitdepth
    min_lr = args.min_lr
    write_real_bitstream = args.write_real_bitstream == 'True'
    write_pth = args.write_pth == 'True'
    mid_test = args.mid_test == 'True'
    
    gop_size = len(group_range)
    # frame_num = gop_size


    reading_data = Read_Data(dataset, group_range)
    low_enc_ret = enc_all_frame_low_xyz(reading_data, frame_num=gop_size)
    
    Gen_Model = lambda: LINR_PCGC_Model({'scale_num':scale_num, 'in_channel':len(offsets_ini), 'hidden_channel_conv':args.hidden_channel_conv, 'block_layers':args.block_layers, 'outstage':8, 'instage':1}).cuda()
    
    
    estd_model = Gen_Model()
    # MinkowskiEngine can not be deepcopied, so we need to reinitialize it
    model_ori = Gen_Model()

    # model_size = esti_model_size(estd_model)
    # print('model_size:', model_size)
    params = torch.cat([p.view(-1) for p in estd_model.parameters()])
    param_num = len(params)
    print('DBG!!! model size:\t', param_num)

    optimizer = torch.optim.Adam(
        estd_model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )

    # optimizer.add_param_group({'params': embed_obj.parameters()})

    if last_model_pth is not None:
        if os.path.exists(last_model_pth) and os.path.isfile(last_model_pth):
            ckpt = torch.load(last_model_pth)
            estd_model.load_state_dict(ckpt['model'])
            opt_state = ckpt['optimizer_state_dict']
            optimizer.load_state_dict(opt_state)

            print('load pretrain model:', last_model_pth)

    start_epoch = 0
    best_loss_test = 99999999
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # 先test一下，有错误提前报错
    # print('Try to test .....')
    # if not os.path.exists(model_path) or not load_flag:
    #     torch.save(
    #         {
    #             'model':estd_model.state_dict(),
    #             'embed_obj':embed_obj.state_dict(),
    #             'epoch':start_epoch,
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'loss': best_loss_test
    #         }, 
    #         model_path
    #     )
    # real_out = Test_real_2({'model_path':model_path, 'estd_model':estd_model, 'esti_compress_model':esti_compress_model,'model_ori':model_ori, 'frame_num':gop_size, 'compress_model_test':Model_Estimate().compress_test, 'reading_data':reading_data,'result_dir':f'{gop_result_dir}/0','write_flag':False,'low_enc_ret':low_enc_ret,'embed_obj':embed_obj, 'handle_inarg_test':handle_inarg_test})
    # real_bpp_all = real_out['bpp_all']
    # enc_time = real_out['enc_time']
    # dec_time = real_out['dec_time']
    # print('real_bpp_all:', real_bpp_all)
    # print('enc_time:', enc_time)
    # print('dec_time:', dec_time)
    
    estd_model.train()
    
    train_time = 0
    bitdepths = np.arange(5.5,8.1,0.5)
    # bitdepths = np.arange(8,31,5).astype(np.float64)
    check_frame = gop_size//len(bitdepths)
    if check_frame <= 0:
        check_frame = 1
    samplefactor = check_frame / gop_size
    check_frames = random.sample(range(gop_size), check_frame)
    check_frames.sort()
    
    
    result_out = []
    bitdepth_use = model_bitdepth
    for epoch in range(start_epoch, epoch_num):
        estd_model.train()
        st1 = time.time()
        loss_sum = 0
        # 从0, frame_num-1中随机取16个数
        # tr_idxs = random.sample(range(frame_num), sample_num)
        
        
        for frame_idx in range(gop_size):
            # try:
            
            frame_data = reading_data[frame_idx]
            
            all_input_info = frame_data['all_input_info']
            point_num = frame_data['point_num']
            
            
            bits = overfit_one_frame(estd_model, all_input_info)
            loss = bits / point_num
            loss.backward(retain_graph=True)                
            
            loss_sum += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            torch.cuda.empty_cache()

            
                
            
        st2 = time.time()
        train_time += st2 - st1
        train_time_avg = train_time / gop_size
        loss_mean = loss_sum / gop_size
        print(f'epoch: {epoch}')
        
        # print(f'frame: {epoch*framenum+i}')
        print('loss:', loss_mean)
        
        # print('train_time:', train_time)  
        print('train_time_avg:', train_time_avg)
        
        
        logger.info(f'epoch: {epoch}')
        logger.info(f'loss: {loss_mean}')
        logger.info(f'train_time: {train_time}')
        logger.info(f'train_time_avg: {train_time_avg}')
        
        epoch_result = {'epoch':epoch, 'loss':loss_mean, 'train_time':train_time, 'train_time_avg':train_time_avg}
        
        
        if (mid_test) and (epoch < 10 or epoch%args.check_freq==0):
        # if epoch%args.check_freq==0 and epoch >=10:
            # emb_param = torch.cat([p.view(-1) for p in embed_obj.parameters()])
            # emb_byte = emb_param.detach().cpu().numpy().tobytes()
            # emb_zlib = zlib.compress(emb_byte)
            # emb_flag = 0 if len(emb_byte) < len(emb_zlib) else 1
            # if emb_flag == 0:
            #     emb_bits = len(emb_byte)*8
            # else:
            #     emb_bits = len(emb_zlib)*8
            
            emb_bits = 0
            
            bitdepth_final = 8
            # bpps_all_depth = []
            bpp_t = None
            model_bpp = None
            

            

            
            with torch.no_grad():
                estd_model.eval()
                bpp_check_final = 99999999
                # 随机取0,frame_num-1中的check_frame个不同的数
                assert check_frame <= gop_size

                
                
                # 因为压缩后的model实际上改变的是model_ori的参数，所以已经被覆盖了，这里需要重新计算
                # 这是由于含有ME的model无法deepcopy
                compress_out_to_use = esti_compress_model(estd_model, model_ori, bitdepth_use)
                model_to_use = compress_out_to_use['new_model']
                modelbit_use = compress_out_to_use['bit_real']
                enc_mode_to_use = compress_out_to_use['enc_mode']
                
                
                
                        
                bits_test_frame = 0
                point_test_frame = 0
                xyzlow_test_frame = 0
                for frame_idx in range(gop_size):
                    frame_data = reading_data[frame_idx]
            
                    all_input_info = frame_data['all_input_info']
                    point_num = frame_data['point_num']
                    
                    
                    bits_t_tmp = overfit_one_frame(model_to_use, all_input_info)
                
                    # bpp_t_tmp = bits_t_tmp / point_num
                    bits_test_frame += bits_t_tmp.item()
                    point_test_frame += point_num

                    xyzlow_test_frame += frame_data['xyzQ_low_bits']
                
                    torch.cuda.empty_cache()
                bpp_t = bits_test_frame / point_test_frame
                model_bpp = modelbit_use / point_test_frame
                
                emb_bpp = emb_bits / point_test_frame
                xyzlow_bpp = xyzlow_test_frame / point_test_frame
                
                bpp_all = bpp_t + model_bpp + emb_bpp + xyzlow_bpp
                
            print('bpp_t:', bpp_t)
            print('model_bpp:', model_bpp)
            print('emb_bpp:', emb_bpp)
            print('xyzlow_bpp:', xyzlow_bpp)
            print('bpp_all:', bpp_all)
            print('enc_mode:', enc_mode_to_use)

            if bpp_all < best_loss_test and write_pth:
                best_loss_test = bpp_all
                torch.save(
                    {
                        'model':estd_model.state_dict(),
                        'epoch':epoch,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss_test,
                        'bitdepth': bitdepth_use
                    }, 
                    model_path
                )

                if epoch % 50 == 0 and write_real_bitstream:
                    real_out = Test_one_gop({'model_path':model_path,'Gen_Model':Gen_Model, 'frame_num':gop_size, 'compress_model_test':Model_Estimate().compress_test, 'reading_data':reading_data,'result_dir':f'{gop_result_dir}/{epoch}','write_flag':True,'low_enc_ret':low_enc_ret})
                else:
                    real_out = Test_one_gop({'model_path':model_path, 'esti_compress_model':esti_compress_model,'Gen_Model':Gen_Model, 'frame_num':gop_size, 'compress_model_test':Model_Estimate().compress_test, 'reading_data':reading_data,'result_dir':f'{gop_result_dir}/{epoch}','write_flag':False,'low_enc_ret':low_enc_ret})
                real_bpp_all = real_out['bpp_all']
                enc_time = real_out['enc_time']
                dec_time = real_out['dec_time']
                
                
            print('real_bpp_all:', real_bpp_all)
            print('fake_bpp_all:', bpp_all)
            print('train_time:', train_time)
            print('enc_time:', enc_time)
            print('dec_time:', dec_time)
            
            
            
            logger.info(f'real_bpp_all: {real_bpp_all}')
            logger.info(f'fake_bpp_all: {bpp_all}')
            logger.info(f'bpp_t: {bpp_t}')
            logger.info(f'model_bpp: {model_bpp}')
            logger.info(f'emb_bpp: {emb_bpp}')
            logger.info(f'xyzlow_bpp: {xyzlow_bpp}')
            
            logger.info(f'enc_time: {enc_time}')
            logger.info(f'dec_time: {dec_time}')
            logger.info(f'enc_mode: {enc_mode_to_use}')
            logger.info(f'model_bitdepth_final: {bitdepth_use}')
            
            
            epoch_result['real_bpp_all'] = real_bpp_all
            epoch_result['fake_bpp_all'] = bpp_all
            epoch_result['bpp_t'] = bpp_t
            epoch_result['model_bpp'] = model_bpp
            epoch_result['emb_bpp'] = emb_bpp
            epoch_result['xyzlow_bpp'] = xyzlow_bpp
            
            epoch_result['enc_time'] = enc_time
            epoch_result['dec_time'] = dec_time
            epoch_result['enc_mode'] = enc_mode_to_use
            epoch_result['model_bitdepth_final'] = bitdepth_use
        
        else:
            # if epoch % 5 == 0:
            if (loss_mean < best_loss_test) and write_pth:
                best_loss_test = loss_mean
                torch.save(
                    {
                        'model':estd_model.state_dict(),
                        'epoch':epoch,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss_test,
                        'bitdepth': bitdepth_use
                    }, 
                    model_path
                )
        
        result_out.append(epoch_result)
        with open(json_ret_dir, 'w') as f:
            json.dump(result_out, f, indent=4)
        
        logger.info('')
        for param_group in optimizer.param_groups:
            
            if param_group['lr'] < min_lr:
                param_group['lr'] = min_lr
            print('current lr:', param_group['lr'])
    logger.info('')

    if loss_mean < best_loss_test and write_pth:
        best_loss_test = loss_mean
        torch.save(
            {
                'model':estd_model.state_dict(),
                'epoch':epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss_test,
                'bitdepth': bitdepth_use
            }, 
            model_path
        )
    return model_path




def overfit_one_frame(model, all_inargs):
    all_bit = 0
    # time_embed = embed_args['time']
    for s_idx, inargs in enumerate(all_inargs):
        # scale_embed_info = embed_args['pos'][s_idx]
        putin_args = {}
        putin_args.update(inargs)
        
        xyzqsc_t = inargs['xyzqsc_t']
        # pos_embed = inargs['pos_embed']
        # sparse_x = xyzqsc_t.get_offset_sparse()
        coord = xyzqsc_t.get_coord()
        putin_args['coord'] = coord
        offset_tensor = xyzqsc_t.get_offset_tensor()
        putin_args['offset_tensor'] = offset_tensor

        scale_bit = model(putin_args)
        all_bit += scale_bit
    return all_bit



if __name__ == '__main__':
    parser = argparse.ArgumentParser('LINR-PCGC')


    parser.add_argument('--others_epoch', default=10, type=int)
    parser.add_argument('--first_epoch', default=10, type=int)
    
    parser.add_argument('--gop_size', type=int, default=32, help='batch size in training [default: 24]')
    parser.add_argument('--frame_num', type=int, default=96, help='repeat number')


    parser.add_argument('--learning_rate', default=0.01, type=float,
                        help='learning rate in training [default: 0.001]')
    parser.add_argument('--gamma', type=float, default=0.992, help='gamma')
    parser.add_argument('--min_lr', type=float, default=4e-4, help='min learning rate')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--step_size', type=int, default=32, help='step size')


    parser.add_argument('--scale_num', type=int, help='scale number')
    parser.add_argument('--min_point_num', type=int, default=64, help='min point number')


    parser.add_argument('--load', default='False', type=str)
    parser.add_argument('--pretrain_path', type=str)
    parser.add_argument('--write_pth', type=str, default='True')
    parser.add_argument('--seed', type=int, default=8807)
    parser.add_argument('--delete_cache', type=str, default='False')
    parser.add_argument('--write_real_bitstream', type=str, default='False')
    parser.add_argument('--check_freq', type=int, default=5)


    parser.add_argument('--ori_dir', type=str, default='/home/huangwenjie/pythonfiles/LINR_PCGC/dataset/8iVFB/8iVFBv2/loot')
    parser.add_argument('--ori_dtype', type=str, default='ply')
    parser.add_argument('--handle_dir', type=str, default='tmp/loot')


    parser.add_argument('--model_path', type=str,default=None)
    parser.add_argument('--result_dir', type=str, default='output/loot')

    parser.add_argument('--hidden_channel_mlp', type=int, default=24)
    parser.add_argument('--mlp_out_channel', type=int, default=10)
    parser.add_argument('--hidden_channel_conv', type=int, default=8)
    parser.add_argument('--block_layers', type=int, default=1)
    parser.add_argument('--model_bitdepth', type=int, default=8)

    parser.add_argument('--overfit', type=str, default='True')
    parser.add_argument('--mid_test', type=str, default='True')
    
    parser.add_argument('--encode', type=str, default='False')
    parser.add_argument('--encode_dir', type=str, default='result_enc/loot')
    parser.add_argument('--decode', type=str, default='True')
    parser.add_argument('--decode_dir', type=str, default='result_dec/loot')
    


    args_ori = parser.parse_args()
    print(args_ori)
    
    setup_exp(args_ori)
    logpath = os.path.join(args_ori.result_dir, f'info.log')
    logger = set_logger(logpath)
    
    overfit_enc_dec(args_ori)
