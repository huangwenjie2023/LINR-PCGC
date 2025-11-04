import torchac
import torch
import zlib
import numpy as np

import torchac
import time
from scipy.stats import laplace

def mylaplace_pdf(x, mu, b):
    return torch.exp(-torch.abs(x-mu)/b) / (2*b)

def mynormal_pdf(x, mu, std):
    return torch.exp(-0.5*((x-mu)/std)**2) / (std * (2*np.pi)**0.5)

def models_are_equal(model1, model2):
    # 比较结构
    if type(model1) != type(model2):
        return False

    # 比较参数
    all_err = 0
    num_cout = 0
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        all_err += torch.sum(torch.abs(param1.data - param2.data))
        num_cout += torch.numel(param1.data)
    avg_err = all_err / num_cout
    return avg_err

def esti_model_size(model):
    
    # 提取并展平模型参数
    params = torch.cat([p.view(-1) for p in model.parameters()])
    # 计算bit数
    bit = len(params) * 32
    # print(bit)
    return bit

class Model_Estimate:
    def __init__(self):
        pass
    
    def quant_uniform(self, param_lst: torch.Tensor, bitdepth: int = 8):
        min_n = param_lst.min()
        max_n = param_lst.max()
        ten_range = max_n - min_n
        
        
        # 区间长度归一化 加 量化
        new_p = torch.round(param_lst / ten_range * (2**bitdepth-1))
        
        new_p_min = new_p.min()
        
        est_new_p_min = torch.round(min_n / ten_range * (2**bitdepth-1))
        
        assert new_p_min == est_new_p_min
        # 正值化
        new_p = new_p - new_p_min
        
        assert new_p.min() >= 0 and new_p.max() <= np.ceil(2**bitdepth-1)
        
        recon_new_p = new_p
        
        # 反正值化
        recon_new_p = recon_new_p + new_p_min
        
        # 反量化 反区间长度归一化
        recon_param_lst = recon_new_p / (2**bitdepth-1) * ten_range
        
        return new_p, recon_param_lst
    
    def quant_uniform2(self, param_lst: torch.Tensor, bitdepth: int = 8):
        min_n = param_lst.min()
        max_n = param_lst.max()
        ten_range = max_n - min_n
        
        sym_max_num = np.ceil(2**bitdepth) - 1
        # 区间长度归一化 加 量化
        new_p = torch.round((param_lst-min_n) / ten_range * sym_max_num)
        
        assert new_p.min() >= 0 and new_p.max() <= sym_max_num
        
        recon_new_p = new_p
        
        # 反正值化
        # recon_new_p = recon_new_p + new_p_min
        
        # 反量化 反区间长度归一化
        recon_param_lst = recon_new_p / sym_max_num * ten_range + min_n
        
        return new_p, recon_param_lst
    
    
    
    
    def quant_non_laplace(self, param_tensor):
        pass
    
    def estibits(self, model, new_model, bitdepth: int = 8):
        st1 = time.time()
        # 编解码都有
        params_ten = torch.cat([p.view(-1) for p in model.parameters()])
        min_param = params_ten.min()
        max_param = params_ten.max()
        quant_ret, recon_ret = self.quant_uniform2(params_ten, bitdepth) 
        
        
        # 只有解码端有
        current_index = 0
        for p in new_model.parameters():
            num_elements = p.numel()  # 获取参数的元素个数
            new_values = recon_ret[current_index:current_index + num_elements].view_as(p)
            p.data.copy_(new_values)  # 将重构的参数赋值回模型
            current_index += num_elements


        # 验证重构的参数
        err = models_are_equal(model, new_model)
        # print('err:', err)
        

        # 估计laplace分布
        mu = torch.round(quant_ret.mean())
        b = torch.round((quant_ret-mu).abs().mean())
        
        # std = torch.std(quant_ret)
        
        likelihood_laplace = mylaplace_pdf(quant_ret, mu, b)
        # likelihood_normal = mynormal_pdf(quant_ret, mu, std)
        
        # print('mu:', mu)
        # print('b:', b)
        
        # 估计模型大小
        bits_laplace = -torch.sum(torch.log2(likelihood_laplace))
        # bits_normal = -torch.sum(torch.log2(likelihood_normal))
        
        bits_laplace = bits_laplace.item()
        # bits_normal = bits_normal.item()
        enc_mode = 2
        bits = bits_laplace
        # print('laplace')

        
        # bits还需要加上 最大值,最小值, mu, b/std的大小
        bits = bits + 2*bitdepth
        
        bpp = bits / len(quant_ret)
        
        
        np_type = np.uint8
        if bitdepth > 8:
            np_type = np.uint16
        if bitdepth > 16:
            np_type = np.uint32
        # 计算下界
        quant_numpy = quant_ret.detach().cpu().numpy().astype(np_type)
        quant_byte = quant_numpy.tobytes()
        quant_zlib = zlib.compress(quant_byte)
        bit_zlib_bound = len(quant_zlib) * 8
        bpp_zlib_bound = bit_zlib_bound / len(quant_ret)
        
        bpp_low_bound = bpp_zlib_bound if bpp_zlib_bound < bitdepth else bitdepth
        
        # 带上两个标志位 00直接编 01带zlib 编 10 laplace编  11正态分布编
        bit_real = bits + 2 + 2*32 
        if bpp > bpp_low_bound:
            # print('Warning: AE bpp is larger than lowbound')
            enc_mode = 1
            if bpp_low_bound == bitdepth:
                enc_mode = 0
                # print('zlib bpp is larger than bitdepth')
            bit_real = bpp_low_bound*len(quant_ret) + 2
        bpp_real = bit_real / len(quant_ret)
        st2 = time.time()
        enc_time = st2 - st1
        dec_time = st2 - st1
        final_byte = b'0'
        return {'new_model': new_model, 'bpp_real': bpp_real, 'bit_real': bit_real, 'enc_mode': enc_mode, 'laplace_bpp': bits_laplace / len(quant_ret), 'zlib_bpp': bpp_zlib_bound, 'final_bytes': final_byte, 'min_param': min_param, 'max_param': max_param, 'mu': mu, 'b': b, 'enc_time': enc_time, 'dec_time': dec_time, 'recon_ret': recon_ret}
    
    def codec(self, model, new_model, bitdepth: int = 8):
        params_ten = torch.cat([p.view(-1) for p in model.parameters()])
        min_param = params_ten.min()
        max_param = params_ten.max()
        quant_ret, recon_ret = self.quant_uniform2(params_ten, bitdepth) 
        
        current_index = 0
        for p in new_model.parameters():
            num_elements = p.numel()  # 获取参数的元素个数
            new_values = recon_ret[current_index:current_index + num_elements].view_as(p)
            p.data.copy_(new_values)  # 将重构的参数赋值回模型
            current_index += num_elements

        # 验证重构的参数
        # for p in model.parameters():
        #     print(p.data)
        err = models_are_equal(model, new_model)
        # print('err:', err)
        # print('relative error:', err / (max_param - min_param))
        
        
        # 估计laplace分布
        mu = torch.round(quant_ret.mean())
        b = torch.round((quant_ret-mu).abs().mean())
        
        # std = torch.std(quant_ret)
        
        likelihood_laplace = mylaplace_pdf(quant_ret, mu, b)
        # likelihood_normal = mynormal_pdf(quant_ret, mu, std)
        
        # print('mu:', mu)
        # print('b:', b)
        
        # 估计模型大小
        bits_laplace = -torch.sum(torch.log2(likelihood_laplace))
        # bits_normal = -torch.sum(torch.log2(likelihood_normal))
        
        bits_laplace = bits_laplace.item()


        enc_mode = 2
        bits = bits_laplace

        
        # bits还需要加上 最大值,最小值, mu, b/std的大小
        bits = bits + 2*bitdepth
        
        bpp = bits / len(quant_ret)
        
        
        np_type = np.uint8
        if bitdepth > 8:
            np_type = np.uint16
        if bitdepth > 16:
            np_type = np.uint32
        # 计算下界
        quant_numpy = quant_ret.detach().cpu().numpy().astype(np_type)
        quant_byte = quant_numpy.tobytes()
        quant_zlib = zlib.compress(quant_byte)
        bit_zlib_bound = len(quant_zlib) * 8
        bpp_zlib_bound = bit_zlib_bound / len(quant_ret)
        
        bpp_low_bound = bpp_zlib_bound if bpp_zlib_bound < bitdepth else bitdepth
        
        # 带上两个标志位 00直接编 01带zlib 编 10 laplace编  11正态分布编
        # 2是标志位 2*32是归一化参数
        bit_real = bits + 2 + 2*32
        side_info_bit = 2+2*32
        
        bit_laplace_real = 999999999999
        if bpp > bpp_low_bound:
            # print('Warning: AE bpp is larger than lowbound')
            enc_mode = 1
            final_bytes = quant_zlib
            if bpp_low_bound == bitdepth:
                enc_mode = 0
                final_bytes = quant_byte
                # print('zlib bpp is larger than bitdepth')
            bit_real = bpp_low_bound*len(quant_ret) + 2 + 2*32

        else:

            sym_tensor = quant_ret.to(torch.int16)

            pdf_sample = mylaplace_pdf(torch.arange(np.ceil(2**bitdepth),device=mu.device), torch.round(mu), torch.round(b))
            pdf_sample = pdf_sample / pdf_sample.sum()
            
            cdf_sample = torch.cumsum(pdf_sample, dim=-1)
            cdf_sample = cdf_sample.to(torch.float32)
            # cdf_quantized = (cdf_sample * (2**15 - 1)).round()
            cdf_quantized = cdf_sample

            # cdf_quantized = cdf_quantized.to(torch.int16)
            # cdf_quantized = torch.cat([cdf_quantized, torch.tensor([0], dtype=torch.int16)])
            cdf_quantized = torch.cat([cdf_quantized, torch.tensor([0], dtype=torch.float32,device=mu.device)])
            # 扩展CDF
            cdf_expanded = cdf_quantized.repeat(len(sym_tensor), 1).detach().cpu()

            sym_to_enc = sym_tensor.detach().cpu()
            encoded_bytes = torchac.encode_float_cdf(cdf_expanded, sym_to_enc)

            # 解码
            decoded_sym = torchac.decode_float_cdf(cdf_expanded, encoded_bytes)

            # 验证解码结果
            assert torch.equal(sym_to_enc, decoded_sym), "解码后的符号与原始符号不匹配！"
            # enc_bpp = (len(encoded_bytes) * 8+2*bitdepth) / len(sym_tensor)
            # print('enc_bpp:', enc_bpp)
            # print('finish')
            
            bit_laplace_real = len(encoded_bytes) * 8 + 2*np.ceil(bitdepth) + 2 + 2*32
            
            # 还有机会再选择zlib
            if bit_laplace_real > bpp_low_bound*len(quant_ret) + 2 + 2*32:
                # print('Warning: AE bpp is larger than lowbound')
                enc_mode = 1
                final_bytes = quant_zlib
                if bpp_low_bound == bitdepth:
                    enc_mode = 0
                    final_bytes = quant_byte
                    # print('zlib bpp is larger than bitdepth')
                bit_real = bpp_low_bound*len(quant_ret) + 2 + 2*32
                
            else:
                bit_real = bit_laplace_real
                final_bytes = encoded_bytes
                side_info_bit = + 2*np.ceil(bitdepth) + 2 + 2*32
                
        bpp_real = bit_real / len(quant_ret)
 
        

        return {'new_model': new_model, 'bpp_real': bpp_real, 'bit_real': bit_real, 'side_info_bit': side_info_bit,'enc_mode': enc_mode, 'laplace_real_bpp':  bit_laplace_real / len(quant_ret), 'zlib_bpp': bpp_zlib_bound, 'min_param': min_param, 'max_param': max_param, 'mu': mu, 'b': b, 'final_bytes': final_bytes}
    
    
    @torch.no_grad()
    def compare_methods(self, model, new_model, bitdepth: int = 8):
        params_ten = torch.cat([p.view(-1) for p in model.parameters()])
        
        bit_ori = params_ten.numel() * 32
        
        min_param = params_ten.min()
        max_param = params_ten.max()
        quant_ret, recon_ret = self.quant_uniform2(params_ten, bitdepth) 
        
        current_index = 0
        for p in new_model.parameters():
            num_elements = p.numel()  # 获取参数的元素个数
            new_values = recon_ret[current_index:current_index + num_elements].view_as(p)
            p.data.copy_(new_values)  # 将重构的参数赋值回模型
            current_index += num_elements
        mu = torch.round(quant_ret.mean())
        b = torch.round((quant_ret-mu).abs().mean())
        
        np_type = np.uint8
        if bitdepth > 8:
            np_type = np.uint16
        if bitdepth > 16:
            np_type = np.uint32
            
        quant_numpy = quant_ret.detach().cpu().numpy().astype(np_type)
        quant_byte = quant_numpy.tobytes()
        quant_zlib = zlib.compress(quant_byte)
        
        
        bit_tobyte = len(quant_byte) * 8 + 2 + 2*32
        bit_zlib = len(quant_zlib) * 8 + 2 + 2*32

        
        
        sym_tensor = quant_ret.to(torch.int16)

        pdf_sample = mylaplace_pdf(torch.arange(np.ceil(2**bitdepth),device=mu.device), torch.round(mu), torch.round(b))
        pdf_sample = pdf_sample / pdf_sample.sum()
        
        cdf_sample = torch.cumsum(pdf_sample, dim=-1)
        cdf_sample = cdf_sample.to(torch.float32)
        # cdf_quantized = (cdf_sample * (2**15 - 1)).round()
        cdf_quantized = cdf_sample

        # cdf_quantized = cdf_quantized.to(torch.int16)
        # cdf_quantized = torch.cat([cdf_quantized, torch.tensor([0], dtype=torch.int16)])
        cdf_quantized = torch.cat([cdf_quantized, torch.tensor([0], dtype=torch.float32,device=mu.device)])
        # 扩展CDF
        cdf_expanded = cdf_quantized.repeat(len(sym_tensor), 1).detach().cpu()

        sym_to_enc = sym_tensor.detach().cpu()
        encoded_bytes = torchac.encode_float_cdf(cdf_expanded, sym_to_enc)
        
        
        bit_laplace_real = len(encoded_bytes) * 8 + 2*np.ceil(bitdepth) + 2 + 2*32
        return {'bit_ori': bit_ori, 'bit_tobyte': bit_tobyte, 'bit_zlib': bit_zlib, 'bit_laplace': bit_laplace_real}
        
        
    
    @torch.no_grad()
    def compress_test(self, model, new_model, bitdepth: int = 8):
        st1 = time.time()
        compress_out = self.compress_model(model, bitdepth)
        st2 = time.time()
        recon_model, recon_ret = self.decompress_model(new_model, compress_out)
        st3 = time.time()
        assert (compress_out['recon_ret'] != recon_ret).sum() == 0
        compress_out['enc_time'] = st2 - st1
        compress_out['dec_time'] = st3 - st2
        compress_out['new_model'] = recon_model
        return compress_out
    
    
    def compress_model(self, model, bitdepth: int = 8, derive_new_model=False, model_ori=None):
        params_ten = torch.cat([p.view(-1) for p in model.parameters()])
        min_param = params_ten.min()
        max_param = params_ten.max()
        quant_ret, recon_ret = self.quant_uniform2(params_ten, bitdepth) 
        
        if derive_new_model:
            if model_ori is None:
                print('Warning: model_ori is None')
            else:
                new_model = model_ori
                current_index = 0
                for p in new_model.parameters():
                    num_elements = p.numel()  # 获取参数的元素个数
                    new_values = recon_ret[current_index:current_index + num_elements].view_as(p)
                    p.data.copy_(new_values)  # 将重构的参数赋值回模型
                    current_index += num_elements
        else:
            new_model = None
        # 估计laplace分布
        mu = torch.round(quant_ret.mean())
        b = torch.round((quant_ret-mu).abs().mean())
        
        
        likelihood_laplace = mylaplace_pdf(quant_ret, mu, b)
        
        # 估计模型大小
        bits_laplace = -torch.sum(torch.log2(likelihood_laplace))
        
        bits_laplace = bits_laplace.item()


        enc_mode = 2
        bits = bits_laplace

        
        # bits还需要加上 最大值,最小值, mu, b/std的大小
        bits = bits + 2*bitdepth
        
        bpp = bits / len(quant_ret)
        
        
        np_type = np.uint8
        if bitdepth > 8:
            np_type = np.uint16
        if bitdepth > 16:
            np_type = np.uint32
        # 计算下界
        quant_numpy = quant_ret.detach().cpu().numpy().astype(np_type)
        quant_byte = quant_numpy.tobytes()
        quant_zlib = zlib.compress(quant_byte)
        bit_zlib_bound = len(quant_zlib) * 8
        bpp_zlib_bound = bit_zlib_bound / len(quant_ret)
        
        bpp_low_bound = bpp_zlib_bound if bpp_zlib_bound < bitdepth else bitdepth
        
        # 带上两个标志位 00直接编 01带zlib 编 10 laplace编  11正态分布编
        # 2是标志位 2*32是归一化参数
        bit_real = bits + 2 + 2*32
        side_info_bit = 2+2*32
        
        bit_laplace_real = 999999999999
        if bpp > bpp_low_bound:
            # print('Warning: AE bpp is larger than lowbound')
            enc_mode = 1
            final_bytes = quant_zlib
            if bpp_low_bound == bitdepth:
                enc_mode = 0
                final_bytes = quant_byte
                # print('zlib bpp is larger than bitdepth')
            bit_real = bpp_low_bound*len(quant_ret) + 2 + 2*32

        else:
            if bitdepth <= 8:
                sym_tensor = quant_ret.to(torch.int16)
                
                assert (sym_tensor != quant_ret).sum() == 0
                
                
                
                pdf_sample = mylaplace_pdf(torch.arange(np.ceil(2**bitdepth),device=mu.device), torch.round(mu), torch.round(b))
                pdf_sample = pdf_sample / pdf_sample.sum()
                
                cdf_sample = torch.cumsum(pdf_sample, dim=-1)
                cdf_sample = cdf_sample.to(torch.float32)
                cdf_quantized = cdf_sample

                
                cdf_quantized = torch.cat([cdf_quantized, torch.tensor([0], dtype=torch.float32,device=mu.device)])
                # 扩展CDF
                cdf_expanded = cdf_quantized.repeat(len(sym_tensor), 1).detach().cpu()
                sym_to_enc = sym_tensor.detach().cpu()
                encoded_bytes = torchac.encode_float_cdf(cdf_expanded, sym_to_enc)


                # dec_syms = torchac.decode_float_cdf(cdf_expanded, encoded_bytes)
                # assert (sym_to_enc != dec_syms).sum() == 0
                # assert (dec_syms != sym_tensor).sum() == 0
                
                bit_laplace_real = len(encoded_bytes) * 8 + 2*np.ceil(bitdepth) + 2 + 2*32
            else:
                bit_laplace_real = float('inf')
                
            # 还有机会再选择zlib
            if bit_laplace_real > bpp_low_bound*len(quant_ret) + 2 + 2*32:
                # print('Warning: AE bpp is larger than lowbound')
                enc_mode = 1
                final_bytes = quant_zlib
                if bpp_low_bound == bitdepth:
                    enc_mode = 0
                    final_bytes = quant_byte
                    # print('zlib bpp is larger than bitdepth')
                bit_real = bpp_low_bound*len(quant_ret) + 2 + 2*32
                
            else:
                bit_real = bit_laplace_real
                final_bytes = encoded_bytes
                side_info_bit = 2*np.ceil(bitdepth) + 2 + 2*32
                
        bpp_real = bit_real / len(quant_ret)
 
        return {
            'bpp_real': bpp_real, 'bit_real': bit_real, 'side_info_bit': side_info_bit,
            'bitdepth':bitdepth, 'enc_mode': enc_mode, 
            'laplace_real_bpp':  bit_laplace_real / len(quant_ret), 
            'zlib_bpp': bpp_zlib_bound, 'min_param': min_param, 
            'max_param': max_param, 'mu': mu, 'b': b, 
            'final_bytes': final_bytes,'recon_ret': recon_ret, 'new_model': new_model,
            # 'loc':locals()
            }
    
    
    
    def decompress_model(self, new_model, enc_out):
        
        params_new = torch.cat([p.view(-1) for p in new_model.parameters()])
        tensor_len = len(params_new)
        device = params_new.device
        
        enc_mode = enc_out['enc_mode']
        final_bytes = enc_out['final_bytes']
        min_param = enc_out['min_param']
        max_param = enc_out['max_param']
        mu = enc_out['mu']
        b = enc_out['b']
        # 如果mu和b不是tensor
        if type(mu) != torch.Tensor:
            mu = torch.tensor(mu, device=device)
        if type(b) != torch.Tensor:
            b = torch.tensor(b, device=device)

        bitdepth = enc_out['bitdepth']
        

        
        if enc_mode == 0:
            quant_numpy = np.frombuffer(final_bytes, dtype=np.uint8)
        elif enc_mode == 1:
            quant_numpy = np.frombuffer(zlib.decompress(final_bytes), dtype=np.uint8)
        else:
            pdf_sample = mylaplace_pdf(torch.arange(np.ceil(2**bitdepth),device=mu.device), torch.round(mu), torch.round(b))
            pdf_sample = pdf_sample / pdf_sample.sum()
            
            cdf_sample = torch.cumsum(pdf_sample, dim=-1)
            cdf_sample = cdf_sample.to(torch.float32)
            cdf_quantized = cdf_sample
            
            cdf_quantized = torch.cat([cdf_quantized, torch.tensor([0], dtype=torch.float32,device=mu.device)])
            # 扩展CDF
            cdf_expanded = cdf_quantized.repeat(tensor_len, 1).detach().cpu()

            dec_sym = torchac.decode_float_cdf(cdf_expanded, final_bytes)
            
            quant_numpy = dec_sym.cpu().numpy()
        
        recon_ret = torch.tensor(quant_numpy, dtype=torch.float32, device=params_new.device)
        sym_max_num = np.ceil(2**bitdepth) - 1
        
        ten_range = max_param - min_param
        recon_ret = recon_ret / sym_max_num * ten_range + min_param

                
        current_index = 0
        for p in new_model.parameters():
            num_elements = p.numel()  # 获取参数的元素个数
            new_values = recon_ret[current_index:current_index + num_elements].view_as(p)
            p.data.copy_(new_values)  # 将重构的参数赋值回模型
            current_index += num_elements
        
        return new_model, recon_ret
    
    
