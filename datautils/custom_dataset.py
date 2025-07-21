import os
import pickle
import numpy as np
import open3d as o3d
from models.module_utils import qscTensor, octree_level_obj
import torch
from torch.nn import functional as F
from models.sort_functions import sort_by_coord_sum_c
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def read_ply_o3d(filedir, dtype='int32'):
    pcd = o3d.io.read_point_cloud(filedir)
    coords = np.asarray(pcd.points).astype(dtype)

    return coords

def write_ply_o3d(filedir, coords, dtype='int32', normal=False, knn=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords.astype(dtype))
    if normal:
        assert knn is not None
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    o3d.io.write_point_cloud(filedir, pcd, write_ascii=True)
    f = open(filedir)
    lines = f.readlines()
    lines[4] = 'property float x\n'
    lines[5] = 'property float y\n'
    lines[6] = 'property float z\n'
    if normal:
        lines[7] = 'property float nx\n'
        lines[8] = 'property float ny\n'
        lines[9] = 'property float nz\n'
    fo = open(filedir, "w")
    fo.writelines(lines)
    
    return

def write_ply_ascii(filedir, coords, dtype='int32'):
    if os.path.exists(filedir):
        os.remove(filedir)  # 直接删除文件，避免 `os.system('rm')` 依赖

    with open(filedir, 'w') as f:
        # 写入头部
        f.writelines([
            "ply\n",
            "format ascii 1.0\n",
            f"element vertex {coords.shape[0]}\n",
            "property float x\n",
            "property float y\n",
            "property float z\n",
            "end_header\n"
        ])

        # 批量转换为字符串格式（避免逐行调用 write）
        coords = coords.astype(dtype)
        lines = [" ".join(map(str, p)) + "\n" for p in coords]
        f.writelines(lines)  # 一次性写入所有数据

    return
 
def inplace_replace_ply_header(file_path):
    with open(file_path, "r+b") as f:
        # 读取头部（逐行）
        header_lines = []
        while True:
            line = f.readline()
            header_lines.append(line)
            if line.strip() == b"end_header":
                break  # 头部读取完成
        
        # 修改头部中的 "double" 为 "float"
        new_header_lines = [line.replace(b"double", b"float ") for line in header_lines]

        # 计算原始头部和新头部的大小
        original_header_size = sum(len(line) for line in header_lines)
        new_header_size = sum(len(line) for line in new_header_lines)

        # 如果新头部和旧头部大小不同，则需要移动数据
        # if new_header_size != original_header_size:
        #     print("新头部大小与原头部不同，无法直接就地修改")
        #     return

        # 就地回写修改后的头部
        f.seek(0)
        for line in new_header_lines:
            f.write(line)
        
        # print("PLY 头部修改完成，数据未变:", file_path)
def cvt_bin2dec(data):
    r, c = data.shape
    out_data = np.zeros((r, 1))
    for i in range(c):
        out_data = 2*out_data + data[:, i]
    return out_data
        
def cvt_dec2hot(data, max_d=15):
    # 将data(N*1)转换为one-hot编码, data为0-max_d的整数
    # 使用torch实现
    num_classes = max_d+1
    one_hot_encoded = F.one_hot(data.squeeze(dim=-1), num_classes=num_classes).float()
    return one_hot_encoded


class Read_Data:
    def __init__(self, dataset, idx_range):
        self.dataset = dataset
        self.offset_idx = idx_range[0]
        self.idx_range = idx_range
    def __getitem__(self, idx):
        return self.dataset[idx+self.offset_idx]
    def __len__(self):
        return len(self.idx_range)
    
class Read_Data_with_cache(Read_Data):
    def __init__(self, dataset, idx_range):
        super(Read_Data_with_cache, self).__init__(dataset, idx_range)
        print('Reading data with cache')
        for idx in self.idx_range:
            self.dataset[idx]
        print('Reading data with cache done')
        
        

class MytestDataset:
    def __init__(self, ori_dir, ori_type='npy'):
        ori_data_names = os.listdir(ori_dir)
        self.ori_type = ori_type
        ori_data_names = sorted(ori_data_names)
        self.all_files_path = []
        # self.all_files_name = []
        # self.all_files_path = [os.path.join(ori_dir, name) for name in ori_data_names]
        for name in ori_data_names:
            if name.endswith('.'+ori_type):
                self.all_files_path.append(os.path.join(ori_dir, name))
        if self.all_files_path == []:
            raise ValueError('No file found in the directory')
        
    def __getitem__(self, idx):
        file_path = self.all_files_path[idx]
        assert os.path.exists(file_path)
        data = self.handle_data(file_path)
        return data
    
    def handle_data(self, file_path):
        if self.ori_type == 'npy':
            coord_data = np.load(file_path)
        elif self.ori_type == 'ply':
            coord_data = read_ply_o3d(file_path)
        else:
            raise ValueError('ori_type should be npy or ply')
        coord_data = torch.tensor(coord_data, dtype=torch.int32, device=device)
        coord_data = sort_by_coord_sum_c(coord_data)
        return coord_data


class MyDataset:
    def __init__(self, ori_dir, handle_dir=None, scale_num=None, ori_type='npy',stage=4, derive_neigbor=False, derive_ori=False):
        self.derive_neigbor = derive_neigbor
        self.derive_ori = derive_ori       
        load_cache = handle_dir is not None
        if handle_dir is None:
            print('Warning: handle_dir is None, the cache will not be saved')
        else:
            if not os.path.exists(handle_dir):
                os.makedirs(handle_dir)
            print('handle_dir:', handle_dir)
        
        
        self.stage = stage
        if stage == 3:
            self.stage_list = [[0,7],[1,6],[2,3,4,5]]
        elif stage == 4:
            self.stage_list = [[0,1],[2,3],[4,5],[6,7]]
        elif stage == 8:
            self.stage_list = [[0],[1],[2],[3],[4],[5],[6],[7]]
        ori_data_names = os.listdir(ori_dir)
        self.ori_type = ori_type
        self.scale_num = scale_num
        self.load_cache = load_cache
        ori_data_names = sorted(ori_data_names)
        self.all_files_path = []
        # self.all_files_name = []
        self.all_handle_path = []
        # self.all_files_path = [os.path.join(ori_dir, name) for name in ori_data_names]
        for name in ori_data_names:
            
            data_pth = os.path.join(ori_dir, name)
            if os.path.isdir(data_pth):
                continue
        
            if name.endswith('.'+ori_type):
                self.all_files_path.append(os.path.join(ori_dir, name))
                f_name = name.split('.')[0]
                if self.load_cache:
                    self.all_handle_path.append(os.path.join(handle_dir, f_name+'.pkl'))

        if self.all_files_path == []:
            raise ValueError('No file found in the directory')

        self.min_point_num = 64
        
    def split(self, data):
        data_lst = []
        for s_idx in range(self.stage):
            sli_idx = self.stage_list[s_idx]
            data_lst.append(data[:,sli_idx])
        return data_lst
            
    def set_prefix_data(self, setdata):
        
        offsets_ini = setdata.get('offsets_ini')
        offset_of_neigbor = setdata.get('offset_of_neigbor')
        
        if offsets_ini is None:
            offsets_ini = [[0,0,0],[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]]
            # offsets = [[i, j, k] for i in range(-1, 2) for j in range(-1, 2) for k in range(-1, 2)]
            # offsets = offsets +[[2,0,0],[-2,0,0],[0,2,0],[0,-2,0],[0,0,2],[0,0,-2]]
            self.offsets_ini = torch.tensor(offsets_ini).to(device)
        else:
            self.offsets_ini = offsets_ini
            
        
        if offset_of_neigbor is None:
            offset_of_neigbor = [[0,0,0],[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]]
            self.offset_of_neigbor = torch.tensor(offset_of_neigbor, dtype=torch.float32).to(device)
        else:
            self.offset_of_neigbor = offset_of_neigbor
    
        self.min_point_num = setdata.get('min_point_num', 64)
        
    def __getitem__(self, idx):
        
        
        file_path = self.all_files_path[idx]
        
        if self.load_cache:
            handle_path = self.all_handle_path[idx]
            if os.path.exists(handle_path):
                try:
                    with open(handle_path, 'rb') as f:
                        data = pickle.load(f)
                    return data
                except:
                    print('Warning:', handle_path, 'is broken, will be removed, and regenerate')
                    os.remove(handle_path)
        
        assert os.path.exists(file_path)
        data = self.handle_data(file_path)
        
        if self.load_cache:
            handle_path = self.all_handle_path[idx]
            dir_name = os.path.dirname(handle_path)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            with open(handle_path, 'wb') as f:
                pickle.dump(data, f)
                
        return data
    
    def handle_data(self, file_path):
        offsets_ini = self.offsets_ini
        offset_of_neigbor = self.offset_of_neigbor
        
        if self.ori_type == 'npy':
            coord_data = np.load(file_path)
        elif self.ori_type == 'ply':
            coord_data = read_ply_o3d(file_path)
        else:
            raise ValueError('ori_type should be npy or ply')
        # return data
        scale_num = self.scale_num
        coord_data = coord_data[:, :3]
        
        coord_data_min = coord_data.min(axis=0)
        
        # if np.any(coord_data_min < 0):
        coord_data = coord_data - coord_data_min
        
        
        xyz = torch.tensor(coord_data, dtype=torch.int32, device=device)
        xyz = torch.unique(xyz, dim=0)
        attr = torch.ones((xyz.shape[0], 1), device=device)
        xyzqsc_t = qscTensor(xyz, attr)

        ori_data = xyzqsc_t.get_coord()
        to_save_data = {}
        all_input_info = []
        if scale_num is None:
            scale_num = 100000
        for s_idx in range(scale_num):
            # s_embed = embedding(torch.tensor([s_idx], device=device))
            
            xyzqsc_t.set_oct_level()
            parent_C, occupancy = xyzqsc_t.get_oct_level()
            # 确保可以复原
            assert (xyzqsc_t.coord != octree_level_obj.upper_layer(parent_C, occupancy)).sum() == 0            
            # odd_data = cvt_bin2dec(occupancy_odd)
            # even_data = cvt_bin2dec(occupancy_even)
            
            # odd_data_onehot = cvt_dec2hot(odd_data, 15)
            # even_data_onehot = cvt_dec2hot(even_data, 15)
            
            
            # 根据低尺度的信息构建低尺度信息集合
            xyzQ = parent_C
            attrQ = torch.ones((xyzQ.shape[0], 1), device=device)
            xyzQqsc_t = qscTensor(xyzQ, attrQ)
            
            assert (xyzQqsc_t.coord != xyzQ).sum() == 0
            
            # 设置更加丰富的低尺度信息为后续超分做准备
            # xyzQqsc_t.set_offset_sparse(offsets_ini)
            xyzQqsc_t.set_offset_tensor(offsets_ini)
            
            # 准备周围邻居的idxs信息，为后续查找做准备
            if self.derive_neigbor:
                xyzQqsc_t.set_neigber_idxs(offset_of_neigbor)
            
            all_input_info.append({
                'xyzqsc_t':xyzQqsc_t,
                'ground_truth':xyzqsc_t.get_coord(),
                'scale_idx':s_idx,
                'occ_lst': self.split(occupancy),
            })
            
            if xyzQ.shape[0] < self.min_point_num or s_idx == scale_num-1:
                # 计算xyzQ即将占用的大小
                max_data = xyzQ.max().detach().cpu().numpy()
                # 表示一个数需要的bit数
                bitdepthQ = int(np.ceil(np.log2(max_data+1)))
                # 这个空间最多表示多少个点
                max_point_num = (2**bitdepthQ)**3
                
                # 数据有多少点
                point_num = xyzQ.shape[0]
                
                # enc_point
                enc_point_num = point_num if point_num < max_point_num-point_num else max_point_num-point_num
                
                # 编码需要的bit数
                xyzQ_low_bits = enc_point_num * bitdepthQ * 3
                
                break
            
            xyzqsc_t = xyzQqsc_t
        if self.scale_num is None:
            self.scale_num = s_idx+1
        to_save_data['all_input_info'] = all_input_info
        to_save_data['xyzQ_low_bits'] = xyzQ_low_bits
        to_save_data['point_num'] = xyz.shape[0]
        if self.derive_ori:
            to_save_data['ori'] = ori_data
        else:
            to_save_data['ori'] = None
        to_save_data['coord_data_min'] = coord_data_min.tolist()
        return to_save_data


