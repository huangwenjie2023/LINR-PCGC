import os, sys
filedir = os.path.dirname(os.path.abspath(__file__))
rootdir = os.path.dirname(filedir)
sys.path.append(rootdir)
from custom_dataset import inplace_replace_ply_header

path = '/home/huangwenjie/python_files/PCAC/LINR_PCGC/result_dec/basketball/frame0000.ply'
inplace_replace_ply_header(path)