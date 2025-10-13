# LINR-PCGC: Lossless Implicit Neural Representations for Point Cloud Geometry Compression

Existing AI-based point cloud compression methods struggle with dependence on specific training data distributions, which limits their real-world deployment. Implicit Neural Representation (INR) methods solve the above problem by encoding overfitted network parameters to the bitstream, resulting in more distribution-agnostic results. However, due to the limitation of encoding time and decoder size, current INR based methods only consider lossy geometry compression. In this paper, we propose the first INR based lossless point cloud geometry compression method called Lossless Implicit Neural Representations for Point Cloud Geometry Compression (LINR-PCGC). To accelerate encoding speed, we design a group of point clouds level coding framework with an effective network initialization strategy, which can reduce around 60% encoding time. A lightweight coding network based on multiscale SparseConv, consisting of scale context extraction, child node prediction, and model compression modules, is proposed to realize fast inference and compact decoder size. Experimental results show that our method consistently outperforms traditional and AI-based methods: for example, with the convergence time in the MVUB dataset, our method reduces the bitstream by approximately 21.21% compared to G-PCC TMC13v23 and 21.95% compared to SparsePCGC. Our project can be seen on <https://huangwenjie2023.github.io/LINR-PCGC/>

## News

- 2025.8.30 We have posted the manuscript on arxiv (<https://arxiv.org/abs/2507.15686>).

## Requirments

- pytorch **1.13.1+cu11.7**
- Recommended CUDA version: **11.7**
- RTX 3090 GPU, 24GB memory
- MinkowskiEngine 0.5.4
- **[Traindata]**: LINR-PCGC is an INR-based method which doesn't use the training.
- **[Testdata]**: 8iVFB, Owlii, MVUB, which will be loaded to BaiduNetdisk soon.

## Usage

### Testing

The following example commands are provided to illustrate the general testing process.

```bash
# For Testing with metric results, encode and decode. 
# In most cases, this is all that is needed to complete the entire process
CUDA_VISIBLE_DEVICES=0 python main.py --overfit True --mid_test True --encode True --decode True --handle_dir ./tmp/loot --result_dir ./output/loot --encode_dir result_enc/loot --decode_dir result_dec/loot --ori_dir /home/huangwenjie/pythonfiles/LINR_PCGC/dataset/8iVFB/8iVFBv2/loot --first_epoch 10 --others_epoch 10 --gop_size 32 --frame_num 96
```

```bash
# For just encoding without middle metric results.
CUDA_VISIBLE_DEVICES=0 python main.py --overfit True --mid_test False --encode True --decode False --handle_dir ./tmp/loot --result_dir ./output/loot --encode_dir result_enc/loot --decode_dir result_dec/loot --ori_dir /home/huangwenjie/pythonfiles/LINR_PCGC/dataset/8iVFB/8iVFBv2/loot --first_epoch 10 --others_epoch 10 --gop_size 32 --frame_num 96
```

```bash
# For just decoding without middle metric results.
CUDA_VISIBLE_DEVICES=0 python main.py --overfit True --mid_test False --encode False --decode True --handle_dir ./tmp/loot --result_dir ./output/loot --encode_dir result_enc/loot --decode_dir result_dec/loot 
```
