import torch
def quatize_with_feat(coords, feat, precise, quant_mode='floor'):
    # assert factor <= 1
    if precise==1: return coords, feat
    # coords = x.C[:,1:]
    coordsQ = coords / precise
    if quant_mode=='round': coordsQ = torch.round(coordsQ)
    elif quant_mode=='floor': coordsQ = torch.floor(coordsQ)
    coordsQ = coordsQ.int()
    # 去重
    new_coord = torch.unique(coordsQ, dim=0)
    # new_coord = new_coord.to(coords.device)
    new_feat = torch.zeros(new_coord.size(0), feat.size(1), dtype=feat.dtype, device=feat.device)
    for i, c in enumerate(coordsQ):
        idx = (new_coord == c).all(dim=1).nonzero()
        new_feat[idx] += feat[i]
    return new_coord, new_feat

def quantize(coords, precise, quant_mode='floor'):
    # assert factor <= 1
    if precise==1: return coords
    # coords = x.C[:,1:]
    coordsQ = coords / precise
    if quant_mode=='round': coordsQ = torch.round(coordsQ)
    elif quant_mode=='floor': coordsQ = torch.floor(coordsQ)
    coordsQ = coordsQ.int()
    # 去重
    new_coord = torch.unique(coordsQ, dim=0)
    # new_coord = new_coord.to(coords.device)
    return new_coord

def dequatize(coords, precise):
    return coords * precise


def quat_dequat(coords, precise):
    return dequatize(quantize(coords, precise), precise)
