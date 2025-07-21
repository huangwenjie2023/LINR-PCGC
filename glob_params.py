import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
offsets_ini = torch.tensor([[0,0,0],[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]], device=device)
offset_of_neigbor = torch.tensor([[0,0,0],[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]], device=device)
