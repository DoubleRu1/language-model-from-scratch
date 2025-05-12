import torch
from torch.utils.data import Dataset

class TokenDataset(Dataset):
    def __init__(self, token_ids, block_size):
        self.token_ids = token_ids
        self.block_size = block_size

    def __len__(self):
        return len(self.token_ids) // self.block_size

    def __getitem__(self, idx):
        i = idx * self.block_size
        x = torch.tensor(self.token_ids[i:i+self.block_size], dtype=torch.long)
        y = torch.tensor(self.token_ids[i+1:i+self.block_size+1], dtype=torch.long)
        return x, y
    


