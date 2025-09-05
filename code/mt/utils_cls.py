# Toy dataset + utils for Transformer
import torch
from torch.utils.data import DataLoader, Dataset

class ToyDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_ids = torch.tensor([self.src_vocab.get(ch, 1) for ch in src])
        tgt_ids = torch.tensor([self.tgt_vocab.get(ch, 1) for ch in tgt])
        return src_ids, tgt_ids

def get_data():
    src_vocab = {'<pad>':0, '<unk>':1, 'a':2, 'b':3, 'c':4}
    tgt_vocab
