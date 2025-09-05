# Minimal training loop for toy translation dataset
import torch
import torch.nn as nn
import torch.optim as optim
from transformer import Transformer
from utils_mt import get_data, create_masks, evaluate_bleu

def main():
    src_vocab, tgt_vocab, train_loader, val_loader = get_data()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Transformer(src_vocab, tgt_vocab).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(30):
        model.train()
        for src, tgt_in, tgt_out in train_loader:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            src_mask, tgt_mask = create_masks(src, tgt_in)
            outputs = model(src, tgt_in, src_mask, tgt_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), tgt_out.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch} Loss {loss.item():.4f}')
        evaluate_bleu(model, val_loader, device)

if __name__ == '__main__':
    main()
