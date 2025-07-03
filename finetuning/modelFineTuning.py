import torch
import torch.nn as nn
import torch.functional as F
from architecture.utils import TransformerBlock

class RoBERTa(nn.Module):
    def __init__(self, vocab_size, padding_idx, num_labels = 2, max_sequence_length = 128, d_model = 256, layers=6):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding(max_sequence_length, d_model)
        self.trf_block = nn.Sequential(*[TransformerBlock(d_model=d_model) for _ in range(layers)])
        self.dropout = nn.Dropout(0.1)
        self.class_head = nn.Linear(d_model, num_labels)

    def forward(self, x, attn_mask):
        batch_size, seq_len = x.shape
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=x.device)).unsqueeze(0)
        x = tok_emb + pos_emb


        for block in self.trf_block:
            x = block(x, attn_mask)
        
        cls_token = x[:, 0, :]
        x = self.dropout(cls_token)
        x = self.class_head(x)

        return x
