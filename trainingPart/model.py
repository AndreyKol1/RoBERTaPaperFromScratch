import torch
import torch.nn as nn
import torch.functional as F
from architecture.MLMHead import MLMHead
from architecture.utils import TransformerBlock


class RoBERTa(nn.Module):
    def __init__(self, vocab_size, padding_idx, max_sequence_length = 128, d_model = 256, layers=6):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding(max_sequence_length, d_model)
        self.trf_block = nn.Sequential(*[TransformerBlock(d_model=d_model) for _ in range(layers)])
        self.mlmHead = MLMHead(d_model)

    def forward(self, x, attn_mask):
        batch_size, seq_len = x.shape
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=x.device)).unsqueeze(0)
        x = tok_emb + pos_emb


        for block in self.trf_block:
            x = block(x, attn_mask)

        x = self.mlmHead(x)
        x = F.linear(x, self.tok_emb.weight) # weight tying technique to save parameters(reusing existing weight matrix instead of creating new one)

        return x
