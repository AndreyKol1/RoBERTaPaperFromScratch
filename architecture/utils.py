import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model = 256, num_heads = 8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0, "Number of dimensions should be divisible by heads"

        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.projection = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, attention_mask=None): 
        batch_size, seq_length, d_model = x.shape
        Q = self.W_q(x) #(batch_size, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2) # (batch_size, num_heads, seq_length, d_k)
        K = K.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

        attention_scores = Q @ K.transpose(2, 3)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2) # (batch_dim, 1, 1, seq_length)
            mask = mask.to(attention_scores.device) # making mask to prevent model attending to PAD tokens
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = torch.softmax(attention_scores / math.sqrt(self.d_k),  dim=-1) 
        attention_weights = self.dropout(attention_weights)

        final_weights = attention_weights @ V # (batch_size, num_heads, seq_length, d_k)
        final_weights = final_weights.transpose(1,2).contiguous().view(batch_size, seq_length, d_model)

        out_projection = self.projection(final_weights)

        return out_projection   

class FeedForward(nn.Module):
    def __init__(self, d_model = 256):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        return self.projection(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model = 256):
        super().__init__()
        self.attn = MultiHeadAttention()
        self.ffn = FeedForward()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask):
        residual = x
        x = self.norm1(x)
        x = self.attn(x, attn_mask)

        x += residual

        residual = x

        x = self.norm2(x)
        x = self.ffn(x)
        x += residual

        return x