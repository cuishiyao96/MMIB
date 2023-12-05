import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
import pdb
import math


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, in_features, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        print("heads: ", h, "d_model: ", d_model)
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(in_features, d_model, bias=None), 3)
        self.last_linear = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """Implements Figure 2"""
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # pdb.set_trace()
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        return self.last_linear(x)



class CrossAttention(nn.Module):
    def __init__(self, heads, in_size, out_size, dropout):
        super(CrossAttention, self).__init__()

        self.heads = heads
        self.hidden_size = out_size

        self.cross_attn = MultiHeadedAttention(self.heads,  in_size, out_size, dropout)
        self.ln1 = nn.LayerNorm(self.hidden_size)
        self.ln2 = nn.LayerNorm(self.hidden_size)
        self.dense = nn.Linear(in_size, self.hidden_size)
       

    def forward(self, query, key, value, key_mask):
        
        attn_output = self.cross_attn(query, key, value, key_mask)##
        # P = self.ln1(self.dense(query) + attn_output)
        PP = self.ln1(query + attn_output)
        P = self.ln2(PP + self.dense(PP))
        return P