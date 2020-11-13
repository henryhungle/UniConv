import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import math, copy, time
from torch.autograd import Variable
from collections import OrderedDict 
import pdb 


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
        
    def load_pretrained(self, layer):
        self.a_2 = nn.Parameter(layer.a_2)
        self.b_2 = nn.Parameter(layer.b_2)

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout, gated_conn=False):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.gated_conn = gated_conn
        if gated_conn:
            self.g_w1 = nn.Linear(size, size)
            self.g_w2 = nn.Linear(size, size)
            self.g_w3 = nn.Linear(size, 1)

    def load_pretrained(self, pretrained_layer):
        self.norm.load_pretrained(pretrained_layer.norm)
                    
    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        out = sublayer(self.norm(x))
        if self.gated_conn:
            g_s = torch.sigmoid(self.g_w3(self.dropout(torch.tanh(self.g_w1(out) + self.g_w2(x)))))
            return g_s * x + (1 - g_s) * self.dropout(out)
        else:
            return x + self.dropout(out)

    def expand_forward(self, x, sublayer):
        out = self.dropout(sublayer(self.norm(x)))
        out = out.mean(1).unsqueeze(1).expand_as(x)
        return x + out 

    def nosum_forward(self, x, sublayer):
        return self.dropout(sublayer(self.norm(x)))
    
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores =torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, d_in=-1, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        if d_in < 0: 
            d_in = d_model
        self.linears = clones(nn.Linear(d_in, d_model), 3)
        self.linears.append(nn.Linear(d_model, d_in))
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def load_pretrained(self, pretrained_attn):
        count = 0
        for linear in self.linears:
            linear.weight.data.copy_(pretrained_attn.linears[count].weight.data)
            count += 1 

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1, d_out=-1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        if d_out < 0:
            d_out = d_model 
        self.w_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)
        
    def load_pretrained(self, pretrained_ff):
        self.w_1.weight.data.copy_(pretrained_ff.w_1.weight.data)
        self.w_2.weight.data.copy_(pretrained_ff.w_2.weight.data)
        
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
        
    def load_pretrained(self, pretrained_emb):
        self.lut.weight.data.copy_(pretrained_emb.lut.weight.data)

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
    
class SubLayer(nn.Module):
    def __init__(self, size, attn, ff, dropout, nb_attn):
        super(SubLayer, self).__init__()
        self.ff = ff
        self.size = size 
        self.attn = clones(attn, nb_attn)
        self.sublayer = clones(SublayerConnection(size, dropout), nb_attn+1)
        
    def load_pretrained(self, pretrained_layer):
        count = 0
        for attn in self.attn:
            attn.load_pretrained(pretrained_layer.attn[count])
            count += 1
        count = 0
        for sublayer in self.sublayer:
            sublayer.load_pretrained(pretrained_layer.sublayer[count])
            count += 1 
        self.ff.load_pretrained(pretrained_layer.ff)

    def forward(self, tgt, tgt_mask, in_txt1=None, in_mask1=None, in_txt2=None, in_mask2=None, in_txt3=None, in_mask3=None, in_txt4=None, in_mask4=None, in_txt5=None, in_mask5=None):
        out = self.sublayer[0](tgt, lambda tgt: self.attn[0](tgt, tgt, tgt, tgt_mask))
        if in_txt1 is not None:
            out = self.sublayer[1](out, lambda out: self.attn[1](out, in_txt1, in_txt1, in_mask1))
            if in_txt2 is not None:
                out = self.sublayer[2](out, lambda out: self.attn[2](out, in_txt2, in_txt2, in_mask2))
                if in_txt3 is not None:
                    out = self.sublayer[3](out, lambda out: self.attn[3](out, in_txt3, in_txt3, in_mask3))
                    if in_txt4 is not None:
                        out = self.sublayer[4](out, lambda out: self.attn[4](out, in_txt4, in_txt4, in_mask4))
                        if in_txt5 is not None:
                            out = self.sublayer[5](out, lambda out: self.attn[5](out, in_txt5, in_txt5, in_mask5))
                            return self.sublayer[6](out, self.ff)
                        else:
                            return self.sublayer[5](out, self.ff)
                    else:
                        return self.sublayer[4](out, self.ff)
                else:
                    return self.sublayer[3](out, self.ff)
            else:
                return self.sublayer[2](out, self.ff)
        else:
            return self.sublayer[1](out, self.ff)

'''
class GradZero(Function):
    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * 0)

def grad_reverse(x):
    return GradReverse()(x)
'''