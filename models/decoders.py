import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import math, copy, time
from torch.autograd import Variable
from collections import OrderedDict 
import pdb 

from models.modules import *

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N, mode):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.mode = mode 
        
    def forward(self, out, out_mask, in_txt1, in_mask1, in_txt2, in_mask2, in_txt3, in_mask3, in_txt4, in_mask4, in_txt5, in_mask5):
        for layer in self.layers:
            out = layer(out, out_mask, in_txt1, in_mask1, in_txt2, in_mask2, in_txt3, in_mask3, in_txt4, in_mask4, in_txt5, in_mask5)
        return self.norm(out)