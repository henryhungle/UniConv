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

class DSTNet(nn.Module):
    def __init__(self, in_embed,
            layer_norm, 
            slot_dst,
            domain_dst,
            domain_slot_dst,
            dst_generator,
            args):
        super(DSTNet, self).__init__()
        self.in_embed=in_embed
        self.layer_norm = layer_norm
        self.slot_dst=slot_dst
        self.domain_dst=domain_dst
        self.domain_slot_dst=domain_slot_dst
        self.dst_generator=dst_generator
        self.args=args
        
    def load_pretrained(self, pretrained_dst):
        self.in_embed[0].load_pretrained(pretrained_dst.in_embed[0])
        count = 0
        for layer in self.layer_norm:
            layer.load_pretrained(pretrained_dst.layer_norm[count])
            count += 1
        self.slot_dst.load_pretrained(pretrained_dst.slot_dst)
        if self.domain_dst is not None: self.domain_dst.load_pretrained(pretrained_dst.domain_dst)
        if self.domain_slot_dst is not None: self.domain_slot_dst.load_pretrained(pretrained_dst.domain_slot_dst)
        self.dst_generator.load_pretrained(pretrained_dst.dst_generator)

    def forward(self, batch, out):
        out = self.encode(batch, out) 
        out = self.track_state(batch, out)
        return out 
        
    def encode(self, batch, out):
        count = 0
        if self.args.domain_flow:
            out['embedded_slots'] = self.layer_norm[count](self.in_embed[0](batch.in_slots))
            count += 1
            out['embedded_domains'] = self.layer_norm[count](self.in_embed[0](batch.in_domains))
            count += 1
        else:
            out['embedded_slots'] = self.layer_norm[count](self.in_embed[0](batch.in_slots).sum(2))
            count += 1
        if self.args.add_prev_dial_state:
            out['embedded_state'] = self.layer_norm[count](self.in_embed(batch.in_state))
            count += 1
        out['embedded_in_txt'] = self.layer_norm[count](self.in_embed(batch.in_txt))
        count += 1 
        if self.args.detach_dial_his:
            out['embedded_in_his'] = self.layer_norm[count](self.in_embed(batch.in_his))
            count += 1 
        return out 

    def track_state(self, batch, out): 
        if not self.args.add_prev_dial_state and not self.args.detach_dial_his:
            out['out_slots'] = self.slot_dst(out['embedded_slots'], None, out['embedded_in_txt'], batch.in_txt_mask)
        elif not self.args.add_prev_dial_state:
            out['out_slots'] = self.slot_dst(out['embedded_slots'], None, out['embedded_in_his'], batch.in_his_mask, out['embedded_in_txt'], batch.in_txt_mask)
        elif not self.args.detach_dial_his:
            out['out_slots'] = self.slot_dst(out['embedded_slots'], None, out['embedded_state'], batch.in_state_mask, out['embedded_in_txt'], batch.in_txt_mask)
        else:
            out['out_slots'] = self.slot_dst(out['embedded_slots'], None, out['embedded_in_his'], batch.in_his_mask, out['embedded_state'], batch.in_state_mask, out['embedded_in_txt'], batch.in_txt_mask)
            
        if self.args.domain_flow:
            if not self.args.detach_dial_his:
                out['out_domains'] = self.domain_dst(out['embedded_domains'], None, out['embedded_in_txt'], batch.in_txt_mask)
            else:
                out['out_domains'] = self.domain_dst(out['embedded_domains'], None, out['embedded_in_his'], batch.in_his_mask, out['embedded_in_txt'], batch.in_txt_mask)

            out_domains = out['out_domains'].unsqueeze(2)
            out_slots = out['out_slots'].unsqueeze(1)
            out_domains = out_domains.expand(out_domains.shape[0], out_domains.shape[1], out_slots.shape[2], out_domains.shape[3])
            out_slots = out_slots.expand(out_slots.shape[0], out_domains.shape[1], out_slots.shape[2], out_slots.shape[3])
            
            if hasattr(self.args, 'ds_fusion') and self.args.ds_fusion == 'sum':
                out['out_states'] = out_domains + out_slots 
            else:
                out['out_states'] = out_domains * out_slots 
                
            if hasattr(self.args, 'norm_ds') and self.args.norm_ds:
                out['out_states'] = self.layer_norm[-1](out['out_states'])
            
            if self.args.domain_slot_dst:
                out_states = out['out_states']
                original_size = out_states.shape
                out_states = out_states.view(out_states.shape[0], -1, out_states.shape[-1])
                out_states = self.domain_slot_dst(out_states, batch.domain_slot_mask) 
                out['out_states'] = out_states.view(*original_size)
        else:
            out['out_states'] = out['out_slots'] 
        return out 

class DST(nn.Module):
    def __init__(self, layer, N):
        super(DST, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def load_pretrained(self, pretrained_dst):
        count = 0
        for layer in self.layers:
            layer.load_pretrained(pretrained_dst.layers[count])
            count += 1
        self.norm.load_pretrained(pretrained_dst.norm)

    def forward(self, states, states_mask, in_txt1=None, in_mask1=None, 
                in_txt2=None, in_mask2=None, in_txt3=None, in_mask3=None):
        for layer in self.layers:
            states = layer(states, states_mask, in_txt1, in_mask1, in_txt2, in_mask2, in_txt3, in_mask3)
        return self.norm(states)
