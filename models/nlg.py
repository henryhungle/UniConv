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
from utils.dataset import add_sys_act_mask

class NLGNet(nn.Module):
    def __init__(self, in_embed,
            pointer_embed,
            out_embed,
            layer_norm, 
            nlg,
            res_generator,
            args):
        super(NLGNet, self).__init__()
        self.in_embed=in_embed
        self.pointer_embed=pointer_embed
        self.out_embed=out_embed
        self.layer_norm = layer_norm
        self.nlg=nlg
        self.res_generator=res_generator
        self.args=args
        if self.args.sys_act:
            self.sys_act_prior = Variable(nn.init.xavier_uniform(torch.zeros(1, args.d_model)), requires_grad=True).squeeze(0).cuda()
            self.res_norm = LayerNorm(args.d_model)
            self.act_norm = LayerNorm(args.d_model)
        
    def forward(self, batch, out):
        out = self.encode(batch, out) 
        out = self.decode_response(batch, out)
        return out
    
    def encode(self, batch, out):
        count = 0
        if self.args.setting in ['e2e']:
            if hasattr(self.args, 'literal_bs') and self.args.literal_bs:
                out['embedded_curr_state'] = self.layer_norm[count](self.in_embed(batch.in_curr_state))
            else:
                curr_state = out['out_states'].reshape(out['out_states'].shape[0], -1, out['out_states'].shape[-1])
                out['embedded_curr_state'] = self.layer_norm[count](curr_state)
        elif self.args.setting in ['c2t']:
            out['embedded_curr_state'] = self.layer_norm[count](self.in_embed(batch.in_curr_state))
        count += 1
        out['embedded_in_txt'] = self.layer_norm[count](self.in_embed(batch.in_txt))
        count += 1 
        if self.args.detach_dial_his:
            out['embedded_in_his'] = self.layer_norm[count](self.in_embed(batch.in_his))
            count += 1 
        if hasattr(self.args, 'sys_act') and self.args.sys_act:
            in_res_embed = self.out_embed(batch.out_txt)
            exp_prior = self.sys_act_prior.unsqueeze(0).unsqueeze(0).expand(in_res_embed.shape[0], -1, self.sys_act_prior.shape[0])
            out['embedded_in_res'] = self.layer_norm[count](torch.cat([exp_prior, in_res_embed], dim=1))
            count += 1
        else:
            out['embedded_in_res'] = self.layer_norm[count](self.out_embed(batch.out_txt))
            count += 1
        out['embedded_in_ptr'] = self.layer_norm[count](self.pointer_embed(batch.out_ptr))
        count += 1
        return out 
    
    def decode_response(self, batch, out):
        if self.args.sys_act:
            batch.out_mask = add_sys_act_mask(batch.out_mask)
        if self.args.setting in ['e2e']:
            if hasattr(self.args, 'literal_bs') and self.args.literal_bs:
                curr_state_mask = batch.in_curr_state_mask
            else:
                if self.args.domain_flow:
                    curr_state_mask = batch.domain_slot_mask
                else:
                    curr_state_mask = None 
        elif self.args.setting in ['c2t']:
            curr_state_mask = batch.in_curr_state_mask
        if self.args.detach_dial_his:
            out['out_res'] = self.nlg(
                out['embedded_in_res'], batch.out_mask, out['embedded_in_his'], batch.in_his_mask,
                out['embedded_in_txt'], batch.in_txt_mask, out['embedded_curr_state'], curr_state_mask,
                out['embedded_in_ptr'], None)
        else:
            out['out_res'] = self.nlg(
                out['embedded_in_res'], batch.out_mask, 
                out['embedded_in_txt'], batch.in_txt_mask, out['embedded_curr_state'], curr_state_mask,
                out['embedded_in_ptr'], None)
        if hasattr(self.args, 'sys_act') and self.args.sys_act:
            out['out_res'] = self.res_norm(out['out_res'][:,1:,:])
            out['out_act'] = self.act_norm(out['out_res'][:,0,:])
        return out 

class NLG(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(NLG, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, out, out_mask, 
                in_txt1=None, in_mask1=None, in_txt2=None, in_mask2=None, 
                in_txt3=None, in_mask3=None, in_txt4=None, in_mask4=None):
        for layer in self.layers:
            out = layer(out, out_mask, in_txt1, in_mask1, in_txt2, in_mask2, in_txt3, in_mask3, in_txt4, in_mask4)
        return self.norm(out)
