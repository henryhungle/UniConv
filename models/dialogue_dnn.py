import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import math, copy, time
from torch.autograd import Variable
from collections import OrderedDict 
import pdb 

from models.utils import * 
from models.modules import *
from models.encoders import *
from models.dst import *
from models.nlg import *
from models.generators import *
# TODO: remove below line 
from models.decoders import *

class DialogueNet(nn.Module):
    def __init__(self, dst_net, nlg_net, dp_net, args):
        super(DialogueNet, self).__init__()
        self.dst_net = dst_net
        self.nlg_net = nlg_net
        self.dp_net = dp_net
        self.args = args 

    def forward(self, batch):
        out = {}
        if self.args.setting in ['dst', 'e2e']:
            out = self.dst_net(batch, out)
        if self.args.setting in ['c2t', 'e2e']:
            out = self.nlg_net(batch, out)
        return out 
                
def make_model(lang, slots, args): 
    c = copy.deepcopy
    attn = MultiHeadedAttention(args.att_h, args.d_model, dropout=args.dropout)
    ff = PositionwiseFeedForward(args.d_model, args.d_ff, args.dropout)
    position = PositionalEncoding(args.d_model, args.dropout)
    
    src_vocab = len(lang['in+domain+bs']['word2idx'])
    in_embed = [Embeddings(args.d_model, src_vocab), c(position)]
    in_embed = nn.Sequential(*in_embed)
    
    dst_net = None
    if args.setting in ['dst', 'e2e']:
        if args.domain_flow:
            slots_vocab = len(slots['slots'])
            domains_vocab = len(slots['domain_slots'])
        else:
            slots_vocab = len(slots['merged_slots'])
        # Create slot-level dst 
        nb_slot_dst_attn = 2
        if args.detach_dial_his:
            nb_slot_dst_attn += 1 
        if args.add_prev_dial_state:
            nb_slot_dst_attn += 1 
        slot_dst_layer = SubLayer(args.d_model, c(attn), c(ff), args.dropout, nb_attn=nb_slot_dst_attn)
        slot_dst = DST(slot_dst_layer, args.nb_blocks_slot_dst)
        # Create encoder for dst 
        nb_layer_norm = nb_slot_dst_attn
        if args.domain_flow:
            nb_layer_norm += 1
            if args.norm_ds:
                nb_layer_norm += 1

        layer_norm = clones(LayerNorm(args.d_model), nb_layer_norm)
        # create domain-level and domain-slot level dst 
        domain_dst = None
        domain_slot_dst = None 
        if args.domain_flow:
            nb_domain_dst_attn = 2
            if args.detach_dial_his:
                nb_domain_dst_attn += 1 
            domain_dst_layer = SubLayer(args.d_model, c(attn), c(ff), args.dropout, nb_attn=nb_domain_dst_attn)
            domain_dst = DST(domain_dst_layer, args.nb_blocks_domain_dst)

            if args.domain_slot_dst:
                domain_slot_dst_layer = SubLayer(args.d_model, c(attn), c(ff), args.dropout, nb_attn=1) 
                domain_slot_dst = DST(domain_slot_dst_layer, args.nb_blocks_domain_slot_dst)
        # Create dst generator 
        pointer_attn = MultiHeadedAttention(1, args.d_model, dropout=0)
        dst_generator = DSTGenerator(lang['dst']['word2idx'], slots, args, in_embed[0], pointer_attn)
        dst_net = DSTNet(
            in_embed=in_embed,
            layer_norm=layer_norm, 
            slot_dst=slot_dst,
            domain_dst=domain_dst,
            domain_slot_dst=domain_slot_dst,
            dst_generator=dst_generator,
            args=args)
        
    nlg_net=None
    dp_net=None
    if args.setting in ['c2t', 'e2e']:
        tgt_vocab = len(lang['out']['word2idx'])
        if args.share_inout:
            out_embed = in_embed 
        else: 
            out_embed = [Embeddings(args.d_model, tgt_vocab), c(position)]   
            out_embed = nn.Sequential(*out_embed)
        pointer_embed = [Embeddings(args.d_model, 2), c(position)]        
        pointer_embed = nn.Sequential(*pointer_embed)
        # Create response decoder 
        nb_res_dec_attn = 4
        if args.detach_dial_his:
            nb_res_dec_attn += 1 
        res_dec_layer = SubLayer(args.d_model, c(attn), c(ff), args.dropout, nb_attn=nb_res_dec_attn)
        nlg = NLG(res_dec_layer, args.nb_blocks_res_dec)
        nb_layer_norm = nb_res_dec_attn
        #if args.sys_act:
        #    nb_layer_norm += 1
        layer_norm = clones(LayerNorm(args.d_model), nb_layer_norm)
        res_generator = ResGenerator(args.d_model, tgt_vocab, out_embed[0])
        nlg_net = NLGNet(
            in_embed=in_embed,
            pointer_embed=pointer_embed,
            out_embed=out_embed,
            layer_norm=layer_norm, 
            nlg=nlg,
            res_generator=res_generator,
            args=args)  
       
        if args.sys_act:
            dp_net = SysActGenerator(args.d_model, len(lang['act']['word2idx']))
        
    model = DialogueNet(
        dst_net=dst_net,
        nlg_net=nlg_net,
        dp_net=dp_net,
        args=args)
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
            
    if args.setting == 'e2e' and args.pretrained_dst is not None:
        print("Loading pretrained dst from {}".format(args.pretrained_dst))
        pretrained_dst = torch.load(args.pretrained_dst)
        model.dst_net.load_pretrained(pretrained_dst.dst_net)
            
    return model
