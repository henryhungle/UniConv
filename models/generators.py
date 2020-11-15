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

class ResGenerator(nn.Module):        
    def __init__(self, d_model, d_out, src_emb=None):
        super(ResGenerator, self).__init__()
        if src_emb is not None:
            self.proj = src_emb.lut.weight
            self.shared_wt = True
        else:
            self.proj = nn.Linear(d_model, vocab)
            self.shared_wt = False
       
    def forward(self, batch, out):
        if self.shared_wt:
            out['out_res_seqs'] = F.log_softmax(torch.matmul(out['out_res'], self.proj.transpose(1,0)), dim=-1)
        else:
            out['out_res_seqs'] = F.log_softmax(self.proj(out['out_res']), dim=-1)
        return out 
    
class SysActGenerator(nn.Module):
    def __init__(self, d_model, d_out):
        super(SysActGenerator, self).__init__()
        self.classifier = nn.Linear(d_model, d_out) 

    def forward(self, batch, out):
        out['out_act_logits'] = self.classifier(out['out_act'])
        return out 

class DSTGenerator(nn.Module):
    def __init__(self, dst_vocab, slots, args, src_emb=None, pointer_attn=None):
        super(DSTGenerator, self).__init__()
        d_model = args.d_model
        domain_slots = slots['domain_slots']
        dropout = args.dropout        
        if args.share_dst_gen:
            self.bi_generator = BinaryGenerator(d_model, dropout)
            self.rnn_generator = RNNGenerator(d_model, -1, dropout, src_emb, pointer_attn)
        else:
            domain_gens = []
            for domain, slots_ls in domain_slots.items():
                gens = []
                for slot in slots_ls:
                    if 'request' in slot or 'booked' in slot or 'is_active' in slot:
                        gens.append(BinaryGenerator(d_model, dropout))
                    else:
                        if 'booking_' in slot:
                            gens.append(RNNGenerator(d_model, -1, dropout, src_emb, pointer_attn))
                        else:
                            if args.dst_classify:
                                gens.append(RNNGenerator(d_model, len(slots['domain_slots_vocab'][domain][slot]), dropout))
                            else:
                                gens.append(RNNGenerator(d_model, len(dst_vocab[domain][slot]), dropout))
                domain_gens.append(nn.ModuleList(gens))
            gen_names = ["dstgen_{}".format(i) for i in domain_slots.keys()]
            self.generators = nn.Sequential(OrderedDict([i for i in zip(gen_names, domain_gens)]))       
        self.slots = slots
        self.args = args
        
    def load_pretrained(self, pretrained_gen):
        if self.args.share_dst_gen:
            self.bi_generator.load_pretrained(pretrained_gen.bi_generator)
            self.rnn_generator.load_pretrained(pretrained_gen.rnn_generator)
        else:
            pdb.set_trace()
        
    def expand_tensor2d(self, batch, dim):
        return batch.unsqueeze(1).expand(
                batch.shape[0], dim, batch.shape[1]).reshape(
            -1, batch.shape[1])
    
    def expand_tensor3d(self, batch, dim):
        return batch.unsqueeze(1).expand(
                batch.shape[0], dim, batch.shape[1], batch.shape[2]).reshape(
            -1, batch.shape[1], batch.shape[2])
    
    def forward(self, batch, out): 
        if self.args.share_dst_gen:
            req_states = out['out_states'].reshape(out['out_states'].shape[0],-1,out['out_states'].shape[-1])[:,batch.req_idx,:]
            inf_states = out['out_states'].reshape(out['out_states'].shape[0],-1,out['out_states'].shape[-1])[:,batch.inf_idx,:]
            out['out_req_states'] = self.bi_generator(req_states)
            inf_states = inf_states.reshape(-1, inf_states.shape[-1])
            in_inf_states = batch.out_inf_state.reshape(-1, batch.out_inf_state.shape[-1])
            exp_embedded_in_txt = self.expand_tensor3d(out['embedded_in_txt'], len(batch.inf_idx))
            exp_in_txt = self.expand_tensor2d(batch.in_txt, len(batch.inf_idx))
            exp_in_txt_mask = self.expand_tensor3d(batch.in_txt_mask, len(batch.inf_idx))
            out['out_inf_states'] = self.rnn_generator(inf_states, in_inf_states, exp_embedded_in_txt, exp_in_txt, exp_in_txt_mask)
            return out 
        else:
            dst_x = {}
            if self.args.domain_flow:
                count = 0
                for domain, indices in self.slots['slots_idx'].items():
                    dst_x[domain] = out['out_states'][:,count,indices,:]
                    count += 1
            else:
                for domain, indices in self.slots['merged_slots_idx'].items():
                    dst_x[domain] =  out['out_states'][:,indices,:]
            out_states = {}
            for domain, domain_state in dst_x.items():
                out_states[domain] = []
                for state_idx in range(domain_state.shape[1]):
                    state_gen = self.generators._modules["dstgen_{}".format(domain)][state_idx]
                    state = domain_state[:,state_idx,:]
                    slot_name = self.slots['domain_slots'][domain][state_idx]
                    if 'request' in slot_name or 'booked' in slot_name or 'is_active' in slot_name:
                        out_states[domain].append(state_gen(state))
                    else:
                        in_state = batch.out_state[domain][state_idx]
                        out_states[domain].append(state_gen(state, in_state, out['embedded_in_txt'], batch.in_txt, batch.in_txt_mask))
            out['out_state_seqs'] = out_states
            return out

class BinaryGenerator(nn.Module):
    def __init__(self, d_model, dropout):
        super(BinaryGenerator, self).__init__()
        self.classifier = nn.Linear(d_model, 1) 
        
    def load_pretrained(self, pretrained_gen):
        self.classifier.weight.data.copy_(pretrained_gen.classifier.weight.data)
        
    def forward(self, state):
        return self.classifier(state)
        
        
class RNNGenerator(nn.Module):
    def __init__(self, d_model, d_out, dropout, src_emb=None, pointer_attn=None):
        super(RNNGenerator, self).__init__()
        if src_emb is not None:
            self.embed = src_emb
            self.all_vocab = True
            self.pointer_gen_W = nn.Linear(d_model*3, 1) 
        else:
            self.embed = Embeddings(d_model, d_out)
            self.all_vocab = False
        self.n_layers = 2
        d_hidden = d_model 
        self.rnn = nn.GRU(d_model, d_hidden, self.n_layers, batch_first=True, dropout=dropout)
        if src_emb is not None:
            self.classifier = src_emb.lut.weight
        else:
            self.classifier = nn.Linear(d_hidden, d_out)
        self.pointer_attn = pointer_attn 
        
    def load_pretrained(self, pretrained_gen):
        if self.all_vocab:
            self.pointer_gen_W.weight.data.copy_(pretrained_gen.pointer_gen_W.weight.data)
            self.pointer_attn.load_pretrained(pretrained_gen.pointer_attn) 
            for w1, w2 in zip(self.rnn.parameters(), pretrained_gen.rnn.parameters()):
                w1.data.copy_(w2.data)
        else:
            pdb.set_trace()
    
    def forward(self, h0, in_txt, encoded_user_uttr, user_uttr, user_uttr_mask):
        in_txt = in_txt[:,:-1]
        embed_in_txt = self.embed(in_txt)
        h0 = h0.unsqueeze(0).expand(self.n_layers, h0.shape[0], h0.shape[1]).contiguous()
        output, hn = self.rnn(embed_in_txt, h0) 
        if self.all_vocab:
            vocab_attn = torch.matmul(output, self.classifier.transpose(1,0))
            self.pointer_attn(output, encoded_user_uttr, encoded_user_uttr, user_uttr_mask)
            pointer_attn = self.pointer_attn.attn.squeeze(1)
            p_vocab = F.softmax(vocab_attn, dim = -1)
            context_index = user_uttr.unsqueeze(1).expand_as(pointer_attn)
            p_context_ptr = torch.zeros(p_vocab.size()).cuda()
            p_context_ptr.scatter_add_(2, context_index, pointer_attn)
            expanded_pointer_attn = pointer_attn.unsqueeze(-1).repeat(1, 1, 1, encoded_user_uttr.shape[-1])
            context_vec = (encoded_user_uttr.unsqueeze(1).expand_as(expanded_pointer_attn) * expanded_pointer_attn).sum(2)
            p_gen_vec = torch.cat([output, context_vec, embed_in_txt], -1)
            vocab_pointer_switches = nn.Sigmoid()(self.pointer_gen_W(p_gen_vec)).expand_as(p_context_ptr)
            p_out = (1 - vocab_pointer_switches) * p_context_ptr + vocab_pointer_switches * p_vocab
            return torch.log(p_out)
        else:
            return F.log_softmax(self.classifier(output), dim=-1)
