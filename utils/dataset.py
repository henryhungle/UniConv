import copy
import logging
import sys
import time
import os
import six
import pickle
import json
import numpy as np
import pdb 
from tqdm import tqdm
import torch 
import torch.utils.data as Data
from torch.autograd import Variable

class Dataset(Data.Dataset):
    def __init__(self, data_info):
        self.dial_id = data_info['dial_id']
        self.turn_id = data_info['turn_id']
        self.history = data_info['history']
        self.in_txt = data_info['in_txt']
        self.prev_st = data_info['prev_st']
        self.curr_st = data_info['curr_st']
        self.out_st = data_info['out_st']
        self.out_ptr = data_info['out_ptr']
        self.out_txt_in = data_info['out_txt_in']
        self.out_txt_out = data_info['out_txt_out']
        if 'act' in data_info:
            self.out_act = data_info['act']
        else:
            self.out_act = None
        self.num_total_seqs = len(data_info['dial_id'])
                
    def __getitem__(self, index): 
        item_info = {
            'dial_id':self.dial_id[index], 
            'turn_id': self.turn_id[index],
            'history': self.history[index],
            'in_txt': self.in_txt[index], 
            'prev_st': self.prev_st[index],
            'curr_st': self.curr_st[index],
            'out_st': self.out_st[index],
            'out_ptr': self.out_ptr[index],
            'out_txt_in': self.out_txt_in[index],
            'out_txt_out': self.out_txt_out[index],
            }
        if self.out_act is not None and len(self.out_act)>0:
            item_info['out_act'] = self.out_act[index]
        return item_info
    
    def __len__(self):
        return self.num_total_seqs
    
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def make_std_mask(tgt, pad, sys_act=False):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    if sys_act: 
        pdb.set_trace()
    return tgt_mask

def prepare_domain_slot_mask(in_slots, in_domains, slots_idx):
    if len(in_domains)==0: return []
    batch_size = in_slots.shape[0]
    mask_len = in_slots.shape[1]*in_domains.shape[1]
    mask = torch.zeros(mask_len)
    count = 0
    for domain, idx in slots_idx.items():
        real_idx = [count*in_slots.shape[1]+i for i in idx]
        mask[real_idx] = 1
        count += 1
    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, mask_len)
    
def add_sys_act_mask(out_mask):
    temp = torch.ones(out_mask.shape[0], out_mask.shape[1], 1)
    out = torch.cat([temp.cuda().int(), out_mask.int()], dim=-1)
    temp = torch.zeros(out.shape[0], 1, out.shape[-1])
    temp[:,:,0] = temp[:,:,0]+1
    out = torch.cat([temp.cuda().int(), out], dim=1)
    return out
    
class Batch:
    def __init__(self, in_domains, in_slots, h_batch, u_batch, in_state, in_curr_state, 
                  out_state, out_ptr, res_in_batch, res_out_batch, ntokens_state, slots_idx,
                  req_idx, inf_idx, req_batch, inf_batch, dial_ids, out_act):
        self.dial_ids = dial_ids 
        self.in_slots = in_slots
        self.in_domains = in_domains
        self.in_his = h_batch
        self.in_his_mask = (h_batch != 2).unsqueeze(-2)
        self.in_txt = u_batch
        self.in_txt_mask = (u_batch != 2).unsqueeze(-2)
        self.in_state = in_state
        self.in_state_mask = (in_state != 2).unsqueeze(-2)
        self.in_curr_state = in_curr_state
        self.in_curr_state_mask = (in_curr_state != 2).unsqueeze(-2)
        self.out_state = out_state
        self.out_ptr = out_ptr
        self.out_txt = res_in_batch
        self.out_mask = make_std_mask(res_in_batch, 2)
        self.out_txt_y = res_out_batch
        self.ntokens_res = (res_out_batch != 2).data.sum()
        self.ntokens_state = ntokens_state
        self.domain_slot_mask = prepare_domain_slot_mask(in_slots, in_domains, slots_idx)
        self.req_idx = req_idx
        self.inf_idx = inf_idx
        self.out_req_state = req_batch
        self.out_inf_state = inf_batch
        self.out_act = out_act
        
    def to_cuda(self):
        self.in_slots = self.in_slots.to('cuda', non_blocking=True)
        self.in_domains = self.in_domains.to('cuda', non_blocking=True)
        self.in_his = self.in_his.to('cuda', non_blocking=True)
        self.in_his_mask = self.in_his_mask.to('cuda', non_blocking=True)
        self.in_txt = self.in_txt.to('cuda', non_blocking=True)
        self.in_txt_mask = self.in_txt_mask.to('cuda', non_blocking=True)
        self.in_state = self.in_state.to('cuda', non_blocking=True)
        self.in_state_mask = self.in_state_mask.to('cuda', non_blocking=True)
        self.in_curr_state = self.in_curr_state.to('cuda', non_blocking=True)
        self.in_curr_state_mask = self.in_curr_state_mask.to('cuda', non_blocking=True)
        if self.out_state is not None:
            out_state = {}
            for domain, domain_states in self.out_state.items():
                out_state[domain] = []
                for state in domain_states:
                    out_state[domain].append(state.to('cuda', non_blocking=True))
            self.out_state = out_state
        if self.out_req_state is not None:
            self.out_req_state = self.out_req_state.contiguous().to('cuda', non_blocking=True)
            self.out_inf_state = self.out_inf_state.contiguous().to('cuda', non_blocking=True)
        self.out_ptr = self.out_ptr.to('cuda', non_blocking=True)
        self.out_txt = self.out_txt.to('cuda', non_blocking=True)
        self.out_txt_y = self.out_txt_y.to('cuda', non_blocking=True)
        self.out_mask = self.out_mask.to('cuda', non_blocking=True)
        if len(self.in_domains)>0:
            self.domain_slot_mask = self.domain_slot_mask.contiguous().to('cuda', non_blocking=True)
        if self.out_act is not None:
            self.out_act = self.out_act.contiguous().to('cuda', non_blocking=True)
        
                
def collate_fn(data, vocab, slots, args):
    def pad_seq(seqs, pad_token, max_length=-1):
        if max_length<0:
            max_length = max([len(s) for s in seqs])
        output = []
        for seq in seqs:
            result = np.ones(max_length, dtype=np.int)*pad_token
            result[:len(seq)] = seq 
            output.append(result)
        return output 

    def prepare_data(seqs):
        return torch.from_numpy(np.asarray(seqs)).long()
            
    def prepare_dst(dst_batch, slots, args):
        if args.share_dst_gen:
            domain_slot_names = slots['domain_slots']
            inf_idx = []
            req_idx = []
            nb_slots = len(slots['slots'])
            if args.domain_flow:
                count = 0
                slots_idx = slots['slots_idx']
                for domain, slot_indices in slots_idx.items():
                    for name_index, index in enumerate(slot_indices):
                        slot_name = domain_slot_names[domain][name_index]
                        if 'request' in slot_name or 'booked' in slot_name or 'is_active' in slot_name:
                            req_idx.append(count*nb_slots+index)
                        else:
                            inf_idx.append(count*nb_slots+index)
                    count += 1
            else:
                slots_idx = slots['merged_slots_idx']
                for domain, slot_indices in slots_idx.items():
                    for name_index, index in enumerate(slot_indices):
                        slot_name = domain_slot_names[domain][name_index]
                        if 'request' in slot_name or 'booked' in slot_name or 'is_active' in slot_name:
                            req_idx.append(index)
                        else:
                            inf_idx.append(index)
            inf_batch = []
            req_batch = []
            num_tokens = 0
            max_inf_len = 0
            for domain, domain_slots in dst_batch.items():
                for idx, slot_values in enumerate(domain_slots):
                    slot_name = domain_slot_names[domain][idx]
                    if 'request' in slot_name or 'booked' in slot_name or 'is_active' in slot_name:
                        req_batch.append(slot_values)
                    else:
                        inf_batch.append(slot_values)
                        lens = [len(i) for i in slot_values]
                        if max(lens)>max_inf_len: max_inf_len=max(lens)
                        #if domain not in num_tokens: num_tokens[domain] = {}
                        #num_tokens[domain][slot_name] = sum(lens)
                        num_tokens += sum(lens)
            req_batch = prepare_data(np.asarray(req_batch).transpose())
            for idx, inf in enumerate(inf_batch):
                inf_batch[idx] = prepare_data(pad_seq(inf, 2, max_length=max_inf_len))
            inf_batch = torch.cat([i.unsqueeze(0) for i in inf_batch], dim=0).permute(1, 0, 2)
            return req_idx, inf_idx, req_batch, inf_batch, num_tokens 
        else:
            num_tokens = {}
            domain_slot_names = slots['domain_slots']
            for domain, domain_slots in dst_batch.items():
                for idx, slot_values in enumerate(domain_slots):
                    slot_name = domain_slot_names[domain][idx]
                    if 'request' in slot_name or 'booked' in slot_name or 'is_active' in slot_name:
                        dst_batch[domain][idx] = prepare_data(slot_values)
                    else:
                        lens = [len(i) for i in slot_values]
                        if domain not in num_tokens:
                            num_tokens[domain] = {}
                        num_tokens[domain][slot_name] = sum(lens)
                        dst_batch[domain][idx] = prepare_data(pad_seq(slot_values, 2))
            return dst_batch, num_tokens
    
    def prepare_slots(slots, n_seqs, vocab, domain_flow):
        if domain_flow:
            slots_ls = slots['slots']
            return [[vocab['in+domain+bs']['word2idx']['<{}>'.format(s)] for s in slots_ls] for i in range(n_seqs)]
        else:
            slots_ls = slots['merged_slots']
            return [[[vocab['in+domain+bs']['word2idx'][s[0]],vocab['in+domain+bs']['word2idx'][s[1]]]
                for s in slots_ls] for i in range(n_seqs)]
                     
    def prepare_domains(slots, n_seqs, vocab, domain_flow):
        if domain_flow: 
            domains = slots['domain_slots'].keys()
            return [[vocab['in+domain+bs']['word2idx']['<{}>'.format(d)] for d in domains] for i in range(n_seqs)]
        else:
            return [] 
        
    def prepare_act(acts, nb_labels):
        out = []
        for act in acts:
            labels = torch.tensor(act).unsqueeze(0)
            out.append(torch.zeros(labels.size(0), nb_labels).scatter_(1, labels, 1.))
        out = torch.cat(out, dim=0)
        return out 
        
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]    
    h_batch = prepare_data(pad_seq(item_info['history'], 2))
    u_batch = prepare_data(pad_seq(item_info['in_txt'], 2))
    prev_st_batch = prepare_data(pad_seq(item_info['prev_st'], 2))
    curr_st_batch = prepare_data(pad_seq(item_info['curr_st'], 2))
    res_in_batch = prepare_data(pad_seq(item_info['out_txt_in'], 2))
    res_out_batch = prepare_data(pad_seq(item_info['out_txt_out'], 2))
    out_ptr_batch = prepare_data(pad_seq(item_info['out_ptr'], 2))
    out_dst_batch = {}
    encoded_states = item_info['out_st']
    for encoded_state in encoded_states[:1]:
        for domain, domain_states in encoded_state.items():
            out_dst_batch[domain] = []
            for state in domain_states:
                out_dst_batch[domain].append([])
    for encoded_state in encoded_states:
        for domain, domain_states in encoded_state.items():
            for state_idx, domain_state in enumerate(domain_states):
                out_dst_batch[domain][state_idx].append(domain_state)
    dst_batch, req_idx, inf_idx, req_batch, inf_batch = None, None, None, None, None 
    if args.share_dst_gen: 
        req_idx, inf_idx, req_batch, inf_batch, dst_tokens = prepare_dst(out_dst_batch, slots, args)
    else:
        dst_batch, dst_tokens = prepare_dst(out_dst_batch, slots, args)
    n_seqs = len(res_in_batch)
    slots_batch = prepare_data(prepare_slots(slots, n_seqs, vocab, args.domain_flow))
    domains_batch = prepare_data(prepare_domains(slots, n_seqs, vocab, args.domain_flow))    
    if hasattr(args, 'sys_act') and args.sys_act:
        act_batch = prepare_act(item_info['out_act'], len(vocab['act']['word2idx']))
    else:
        act_batch = None 
    batch = Batch(domains_batch, slots_batch, h_batch, u_batch, prev_st_batch, curr_st_batch, 
                  dst_batch, out_ptr_batch, res_in_batch, res_out_batch, dst_tokens, slots['slots_idx'],
                  req_idx, inf_idx, req_batch, inf_batch, item_info['dial_id'], act_batch)

    return batch




