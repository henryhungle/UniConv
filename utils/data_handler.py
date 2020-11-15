#!/usr/bin/env python

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
from functools import partial
from tqdm import tqdm
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.dataset import *


DOMAINS = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']
GOAL2REQUEST={
  'cartype': 'type',
#  'entrancefee': 'entrance'
}
LIMITED_REQUESTABLES = ['phone', 'address', 'postcode', 'reference', 'id', 'trainID']

def limit_his_len(data, vocab, max_his_len, only_system_utt):
    if max_his_len < 0: return data 
    sos = vocab['<sos>']
    eos = vocab['<eos>']
    for dial_id, dial in data.items():
        in_txt = dial['encoded_in_txt']
        sos_idx = [i for i,ii in enumerate(in_txt) if ii == sos]
        dial_len = dial['turn_id'] - 1 
        if dial_len > max_his_len:
            his_sos_idx = sos_idx[-(max_his_len*2+1)]
            data[dial_id]['in_txt'] = dial['in_txt'][his_sos_idx:]
            data[dial_id]['encoded_in_txt'] = dial['encoded_in_txt'][his_sos_idx:]
            data[dial_id]['encoded_in_txt_by_in+bs'] = dial['encoded_in_txt_by_in+bs'][his_sos_idx:]
            data[dial_id]['encoded_in_txt_by_in+domain+bs'] = dial['encoded_in_txt_by_in+domain+bs'][his_sos_idx:]
        if dial_len > 0 and max_his_len == 1 and only_system_utt:
            in_txt = data[dial_id]['encoded_in_txt']
            sos_idx = [i for i,ii in enumerate(in_txt) if ii == sos]
            his_sos_idx = sos_idx[-2]
            data[dial_id]['in_txt'] = data[dial_id]['in_txt'][his_sos_idx:]
            data[dial_id]['encoded_in_txt'] = data[dial_id]['encoded_in_txt'][his_sos_idx:]
            data[dial_id]['encoded_in_txt_by_in+bs'] = data[dial_id]['encoded_in_txt_by_in+bs'][his_sos_idx:]
            data[dial_id]['encoded_in_txt_by_in+domain+bs'] = data[dial_id]['encoded_in_txt_by_in+domain+bs'][his_sos_idx:]
    return data

def detach_dial_his(data, vocab, incl_sys_utt=False):
    sos = vocab['<sos>']
    eos = vocab['<eos>']
    for dial_id, dial in data.items():
        in_txt = dial['encoded_in_txt']
        in_txt_by_in_bs = dial['encoded_in_txt_by_in+bs']
        in_txt_by_in_domain_bs = dial['encoded_in_txt_by_in+domain+bs']
        if incl_sys_utt:
            last_sos_idx = len(in_txt) - in_txt[::-1].index(sos) - 1 
            temp = in_txt[last_sos_idx:]
            last_sos_idx = len(temp) - temp[::-1].index(sos) - 1 
        else:
            last_sos_idx = len(in_txt) - in_txt[::-1].index(sos) - 1 
        in_utt = in_txt[last_sos_idx:]
        in_his = in_txt[:last_sos_idx]
        in_his_by_in_bs = in_txt_by_in_bs[:last_sos_idx]
        in_utt_by_in_bs = in_txt_by_in_bs[last_sos_idx:]
        in_his_by_in_domain_bs = in_txt_by_in_domain_bs[:last_sos_idx]
        in_utt_by_in_domain_bs = in_txt_by_in_domain_bs[last_sos_idx:]
        if len(in_his) == 0: #no dialogue history i.e. 1st dialogue turn 
            in_his = [sos, eos]
            in_his_by_in_bs = [sos, eos]
            in_his_by_in_domain_bs = [sos, eos]
        data[dial_id]['encoded_in_his'] = in_his
        data[dial_id]['encoded_in_utt'] = in_utt
        data[dial_id]['encoded_in_his_by_in+bs'] = in_his_by_in_bs
        data[dial_id]['encoded_in_utt_by_in+bs'] = in_utt_by_in_bs
        data[dial_id]['encoded_in_his_by_in+domain+bs'] = in_his_by_in_domain_bs
        data[dial_id]['encoded_in_utt_by_in+domain+bs'] = in_utt_by_in_domain_bs
    return data 

def merge_dst(slots, domain_token_bs=True, share_slots_bs_vocab=True):
    out = []
    idx_out = {}
    counter = 0
    for domain, domain_slots in slots['domain_slots'].items():
        idx_out[domain] = []
        for slot in domain_slots:
            if share_slots_bs_vocab:
                if domain_token_bs:
                    out.append(['<{}>'.format(domain), '<{}>'.format(slot)])
                else:
                    out.append('<{}_{}>'.format(domain, slot))
            else:
                out.append('<{}_{}>'.format(domain, slot))
            idx_out[domain].append(counter)
            counter += 1
    slots['merged_slots'] = out
    slots['merged_slots_idx'] = idx_out
    return slots

def add_dst_vocab(slots):
    slots_vocab = {}
    for domain, domain_slots in slots['domain_slot_values'].items():
        for slot, values in domain_slots.items():
            if 'request' in slot or 'booked' in slot or 'is_active' in slot or 'booking_' in slot: continue
            if domain not in slots_vocab: slots_vocab[domain] = {}
            slots_vocab[domain][slot] = sorted(values)
            if 'none' not in slots_vocab[domain][slot]: slots_vocab[domain][slot].append('none')
    slots['domain_slots_vocab'] = slots_vocab
    return slots 

def make_classify_labels(dial, slots):
    encoded_state = dial['encoded_state']
    original_state = display_state(dial['state'], slots['domain_slots'])
    for domain, domain_slots in slots['domain_slots'].items():
        for slot_idx, slot in enumerate(domain_slots):
            if 'request' in slot or 'booked' in slot or 'is_active' in slot or 'booking_' in slot: continue
            vocab = slots['domain_slots_vocab'][domain][slot]
            new_state = vocab.index('none')
            if domain in original_state and slot in original_state[domain]: 
                new_state = vocab.index(original_state[domain][slot])
            encoded_state[domain][slot_idx] = new_state
    return encoded_state
    
def create_dataset(vocab, slots, data, shuffle, args, bs=-1, nb_workers=-1):
    batch_size, detach_dial_his, add_prev_dial_state, num_workers = args.batch_size, args.detach_dial_his, args.add_prev_dial_state, args.num_workers
    if bs>0: batch_size=bs
    if nb_workers>0: num_workers=nb_workers
    out = {}
    keys = ['dial_id', 'turn_id', 'history', 'in_txt', 'prev_st', 'curr_st', 'out_st', 'out_ptr', 'out_txt_in', 'out_txt_out', 'act']
    for key in keys:
        out[key] = []
    data_iter = tqdm(data.items(), total=len(data), ncols=0)
    count = 0
    for dial_id, dial in data_iter:
        out['dial_id'].append(dial_id)
        out['turn_id'].append(dial['turn_id'])
        if detach_dial_his:
            # assume domain_token_state=TRUE
            out['history'].append(dial['encoded_in_his_by_in+domain+bs'])
            out['in_txt'].append(dial['encoded_in_utt_by_in+domain+bs'])
        else:
            out['history'].append([])
            out['in_txt'].append(dial['encoded_in_txt_by_in+domain+bs'])
        if add_prev_dial_state:
            out['prev_st'].append(dial['encoded_previous_domain_token_state_by_in+domain+bs'])
            out['curr_st'].append(dial['encoded_current_domain_token_state_by_in+domain+bs'])
        else:
            out['prev_st'].append([])
            out['curr_st'].append([])
        out['out_txt_in'].append(dial['encoded_out_txt_in'])
        out['out_txt_out'].append(dial['encoded_out_txt_out'])
        if hasattr(args, 'dst_classify') and args.dst_classify:
            encoded_state = make_classify_labels(dial, slots)
            out['out_st'].append(dial['encoded_state'])
            pdb.set_trace()
        else:
            out['out_st'].append(dial['encoded_state'])
        out['out_ptr'].append(dial['pointer_vector'])
        if hasattr(args, 'sys_act') and args.sys_act:
            out['act'].append(dial['encoded_act'])
        
        count += 1 
        if args.small_data and count == 14:
            break

    dataset = Dataset(out)  
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  collate_fn=partial(collate_fn, vocab=vocab, slots=slots, args=args),
                                                  num_workers=num_workers,
                                                  pin_memory=True)
    return data_loader, len(out['dial_id'])

def display_txt(tokens):
    return ' '.join(tokens)

def display_state(state, domain_slots, active_domains=None):
    out = {}
    if active_domains is not None: 
        for domain, domain_states in state.items():
            state[domain] = [active_domains[DOMAINS.index(domain)]] + state[domain]
    for domain, domain_states in state.items():
        for state_idx, domain_state in enumerate(domain_states):
            slot_name = domain_slots[domain][state_idx]
            if type(domain_state) == list:
                if len(domain_state)==1 or (domain_state[0] == '<sos>' and len(domain_state)>1 and domain_state[1] == '<eos>'):
                    continue 
                if domain not in out: out[domain] = {}
                out[domain][slot_name] = display_txt(domain_state[1:-1])
            else:
                if domain_state == 0: continue 
                if domain not in out: out[domain] = {}
                out[domain][slot_name] = domain_state
    return out 

def get_goal_for_query_request(goal):
    query_goal = {}
    request_goal = {}
    limited_request_goal = {}
    for domain, domain_goal in goal.items():
        query_goal[domain] = {}
        request_goal[domain] = []
        limited_request_goal[domain] = []
        for slot, value in domain_goal.items():
            if slot == 'request':
                for v in value:
                    if v in GOAL2REQUEST:
                        request_goal[domain].append(GOAL2REQUEST[v])
                    else:
                        request_goal[domain].append(v)
                        if v in LIMITED_REQUESTABLES: limited_request_goal[domain].append(v)
            else:
                query_goal[domain][slot] = value 
    return query_goal, request_goal, limited_request_goal

def get_bs_for_query(bs):
    out = {}
    for domain, domain_bs in bs.items():
        out[domain] = {}
        for slot, value in domain_bs.items():
            if 'request' in slot or 'booking' in slot: continue 
            slot_type, slot_name = slot.split('_', 1)
            out[domain][slot_name] = value
    return out

def get_requestables(domain, res):
    tokens = res.split()
    out = []
    for token in tokens:
        if '{}_'.format(domain) in token:
            _, slot = token.split('_', 1)
            out.append(slot)
    return out
