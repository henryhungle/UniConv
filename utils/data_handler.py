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

'''
def get_npy_shape(filename):
    # read npy file header and return its shape
    with open(filename, 'rb') as f:
        if filename.endswith('.pkl'):
            shape = pickle.load(f).shape
        else:
            major, minor = np.lib.format.read_magic(f)
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
    return shape

def align_vocab(pretrained_vocab, vocab, pretrained_weights):
    for module, module_wt in pretrained_weights.items():
        for layer, layer_wt in module_wt.items():
            if 'embed' in layer:
                print("Aligning word emb for layer {} in module {}...".format(layer, module))
                print("Pretrained emb of shape {}".format(layer_wt.shape))
                emb_dim = layer_wt.shape[1]
                embs = np.zeros((len(vocab), emb_dim), dtype=np.float32)
                count = 0 
                for k,v in vocab.items():
                    if k in pretrained_vocab:
                        embs[v] = layer_wt[pretrained_vocab[k]]
                    else:
                        count += 1 
                pretrained_weights[module][layer] = embs
                print("Aligned emb of shape {}".format(embs.shape))
                print("Number of unmatched words {}".format(count))
    return pretrained_weights

def hash_string(s):
    return abs(hash(s)) % (10 ** 8)

def normalise_word_vector(word_vector, norm=1.0):
    word_vector /= math.sqrt(sum(word_vector**2) + 1e-6)
    word_vector *= norm 
    return word_vector

def xavier_vector(word, D=300):
    """ 
    Returns a D-dimensional vector for the word. 

    We hash the word to always get the same vector for the given word. 
    """
    seed_value = hash_string(word)
    np.random.seed(seed_value)

    neg_value = - math.sqrt(6)/math.sqrt(D)
    pos_value = math.sqrt(6)/math.sqrt(D)

    rsample = np.random.uniform(low=neg_value, high=pos_value, size=(D,))
    norm = np.linalg.norm(rsample)
    rsample_normed = rsample/norm
    return rsample_normed

def get_word_emb(vocab, pretrained_word_emb, cache=None):
    if 'bert' in pretrained_word_emb:
        return get_bert_emb(vocab, pretrained_word_emb, cache)
    words = set(vocab.keys())
    print("Loading pretrained word emb {}...".format(pretrained_word_emb))
    with open(pretrained_word_emb, "r") as f:
        lines = f.readlines()
    emb_dim = len(lines[0].split())-1
    print("Embed dim is {}".format(emb_dim))
    embs = np.zeros((len(words), emb_dim), dtype=np.float32)
    embs_dict = dict()
    for line in lines:
        tokens = line.split()
        word = tokens[0]
        emb = np.asarray([float(i) for i in tokens[1:]])
        emb = normalise_word_vector(emb)
        embs_dict[word] = emb 
        if word in vocab:
            #emb = [float(i) for i in tokens[1:]]
            idx = vocab[word]
            embs[idx] = emb
            words.remove(word)
    print("{} Unknown words".format(len(words)))
    for word in words:
        length = len(word)
        vec = None
        for i in range(1, length)[::-1]:
            if word[:i] in embs_dict and word[i:] in embs_dict:
                vec = embs_dict[word[:i]] + embs_dict[word[i:]]
                print("Found component word emb: {}".format(word))
                break
        if vec is None:
            vec = xavier_vector(word, emb_dim)
            print("Adding new word emb: {}".format(word))
        idx = vocab[word]
        embs[idx] = vec 
    embs = np.asarray(embs)
    print("pretrained word embedding of shape {}".format(embs.shape))
    return embs 

def get_vocabulary(dataset_file, cutoff=1, include_caption='none'):
    vocab = {'<unk>':0, '<blank>':1, '<sos>':2, '<eos>':3}
    dialog_data = json.load(open(dataset_file, 'r'))
    word_freq = {}
    for dialog in dialog_data['dialogs']:
        if include_caption == 'caption' or include_caption == 'summary' or include_caption == 'caption,summary':
            if include_caption == 'caption' or include_caption == 'summary':
                caption = dialog[include_caption]
            else:
                caption = dialog['caption'] + dialog['summary']
            for word in caption.split():
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1
        for key in ['question', 'answer']:
            for turn in dialog['dialog']:
                for word in turn[key].split():
                    if word in word_freq:
                        word_freq[word] += 1
                    else:
                        word_freq[word] = 1
    cutoffs = [1,2,3,4,5]
    for cutoff in cutoffs:
        vocab = {'<unk>':0, '<blank>':1, '<sos>':2, '<eos>':3}
        for word, freq in word_freq.items():
            if freq > cutoff:
                vocab[word] = len(vocab) 
        print("{} words for cutoff {}".format(len(vocab), cutoff))
    pdb.set_trace()
    return vocab

# +1: padding positional token = 0
# +1: first <blank> or <caption> = 1 
def get_st_vocabulary(max_history_length=-1):
    if max_history_length > 0: 
        return (max_history_length+1+1)
    else:
        return (10+1+1)

def words2ids(str_in, vocab):
    words = str_in.split()
    sentence = np.ndarray(len(words)+2, dtype=np.int32)
    sentence[0]=vocab['<sos>']
    for i,w in enumerate(words):
        if w in vocab:
            sentence[i+1] = vocab[w]
        else:
            sentence[i+1] = vocab['<unk>']
    sentence[-1]=vocab['<eos>']
    return sentence
    
def remove_domain(data, remove_domain, split):
    domain_idx = -1
    out = {}
    print("Removing/Keeping domain {} from split {}".format(remove_domain, split))
    print("Number of samples {}".format(len(data)))
    for dial_id, dial in data.items():
        if domain_idx == -1:
            domains = list(dial['state'].keys())
            domain_idx = domains.index(remove_domain)
        active_domains = dial['active_domains']
        if split == 'train':
            if active_domains[domain_idx] == 1:
                continue
            out[dial_id] = dial
        else:
        #    if active_domains[domain_idx] != 1:
        #        continue
            out[dial_id] = dial 
    print("Number of remaining samples {}".format(len(out)))
    return out 

def get_active_domains(domain_indices):
    return [d for idx, d in enumerate(DOMAINS) if domain_indices[idx]==1]

def get_bs_for_request(bs, req, limited_req):
    for domain, domain_bs in bs.items():
        for slot, value in domain_bs.items():
            if 'request' in slot:
                slot_type, slot_name = slot.split('_', 1)
                if domain not in req: req[domain] = []
                req[domain].append(slot_name)
                if slot_name in LIMITED_REQUESTABLES:
                    if domain not in limited_req: limited_req[domain] = []
                    limited_req[domain].append(slot_name)
    return req, limited_req

def seq1toseq2_mask(seq1, seq2, pad):
    temp = (seq1 != pad).unsqueeze(-1).expand((seq1.shape[0], seq1.shape[1], seq2.shape[-1]))
    output = temp & (seq2 != pad).unsqueeze(-2).expand((seq2.shape[0], seq1.shape[1], seq2.shape[-1]))
    return output

def make_ngram(features, mask, ngram, pad=False):
        if features.shape[1] < ngram:
            new_features = torch.zeros(features.shape[0], ngram, features.shape[2])
            new_features[:,-features.shape[1]:,:] = features
            ngram_features = new_features.unfold(1, ngram, 1).sum(-1)
            ngram_mask = torch.zeros(features.shape[0], 1, 1)
        else:
            ngram_features = features.unfold(1, ngram, 1).sum(-1)
            #ngram_mask = mask.unfold(2, ngram, 1).sum(-1) == 2
            if mask.shape[1] == 1:
                ngram_mask = mask[:,:,ngram-1:]
            else:
                ngram_mask = mask[:,ngram-1:,ngram-1:]
        if pad and ngram_features.shape != features.shape:
            tensor=torch.zeros_like(features)
            tensor[:,-ngram_features.shape[1]:,:] = ngram_features
            ngram_features = tensor
            tensor=torch.zeros_like(mask)
            if mask.shape[1] == 1:
                tensor[:,:,-ngram_mask.shape[2]:] = ngram_mask
            else:
                tensor[:,-ngram_mask.shape[1]:,-ngram_mask.shape[2]:] = ngram_mask
            ngram_mask = tensor
        return ngram_features.cuda(), ngram_mask.cuda()

def add_active_domain_slots(slots):
    slots['slots'] = ['is_active'] + slots['slots']
    for domain in slots['domain_slots'].keys():
        slots['domain_slots'][domain] = ['is_active'] + slots['domain_slots'][domain]
        slots['slots_idx'][domain] = [0] + [i+1 for i in slots['slots_idx'][domain]]
    return slots
'''

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
        #if count == 14:
        #    break

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
    #if domain == 'taxi' and 'contact number' in res:
    return out
