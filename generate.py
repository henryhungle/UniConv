#!/usr/bin/env python

import argparse
import logging
import math
import sys
import time
import os
import copy
import pickle as pkl
import json
import numpy as np
import pdb

#import six
#import seaborn
#from scipy.ndimage.filters import gaussian_filter
#import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from utils.data_handler import *  
from preprocess_data import * 
from models.utils import * 
from configs.test_configs import *

def draw(data, x, y, ax, cbar=False):
    center = data.mean().item()
    seaborn.heatmap(data, 
                    xticklabels=x, square=True, yticklabels=y, 
                    cbar=cbar, ax=ax, cmap=cmap) #'coolwarm')

def make_plots(layers, att_heads, net, sublayer_idx, x, y, prefix, combine_scores=False):
    if combine_scores:
        fig, axs = plt.subplots(1, len(layers), figsize=(10,10))
    else:
        fig, axs = plt.subplots(len(layers), len(att_heads), figsize=(40,10))
    for idx_l, l in enumerate(layers):
        for idx_h, h in enumerate(att_heads): 
            if combine_scores and idx_h>0:
                att_scores += net.layers[l].attn[sublayer_idx].attn.data[0,h].data.cpu()
            else:
                att_scores = net.layers[l].attn[sublayer_idx].attn.data[0,h].data.cpu()
            norm_att_scores = gaussian_filter(att_scores, sigma=1)
            if combine_scores:
                if h != att_heads[-1]: continue 
                draw(norm_att_scores, x, y if idx_l==0 else [], ax=axs[idx_l])
            else:
                draw(norm_att_scores, x, y if idx_h==0  else [], ax=axs[idx_l*len(att_heads) + idx_h])
    plt.savefig('{}_combinedscores{}.png'.format(prefix, combine_scores), transparent=True)

# Evaluation routine
def generate_response(model, data, loader, vocab, slots):
    #vocablist = sorted(vocab.keys(), key=lambda s:vocab[s])
    result_dialogues = {}
    model.eval()
    batch_idx = 0
    if args.verbose:
        it = enumerate(loader)
    else:
        it = tqdm(enumerate(loader),total=len(loader), desc="", ncols=0)
    with torch.no_grad():
        for batch_idx, batch in it:
        #for dialogue_id, ref_dialogue in data.items():
            #if '2499.json' not in dialogue_id: 
            #    batch_idx += 1
            #    continue 
            batch.to_cuda()
            dialogue_id = batch.dial_ids[0]
            ref_dialogue = data[dialogue_id]
            result_dialogues[dialogue_id] = {} 
            
            dial_context = display_txt(ref_dialogue['in_txt'])
            result_dialogues[dialogue_id]['dial_context'] = dial_context  
            original_state = display_state(ref_dialogue['state'], slots['domain_slots'])
            result_dialogues[dialogue_id]['state'] = original_state
            original_response = display_txt(ref_dialogue['out_txt_out'][:-1])
            result_dialogues[dialogue_id]['response'] = original_response
            
            dst_output = None
            output = {}
            if train_args.setting in ['dst', 'e2e']:
                if train_args.add_prev_dial_state and not args.gt_previous_bs:
                    dialogue_name, turn_id = dialogue_id.split('_')
                    if int(turn_id) > 1: 
                        assert dialogue_name == last_dialogue_name
                        assert int(turn_id) == (last_turn_id + 1)
                        batch.in_state = last_encoded_state
                        batch.in_state_mask = torch.ones(last_encoded_state.shape).unsqueeze(-2).long().cuda()
                output, dst_output = generate_dst(model, batch, vocab, slots, args)
                predicted_state = display_state(dst_output, slots['domain_slots'])
                
                if train_args.add_prev_dial_state and not args.gt_previous_bs:
                    _, last_domain_token_state = make_bs_txt(dst_output, slots['domain_slots'])
                    last_encoded_state = encode(lang['in+domain+bs']['word2idx'], last_domain_token_state)
                    last_encoded_state = torch.from_numpy(np.asarray(last_encoded_state)).unsqueeze(0).long().cuda()
                    last_dialogue_name = dialogue_name
                    last_turn_id = int(turn_id)
                result_dialogues[dialogue_id]['predicted_state'] = predicted_state         
                if args.verbose:
                    print("Dialogue ID: {}".format(dialogue_id))
                    print("Original state: {}".format(original_state))
                    print("Decoded state: {}".format(predicted_state))

            if train_args.setting in ['c2t', 'e2e']:
                res_output = generate_res(model, batch, vocab, slots, args, output, dst_output)
                result_dialogues[dialogue_id]['predicted_response'] = res_output                
                if args.verbose:
                    print("Original response: {}".format(original_response))
                    for idx, response in res_output.items():
                        print('HYP[{}]: {} ({})'.format(idx+1, response['txt'], response['score']))
                
            '''
            if '2499.json_5' in dialogue_id: 
                in_utt_idx = [int(i) for i in batch.in_utt[0].data.cpu()]
                in_utt_txt = [lang['in+domain+bs']['idx2word'][i] for i in in_utt_idx]
                in_his_idx = [int(i) for i in batch.in_his[0].data.cpu()]
                in_his_txt = [lang['in+domain+bs']['idx2word'][i] for i in in_his_idx]
                in_state_idx = [int(i) for i in batch.in_state[0].data.cpu()]
                in_state_txt = [lang['in+domain+bs']['idx2word'][i] for i in in_state_idx]
                slots_ls = slots['slots']
                slots_ls = [i.replace('inform', 'inf').replace('request', 'req') for i in slots_ls]
                domains_ls = list(slots['domain_slots'].keys())
                ds_ls = []
                for d in domains_ls:
                    for s in slots_ls:
                        ds_ls.append('_'.join([d,s]))
                att_heads = list(range(8))
                make_plots(range(2,4), att_heads, model.general_dst, 3, in_utt_txt, slots_ls, 'slot_utt_', True)
                make_plots(range(2,4), att_heads, model.domain_flow_dst, 2, in_utt_txt, domains_ls, 'domain_utt_', True)
            '''
            #if model.c2t:
            #    predicted_state = original_state
            #else:
            if args.verbose:
                print('-----------------------')
            #if batch_idx == 50: break
    return result_dialogues

# Load params and model
print('Loading training params from ' + args.out_dir + '/' + args.model + '.conf')
train_args = pkl.load(open(args.out_dir + '/' + args.model + '.conf', 'rb'))
print("Loading model weights from " + args.out_dir + '/' + args.model + '_{}.pth.tar'.format(args.tep))
model = torch.load(args.out_dir + '/' + args.model + '_{}.pth.tar'.format(args.tep))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load vocab
prefix = train_args.prefix
lang_dir = 'data{}/multi-woz/{}lang.pkl'.format(train_args.data_version, prefix)
print('Extracting vocab from ' + lang_dir)
lang = pkl.load(open(lang_dir, 'rb'))
# Load data 
encoded_data_dir = 'data{}/multi-woz/{}encoded_data.pkl'.format(train_args.data_version, prefix)
print('Extracting data from ' + encoded_data_dir)
encoded_data = pkl.load(open(encoded_data_dir, 'rb'))
# Load slots 
slots_dir = 'data{}/multi-woz/{}slots.pkl'.format(train_args.data_version, prefix)
print('Extracting slots from ' + slots_dir)
slots = pkl.load(open(slots_dir, 'rb')) 
slots = merge_dst(slots)
test_data = encoded_data['test']
test_data = limit_his_len(test_data, lang['in']['word2idx'], train_args.max_dial_his_len, train_args.only_system_utt)
if train_args.detach_dial_his:
    test_data = detach_dial_his(test_data, lang['in']['word2idx'], train_args.incl_sys_utt)

test_dataloader, test_samples = dh.create_dataset(lang, slots, test_data, False, train_args, 1, 0)
print('#test samples = {}'.format(test_samples))

# generate 
print('-----------------------generate--------------------------')
result = generate_response(model, test_data, test_dataloader, lang, slots)
logging.info('writing results to ' + args.out_dir + '/' + args.output)
json.dump(result, open(args.out_dir + '/' + args.output, 'w'), indent=4)
