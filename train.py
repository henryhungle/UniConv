#!/usr/bin/env python
import logging
import math
import sys
import time
import os
import json
import numpy as np
import pickle as pkl
import threading
import pdb 
from tqdm import tqdm 
import torch
import torch.nn as nn

import utils.data_handler as dh
from models.dialogue_dnn import *
from models.label_smoothing import * 
from models.utils import * 
from configs.train_configs import *

def run_epoch(loader, vocab, epoch, slots, model, loss_compute, is_eval=False):
    total_losses = {}
    total_losses['dst_inf'] = 0 
    total_losses['dst_req'] = 0  
    total_losses['nlg_res'] = 0
    total_losses['dp_act'] = 0
    it = tqdm(enumerate(loader),total=len(loader), desc="epoch {}/{}".format(epoch+1, args.num_epochs), ncols=0)
    for j, batch in it:  
        batch.to_cuda()
        if is_eval:
            with torch.no_grad():
                out = model.forward(batch)
                losses = loss_compute(batch, out, is_eval)
        else:
            out = model.forward(batch)
            losses = loss_compute(batch, out, is_eval)
        total_losses['dst_inf'] += losses['dst_inf']
        total_losses['dst_req'] += losses['dst_req']
        total_losses['nlg_res'] += losses['nlg_res']
        total_losses['dp_act'] += losses['dp_act']
        if (j+1) % args.report_interval == 0 and not is_eval:
            print("Epoch {} Step {} DST_Inf_Loss {} DST_Req_Loss {} NLG_Res_Loss {} DP_Act_Loss {}".format
                (epoch+1, j+1, losses['dst_inf'], losses['dst_req'], losses['nlg_res'], losses['dp_act']))
            with open(args.out_dir + '/' + args.model + '_train.csv', "a") as f:
                f.write("{},{},{},{},{},{}\n".format(
                    epoch+1, j+1, losses['dst_inf'], losses['dst_req'], losses['nlg_res'], losses['dp_act']))
        #break
    total_losses['dst_inf'] /= len(loader)
    total_losses['dst_req'] /= len(loader)
    total_losses['nlg_res'] /= len(loader)
    total_losses['dp_act'] /= len(loader)
    return total_losses

prefix = args.prefix
# Extracting language files 
lang_dir = 'data{}/multi-woz/{}lang.pkl'.format(args.data_version, prefix)
print('Extracting vocab from ' + lang_dir)
lang = pkl.load(open(lang_dir, 'rb'))
# Extracting vocabulary files 
encoded_data_dir = 'data{}/multi-woz/{}encoded_data.pkl'.format(args.data_version, prefix)
print('Extracting data from ' + encoded_data_dir)
encoded_data = pkl.load(open(encoded_data_dir, 'rb'))
# Extracting domains-slots files
slots_dir = 'data{}/multi-woz/{}slots.pkl'.format(args.data_version, prefix)
print('Extracting slots from ' + slots_dir)
slots = pkl.load(open(slots_dir, 'rb'))
slots = dh.merge_dst(slots) 
if args.dst_classify:
    slots = dh.add_dst_vocab(slots)

pretrained_word_emb = None 
train_data = encoded_data['train']
valid_data = encoded_data['val']
train_data = dh.limit_his_len(train_data, lang['in']['word2idx'], args.max_dial_his_len, args.only_system_utt)
valid_data = dh.limit_his_len(valid_data, lang['in']['word2idx'], args.max_dial_his_len, args.only_system_utt)
if args.detach_dial_his:
    train_data = dh.detach_dial_his(train_data, lang['in']['word2idx'], args.incl_sys_utt)
    valid_data = dh.detach_dial_his(valid_data, lang['in']['word2idx'], args.incl_sys_utt)
    
# report data summary
print('#in_vocab = {}'.format(len(lang['in+domain+bs']['word2idx'])))
print('#out_vocab = {}'.format(len(lang['out']['word2idx'])))
dst_vocab = lang['dst']['word2idx']
print("Domain and slot vocab statistics")
for domain, slot_values in dst_vocab.items():
    display = []
    print("DOMAIN: {}".format(domain))
    for slot, values in slot_values.items(): 
        display.append("({},{})".format(slot, len(values)))
    print(' '.join(display))

# make datasets
train_dataloader, train_samples = dh.create_dataset(lang, slots, train_data, True, args)
print('#train samples = {}'.format(train_samples))
valid_dataloader, valid_samples = dh.create_dataset(lang, slots, valid_data, False, args)
print('#validation samples = {}'.format(valid_samples))

# create_model
model = make_model(lang, slots, args)
model.cuda()

# Set Criterions, losses, and optimizer 
dst_criterion = None
nlg_criterion = None
dp_criterion = None
if args.setting in ['dst', 'e2e']:
    dst_criterion = get_dst_criterion(slots['domain_slots'], 0.1, args, state_vocab=lang['dst']['word2idx'],
                                      src_vocab=lang['in+domain+bs']['word2idx'])
if args.setting in ['c2t', 'e2e']:
    tgt_vocab = lang['out']['word2idx']
    nlg_criterion = LabelSmoothing(size=len(tgt_vocab), padding_idx=tgt_vocab['<pad>'], smoothing=0.1).cuda()
    if args.sys_act:
        dp_criterion = nn.BCEWithLogitsLoss()
model_opt = NoamOpt(args.d_model, 1, args.warmup_steps, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9), args.fixed_dst)
train_loss = LossCompute(model, dst_criterion, nlg_criterion, dp_criterion, opt=model_opt, args=args, slots=slots)
valid_loss = LossCompute(model, dst_criterion, nlg_criterion, dp_criterion, opt=None, args=args, slots=slots)

# save meta parameters
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)
pkl.dump(args, open(args.out_dir + '/' + args.model + '.conf', 'wb'))
with open(args.out_dir + '/' + args.model + '_params.txt', 'w') as f: 
    for arg in vars(args):
        f.write("{}={}\n".format(arg, getattr(args, arg)))

# initialize status parameters
modelext = '.pth.tar'
min_valid_loss = 1.0e+10
bestmodel_num = 0

# save results 
with open(args.out_dir + '/' + args.model + '_val.csv', 'w') as f:
    f.write('epoch,split,dst_inf_loss,dst_req_loss,nlg_res_loss,dp_act_loss\n')
with open(args.out_dir + '/' + args.model + '_train.csv', 'w') as f:  
    f.write('epoch,step,dst_inf_loss,dst_req_loss,nlg_res_loss,dp_act_loss\n')
print("Saving training results to {}".format(args.out_dir + '/' + args.model + '_train.csv'))
print("Saving val results to {}".format(args.out_dir + '/' + args.model + '_val.csv'))   

for epoch in range(args.num_epochs):
    # training 
    model.train()
    train_losses = run_epoch(train_dataloader, lang, epoch, slots, model, train_loss)
    print("Epoch {} Train DST_Inf_Loss {} DST_Req_Loss {} NLG_Res_Loss {} DP_Act_Loss {}".format
                (epoch+1, train_losses['dst_inf'], train_losses['dst_req'], train_losses['nlg_res'],  train_losses['dp_act']))
    
    # test on validation set 
    model.eval()
    valid_losses = run_epoch(valid_dataloader, lang, epoch, slots, model, valid_loss, is_eval=True)
    print("Epoch {} Val DST_Inf_Loss {} DST_Req_Loss {} NLG_Res_Loss {} DP_Act_Loss {}".format
                (epoch+1, valid_losses['dst_inf'], valid_losses['dst_req'], valid_losses['nlg_res'], valid_losses['dp_act']))

    with open(args.out_dir + '/' + args.model + '_val.csv',"a") as f:
        f.write("{},train,{},{},{},{}\n".format(
            epoch+1, train_losses['dst_inf'], train_losses['dst_req'], train_losses['nlg_res'], train_losses['dp_act']))
        f.write("{},val,{},{},{},{}\n".format(
            epoch+1,valid_losses['dst_inf'], valid_losses['dst_req'], valid_losses['nlg_res'], valid_losses['dp_act']))  
        
    val_loss = valid_losses['dst_inf'] + valid_losses['nlg_res']
    if min_valid_loss > val_loss:
        bestmodel_num = epoch+1
        print('validation loss reduced {} -> {}'.format(min_valid_loss, val_loss))
        min_valid_loss = val_loss
        modelfile = args.out_dir + '/' + args.model + '_best' + modelext
        torch.save(model, modelfile)
        
print('the best model is epoch %d.' % bestmodel_num)
