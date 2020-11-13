import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math, copy, time
import pdb
#from torchtext import data, datasets
from models.label_smoothing import *
import utils.data_handler as dh
from utils.dataset import subsequent_mask
from preprocess_data import get_db_pointer
from preprocess_data import make_bs_txt
from preprocess_data import encode as encode_bs

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer, fixed_dst):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self.fixed_dst = fixed_dst
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) * \
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

def get_dst_criterion(domain_slots, smoothing, args, state_vocab, src_vocab):
    criterions = {}
    if args.share_dst_gen:
        criterions['inf_dst'] = LabelSmoothing(size=len(src_vocab), padding_idx=src_vocab['<pad>'], smoothing=smoothing).cuda()
        criterions['req_dst'] = nn.BCEWithLogitsLoss()
    else:
        for domain, slots_ls in domain_slots.items():
            criterions[domain] = []
            for slot in slots_ls:
                if 'request' in slot or 'booked' in slot or 'is_active' in slot:
                    criterions[domain].append(nn.BCEWithLogitsLoss())
                else:
                    if 'booking_' in slot:
                        vocab = src_vocab
                        criterions[domain].append(LabelSmoothing(size=len(vocab), 
                                                                 padding_idx=vocab['<pad>'], smoothing=smoothing).cuda())
                    else:
                        vocab = state_vocab[domain][slot]
                        criterions[domain].append(LabelSmoothing(size=len(vocab), 
                                                                 padding_idx=vocab['<pad>'], smoothing=smoothing).cuda())
    return criterions

class LossCompute:    
    def __init__(self, model, dst_criterion, nlg_criterion, dp_criterion, opt, args, slots):
        if args.setting in ['dst', 'e2e']: self.dst_generator = model.dst_net.dst_generator
        if args.setting in ['c2t', 'e2e']: 
            self.nlg_generator = model.nlg_net.res_generator
            if args.sys_act: self.act_generator = model.dp_net
        self.dst_criterion = dst_criterion
        self.nlg_criterion = nlg_criterion
        self.dp_criterion = dp_criterion
        self.opt = opt
        self.args = args 
        self.slots = slots
        
    def get_dst_loss(self, out, batch, losses):
        criterion = self.dst_criterion
        domain_slots = self.slots['domain_slots']
        loss = 0
        if self.args.share_dst_gen:
            req_loss = criterion['req_dst'](out['out_req_states'].reshape(-1), batch.out_req_state.reshape(-1).float())
            losses['dst_req'] = req_loss.item()
            loss += req_loss
            inf_loss = criterion['inf_dst'](
                out['out_inf_states'].contiguous().view(-1, out['out_inf_states'].size(-1)),
                batch.out_inf_state[:,:,1:].contiguous().view(-1))/batch.ntokens_state
            losses['dst_inf'] = inf_loss.item()
            loss += inf_loss
        else:
            predicts = out['out_state_seqs']
            for domain, domain_criterions in criterion.items():
                for idx, domain_criterion in enumerate(domain_criterions):
                    predict = predicts[domain][idx]
                    target = batch.out_state[domain][idx]
                    slot = domain_slots[domain][idx]
                    if 'booked' in slot or 'request' in slot or 'is_active' in slot:
                        slot_loss = domain_criterion(predict.squeeze(-1), target.float())
                        loss += slot_loss
                        losses['dst_req'] += slot_loss.item()
                    else:
                        slot_ntokens = batch.ntokens_state[domain][slot]
                        slot_loss = domain_criterion(
                            predict.contiguous().view(-1, predict.size(-1)), target[:,1:].contiguous().view(-1))/slot_ntokens
                        loss += slot_loss
                        losses['dst_inf'] += slot_loss.item()
        return losses, loss
    
    def get_nlg_loss(self, out, batch, losses):
        loss = 0
        nlg_out = out['out_res_seqs']
        nlg_res_loss = self.nlg_criterion(
            nlg_out.contiguous().view(-1, nlg_out.size(-1)), batch.out_txt_y.contiguous().view(-1))/batch.ntokens_res
        losses['nlg_res'] = nlg_res_loss.item()
        loss += nlg_res_loss
        return losses, loss 
        
    def get_dp_loss(self, out, batch, losses):
        loss = 0
        dp_out = out['out_act_logits']
        dp_act_loss = self.dp_criterion(dp_out.view(-1), batch.out_act.view(-1))
        losses['dp_act'] = dp_act_loss.item()
        loss += dp_act_loss
        return losses, loss 
    
    def __call__(self, batch, out, is_eval):
        loss = 0      
        losses = {}
        
        losses['dst_inf'] = 0
        losses['dst_req'] = 0
        if self.args.setting in ['dst', 'e2e']:
            #if self.args.fixed_dst:
                #with torch.no_grad():
                #    out = self.dst_generator(batch, out)
                #losses, _ = self.get_dst_loss(out, batch, losses)
            #else:
            out = self.dst_generator(batch, out)
            losses, dst_loss = self.get_dst_loss(out, batch, losses)
            loss += dst_loss
        
        losses['nlg_res'] = 0
        losses['dp_act'] = 0
        if self.args.setting in ['c2t', 'e2e']:
            out = self.nlg_generator(batch, out)
            losses, nlg_loss = self.get_nlg_loss(out, batch, losses)
            loss += nlg_loss
            if self.args.sys_act:
                out = self.act_generator(batch, out)
                losses, dp_loss = self.get_dp_loss(out, batch, losses)
                loss += dp_loss

        if not is_eval:
            loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
            
        return losses

def binary_decode(h0, nn):
    out = torch.sigmoid(nn(h0)).item()
    if out > 0.5:
        return 1
    else:
        return 0
    
def greedy_decode(h0, nn, word2idx, idx2word, max_len, batch, out):
    sos = word2idx['<sos>']
    in_sos = torch.ones(1,1).fill_(sos).long().cuda()
    in_state = in_sos
    for i in range(max_len-1):
        in_st = torch.cat([in_state, torch.ones(1,1).fill_(sos).long().cuda()], dim=-1)
        prob = nn(h0, in_st, out['embedded_in_txt'], batch.in_txt, batch.in_txt_mask)
        _, next_word = torch.max(prob, dim=-1)
        in_state = torch.cat([in_sos, next_word], dim=-1)
        if next_word[0][-1] == word2idx['<eos>']: break
    out = []
    for state in in_state[0]:
        if state.item() == word2idx['<unk>'] or state.item() == word2idx['<pad>']: continue
        if state.item() not in idx2word: pdb.set_trace()
        out.append(idx2word[state.item()])
    return out

def generate_res(model, batch, lang, slots, args, out, dst_out):
    if not args.gt_db_pointer:
        query_state = dh.get_bs_for_query(dh.display_state(dst_out, slots['domain_slots'], None))
        new_pointer = get_db_pointer(query_state, True)
        batch.out_ptr = torch.from_numpy(new_pointer).unsqueeze(0).long().cuda()
    #if True: #model.args.literal_bs:
    if not args.gt_previous_bs: 
        _, curr_state_txt = make_bs_txt(dst_out, slots['domain_slots'])
        batch.in_curr_state = torch.tensor(encode_bs(lang['in+domain+bs']['word2idx'], curr_state_txt)).unsqueeze(0).cuda()
        batch.in_curr_state_mask = None
    out = model.nlg_net.encode(batch, out) 
    response_out = beam_search_decode(model, batch, out, lang, args)
    return response_out

def generate_dst(model, batch, lang, slots, args):
    out = {} 
    out = model.dst_net.forward(batch, out)
    state = out['out_states']
    if model.args.domain_flow:
        domain_states = {}
        count = 0
        for domain, indices in slots['slots_idx'].items():
            domain_states[domain] = state[:,count,indices,:]
            count += 1
    else:
        domain_states = {}
        for domain, indices in slots['merged_slots_idx'].items():
            domain_states[domain] = state[:,indices,:]
    dst_out = {}
    for domain, domain_slots in slots['domain_slots'].items():
        dst_out[domain] = []
        for slot_idx, domain_slot in enumerate(domain_slots):
            h0 = domain_states[domain][:,slot_idx,:]
            if 'booked' in domain_slot or 'request' in domain_slot or 'is_active' in domain_slot:
                if model.args.share_dst_gen:
                    generator = model.dst_net.dst_generator.bi_generator
                else:
                    generator = model.dst_net.dst_generator.generators._modules["dstgen_{}".format(domain)][slot_idx]
                decoded_state = binary_decode(h0, generator)
            else:
                if model.args.share_dst_gen:
                    generator = model.dst_net.dst_generator.rnn_generator
                    decoded_state = greedy_decode(
                        h0, generator, lang['in+domain+bs']['word2idx'], 
                        lang['in+domain+bs']['idx2word'], args.dst_max_len, batch, out)
                else:
                    generator = model.dst_net.dst_generator.generators._modules["dstgen_{}".format(domain)][slot_idx]
                    if 'booking_' in domain_slot:
                        decoded_state = greedy_decode(
                            h0, generator, lang['in+domain+bs']['word2idx'], 
                        lang['in+domain+bs']['idx2word'], args.dst_max_len, batch, out)
                    else:
                        decoded_state = greedy_decode(
                            h0, generator, lang['dst']['word2idx'][domain][domain_slot], 
                            lang['dst']['idx2word'][domain][domain_slot], args.dst_max_len, batch, out)
            dst_out[domain].append(decoded_state)
    return out, dst_out   

def beam_search_decode(model, batch, encoded, lang, args):
    word2idx, idx2word = lang['out']['word2idx'], lang['out']['idx2word']
    max_len, beam, penalty, nbest, min_len = args.res_max_len, args.beam, args.penalty, args.nbest, args.res_min_len
    sos = word2idx['<sos>']
    eos = word2idx['<eos>']
    unk = word2idx['<unk>']
    ds = torch.ones(1, 1).fill_(sos).long()
    hyplist=[([], 0., ds)]
    best_state=None
    comp_hyplist=[]
    for l in range(max_len):
        new_hyplist = []
        argmin = 0
        for out, lp, st in hyplist:
            batch.out_txt = Variable(st).long().cuda()
            batch.out_mask = Variable(subsequent_mask(st.size(1))).long().cuda()
            if model.args.sys_act:
                in_res_embed = model.nlg_net.out_embed(batch.out_txt)
                exp_prior = model.nlg_net.sys_act_prior.unsqueeze(0).unsqueeze(0).expand(in_res_embed.shape[0], -1, model.nlg_net.sys_act_prior.shape[0])
                embedded_in_res = torch.cat([exp_prior, in_res_embed], dim=1)
            else:
                embedded_in_res = model.nlg_net.out_embed(batch.out_txt)
            if model.args.detach_dial_his:
                layer_norm_idx = 3
            else:
                layer_norm_idx = 2
            encoded['embedded_in_res'] = model.nlg_net.layer_norm[layer_norm_idx](embedded_in_res)
            encoded = model.nlg_net.decode_response(batch, encoded)
            output = model.nlg_net.res_generator(batch, encoded)
            logp = output['out_res_seqs'][:,-1]
            #logp = model.res_generator(output[:, -1])
            lp_vec = logp.cpu().data.numpy() + lp
            lp_vec = np.squeeze(lp_vec)
            if l >= min_len:
                new_lp = lp_vec[eos] + penalty * (len(out) + 1)
                comp_hyplist.append((out, new_lp))
                if best_state is None or best_state < new_lp:
                    best_state = new_lp
            for o in np.argsort(lp_vec)[::-1]:
                if o == unk or o == eos:
                    continue
                new_lp = lp_vec[o]
                if len(new_hyplist) == beam:
                    if new_hyplist[argmin][1] < new_lp:
                        new_st = torch.cat([st, torch.ones(1,1).long().fill_(int(o))], dim=1)
                        new_hyplist[argmin] = (out + [o], new_lp, new_st)
                        argmin = min(enumerate(new_hyplist), key=lambda h:h[1][1])[0]
                    else:
                        break
                else:
                    new_st = torch.cat([st, torch.ones(1,1).long().fill_(int(o))], dim=1)
                    new_hyplist.append((out + [o], new_lp, new_st))
                    if len(new_hyplist) == beam:
                        argmin = min(enumerate(new_hyplist), key=lambda h:h[1][1])[0]
        hyplist = new_hyplist

    if len(comp_hyplist) > 0:
        pred_out = sorted(comp_hyplist, key=lambda h: -h[1])[:nbest]
    else:
        pred_out = [([], 0)]
        best_state = None
        return None
    out = {}
    for n in range(min(nbest, len(pred_out))):
        pred = pred_out[n]
        hypstr = []
        for w in pred[0]:
            if w == eos:
                break
            hypstr.append(idx2word[w])
        hypstr = " ".join(hypstr)
        out[n] = {
          'txt': hypstr,
          'score': pred[1]
        }
    return out

