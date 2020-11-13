# -*- coding: utf-8 -*-
import copy
import json
import os
import re
import shutil
import urllib
from collections import OrderedDict
import operator
from io import BytesIO
from zipfile import ZipFile
from tqdm import tqdm
import pickle as pkl
import numpy as np
import pdb 
import utils.db_pointer as db_pointer
import utils.delexicalizer as delexicalizer
from utils.nlp import normalize
import random 
import copy 
np.set_printoptions(precision=3)
np.random.seed(2)
# GLOBAL VARIABLES

def load_dialogues_ls(file_dir):
    dialogues = []
    fin = open(file_dir)
    for line in fin:
        dialogues.append(line[:-1])
    fin.close()
    return dialogues

#MAX_LENGTH = 50
excluded_domains = []
DOMAINS = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']
DOMAINS = [i for i in DOMAINS if i not in excluded_domains]
DONTCARE = ['dont care', 'dontcare', "don't care", "do n't care"]
GOAL_TYPES = ['info', 'fail_info', 'reqt', 'book', 'fail_book']
SOS = '<sos>'
EOS = '<eos>'
PAD = '<pad>'
UNK = '<unk>'
DEFAULT_TOKENS=[SOS,EOS,PAD,UNK]
IN_WORDS_FREQ_MIN=-1
OUT_WORDS_FREQ_MIN=2

def get_db_pointer(metadata, normalized=False):
    """Create database pointer for all related domains."""
    domains = ['restaurant', 'hotel', 'attraction', 'train']
    pointer_vector = np.zeros(6 * len(domains))
    for domain in domains:
        _, num_entities = db_pointer.query_result(domain, metadata, normalized)
        pointer_vector = db_pointer.one_hot_vector(num_entities, domain, pointer_vector)
    return pointer_vector

def create_dict(word_freqs, min_freq=-1):
    sorted_word_freqs = sorted(word_freqs.items(), key=operator.itemgetter(1))
    sorted_word_freqs = OrderedDict(sorted_word_freqs)
    word2idx = {}
    for ii, ww in enumerate(DEFAULT_TOKENS):
        word2idx[ww] = ii
    index = len(DEFAULT_TOKENS)
    for ww, freq in sorted_word_freqs.items():
        if ww in word2idx: continue 
        if min_freq>0 and freq < min_freq: continue 
        word2idx[ww] = index 
        index += 1 
    idx2word = {v:k for k,v in word2idx.items()}
    out = {
      'sorted_word_freqs': sorted_word_freqs,
      'word2idx': word2idx,
      'idx2word': idx2word
    }
    return out

def create_dst_dict(dsv_dict):
    word2idx = {}
    for domain, slot_values in dsv_dict.items():
        word2idx[domain] = {}
        for slot, values in slot_values.items():
            word2idx[domain][slot] = {} 
            for ii, ww in enumerate(DEFAULT_TOKENS):
                word2idx[domain][slot][ww] = ii
            index = len(DEFAULT_TOKENS)
            for value in values:
                tokens = value.split()
                for token in tokens:
                    if token not in word2idx[domain][slot]:
                        word2idx[domain][slot][token] = index 
                        index += 1 
    idx2word = {}
    for domain, slot_dicts in word2idx.items():
        idx2word[domain] = {}
        for slot, lang_dict in slot_dicts.items():
            idx2word[domain][slot] = {v:k for k,v in lang_dict.items()}
    out = {
        'word2idx': word2idx,
        'idx2word': idx2word
    }
    return out 

def create_bs_dict(word_freqs, dsv_dict):
    word2idx = {}
    for ii,ww in enumerate(DEFAULT_TOKENS):
        word2idx[ww] = ii
    index = len(DEFAULT_TOKENS)
    for ww, freq in word_freqs.items():
        if ww in word2idx: continue 
        word2idx[ww] = index
        index += 1 
    for domain, slot_values in dsv_dict.items():
        for slot, values in slot_values.items():
            for value in values:
                tokens =  value.split()
                for token in tokens:
                    if token not in word2idx:
                        word2idx[token] = index 
                        index += 1 
    idx2word = {v:k for k,v in word2idx.items()}
    out = {
        'word_freqs': word_freqs,
        'word2idx': word2idx,
        'idx2word': idx2word
    }
    return out 

def get_goal(did, dialogue_goal, dsv_dict):
    goal = {}
    booking_goal = {}
    for domain in DOMAINS:
        domain_goal = dialogue_goal[domain]
        domain_sv = dsv_dict[domain]
        if len(domain_goal) == 0:
            continue
        goal[domain] = {}
        booking_goal[domain] = {}
        for k,v in domain_goal.items():
            if k not in GOAL_TYPES:
                if k in DOMAINS:
                    if v != False:
                        goal[domain]['name'] = v 
                else:
                    print("ID: {} Odd goal type in domain goal {} vs. {}".format(did, domain_goal, GOAL_TYPES))
        for k,v in domain_goal['info'].items():
            if k in domain_sv.keys() or domain == 'taxi':
                goal[domain][k] = normalize(v) 
            else:
                print("ID: {} DB-goal unmatched slot type in domain info goal {} vs. {}".format(did, domain_goal['info'], domain_sv.keys()))
        for k,v in domain_goal['fail_info'].items():
            if k in domain_sv.keys() or domain == 'taxi':
                goal[domain]['fail_' + k] = normalize(v)
            else:
                print("ID: {} DB-goal unmatched slot type in domain fail-info goal {} vs. {}".format(did, domain_goal['fail_info'], domain_sv.keys()))
        if 'reqt' in domain_goal:
            goal[domain]['request'] = []
            for k in domain_goal['reqt']:
                if k in domain_sv.keys() or domain == 'taxi':
                    if ' ' in k: # phone number, entrance fee 
                        k = k.replace(' ', '')
                    goal[domain]['request'].append(k)
                else:
                    print("ID: {} DB-goal unmatched slot type in domain reqt goal {} vs. {}".format(did, domain_goal['reqt'], domain_sv.keys()))
        if 'book' in domain_goal:
            for k,v in domain_goal['book'].items():
                if k in ['pre_invalid', 'invalid']:
                    continue
                booking_goal[domain][k] = normalize(v)
        if 'fail_book' in domain_goal:
            for k,v in domain_goal['fail_book'].items():
                if k in ['pre_invalid', 'invalid']:
                    continue 
                booking_goal[domain]['fail_'+k] = normalize(v)
    return goal, booking_goal

def get_state(did, tid, utt_state, dsv_dict):
    state = {}
    unnormalized_state = {}
    booking_state = {} 
    for domain in DOMAINS:
        domain_state = utt_state[domain]
        domain_sv = dsv_dict[domain]
        b_state = domain_state['book']
        for k,v in b_state.items():
            if len(v) > 0:
                if domain not in booking_state:
                    booking_state[domain] = {}
                if k == 'booked':
                    bookings = []
                    for b in v:
                        booking = {}
                        for b_k,b_v in b.items():
                            booking[b_k] = normalize(b_v)
                        bookings.append(booking)
                    booking_state[domain][k] = bookings
                    continue 
                booking_state[domain][k] = normalize(v)
        info_state = domain_state['semi']
        for k,v in info_state.items():
            if len(v) == 0 or v == 'not mentioned' or v in DONTCARE:
                continue 
            if domain not in state:
                state[domain] = {}
                unnormalized_state[domain] = {}
            if k in domain_sv.keys() or domain == 'taxi':
                state[domain][k] = normalize(v)
                unnormalized_state[domain][k] = v 
            else:
                print("ID: {} Turn: {} DB-State unmatched slot type in domain info state {} vs. {}".format(did, tid, info_state, domain_sv.keys()))
    return state, booking_state, unnormalized_state

def get_act(did, tid, acts):
    output_act = {}
    if did[:-5] not in acts:
        #print("ID: {} No annotation of dialogue act".format(did))
        return None, output_act
    if str(tid) not in acts[did[:-5]]:
        #print("ID: {} Turn {} No annotation of dialogue act".format(did, tid))
        return None, output_act 
    sys_act = acts[did[:-5]][str(tid)]
    if sys_act == 'No Annotation':
        #print("ID: {} Turn: {} No annotation of dialogue act".format(did, tid))
        return None, output_act 
    for domain_act,slots in sys_act.items():
        domain,act = domain_act.split('-')
        domain = domain.lower()
        act = act.lower()
        if domain not in output_act:
            output_act[domain] = {}
        if domain == 'general':
            output_act[domain][act] = ''
            continue 
        if domain == 'booking':
            if act == 'request':
                output_act[domain][act] = []
                for slot in slots:
                    output_act[domain][act].append(slot[0].lower())
            else:
                output_act[domain][act] = {}
                for slot in slots:
                    if slot[0] == 'none': continue 
                    output_act[domain][act][slot[0].lower()] = slot[1].lower()
            continue 
        if act =='request':
            output_act[domain][act] = []
            for slot in slots:
                output_act[domain][act].append(slot[0].lower())
        if act == 'inform':
            output_act[domain][act] = {}
            for slot in slots:
                output_act[domain][act][slot[0].lower()] = normalize(slot[1])
    return sys_act, output_act 


def delexicalize_data(ignore_booking, save_data, save_prefix, excluded_domains):
    _, dsv_dict = delexicalizer.prepare_slot_values_independent()
    delex_data = {}
    delex_data['dsv_dict'] = dsv_dict 
    delex_data['delex_dialogues'] = {}
    data = json.load(open('data{}/MULTIWOZ2/data.json'.format(DATA_VERSION)))
    acts = json.load(open('data{}/MULTIWOZ2/dialogue_acts.json'.format(DATA_VERSION)))
    
    count = 0
    for dialogue_name in tqdm(data):
        dialogue = data[dialogue_name]
        goal, booking_goal = get_goal(dialogue_name, dialogue['goal'], dsv_dict)
        if len(excluded_domains) > 0:
            for d in excluded_domains:
                if d in goal.keys(): 
                    continue 
        delex_data['delex_dialogues'][dialogue_name] = {}
        delex_data['delex_dialogues'][dialogue_name]["goal"] = goal
        delex_data['delex_dialogues'][dialogue_name]["booking_goal"] = booking_goal 
        delex_data['delex_dialogues'][dialogue_name]["original_goal"] = dialogue["goal"]
        delex_data['delex_dialogues'][dialogue_name]["dialogue"] = [] 
        for idx in range(0,len(dialogue['log']),2):
            human_turn = dialogue['log'][idx]
            system_turn = dialogue['log'][idx+1]
            turn_id = int(idx/2+1)
            state, booking_state, unnormalized_state = get_state(dialogue_name, turn_id, dialogue['log'][idx+1]['metadata'], dsv_dict)
            original_act, act = get_act(dialogue_name, turn_id, acts)
            is_last_turn = idx >= len(dialogue['log'])-2
            human_txt, _, human_inform_slots, human_request_slots, human_booking_slots = delexicalizer.delexicalize(human_turn['text'], dsv_dict, goal, booking_goal, state, booking_state, None, is_last_turn, False, ignore_booking)
            system_txt, delex_system_txt, _, _, _  = delexicalizer.delexicalize(system_turn['text'], dsv_dict, goal, booking_goal, state, booking_state, act, is_last_turn, True, ignore_booking)
            original_pointer_vector = get_db_pointer(dialogue['log'][idx+1]['metadata'], False)
            pointer_vector = get_db_pointer(state, True)
            #original_pointer_vector = []
            #pointer_vector = []
            item = {
              'human_txt': human_txt,
              'human_inform_slots': human_inform_slots,
              'human_request_slots': human_request_slots,
              'human_booking_slots': human_booking_slots,
              'system_txt': system_txt,
              'delex_system_txt': delex_system_txt,
              'state': state,
              'unnormalized_state': unnormalized_state,
              'booking_state': booking_state,
              'original_state': dialogue['log'][idx+1]['metadata'],
              'act': act,
              'original_act': original_act,
              'pointer_vector': pointer_vector,
              'original_pointer_vector': original_pointer_vector
            }
            delex_data['delex_dialogues'][dialogue_name]['dialogue'].append(item)
        count += 1
        #if count == 100: break
    if save_data:
        with open('data{}/multi-woz/{}delex_data.pkl'.format(DATA_VERSION, save_prefix), 'wb') as outfile:
            pkl.dump(delex_data, outfile)
    return delex_data

def get_all_slots(data, dsv_dict, ignore_booking=False, save_data=False, save_prefix=''):
    all_slots = set()
    domain_slots = {}
    domain_slot_values = {}
    for domain in DOMAINS:
        domain_slots[domain] = set()
        domain_slot_values[domain] = {}
    for dialogue_name, dialogue_data in data.items():
        dialogue = dialogue_data['dialogue']
        for turn in dialogue:
            inform_slots = turn['human_inform_slots']
            for s,v in inform_slots.items():
                domain,slot = s.split('_',1)
                all_slots.add(slot)
                if domain in excluded_domains: pdb.set_trace()
                domain_slots[domain].add(slot)
                if dialogue_name not in TEST_LS and dialogue_name not in VAL_LS:  
                    if not slot in domain_slot_values[domain]:
                        domain_slot_values[domain][slot] = set()
                    _, slot_name = slot.split('_')
                    domain_slot_values[domain][slot].add(v)
            request_slots = turn['human_request_slots']
            for s in request_slots:
                domain,slot = s.split('_',1)
                all_slots.add(slot)
                domain_slots[domain].add(slot)
            if ignore_booking: continue 
            booking_slots = turn['human_booking_slots']
            for s,v in booking_slots.items():
                domain,slot = s.split('_',1)
                all_slots.add(slot)
                domain_slots[domain].add(slot)
                if dialogue_name not in TEST_LS and dialogue_name not in VAL_LS and 'booked' not in slot:
                    if not slot in domain_slot_values:
                        domain_slot_values[domain][slot] = set()
                    _, slot_name = slot.split('_')
                    domain_slot_values[domain][slot].add(v)
    for domain, slots in domain_slots.items():
        for slot in slots:
            if 'request' in slot or 'booked' in slot: continue #binary values 
            _, slot_name = slot.split('_')
            dsv_values = dsv_dict[domain][slot_name]
            for v in dsv_values:
                domain_slot_values[domain][slot].add(v)
    all_slots = sorted(list(all_slots))
    slots_idx = {}
    for k,v in domain_slots.items():
        domain_slots[k] = sorted(list(v))
        slots_idx[k] = [all_slots.index(s) for s in domain_slots[k]]
    out = {
      "slots": all_slots,
      "slots_idx": slots_idx, 
      "domain_slots": domain_slots,
      "domain_slot_values": domain_slot_values
    }
    if save_data: pkl.dump(out, open('data{}/multi-woz/{}slots.pkl'.format(DATA_VERSION, save_prefix), 'wb'))
    return out 

def summarize_bs(domain_slots, inform_slots, request_slots, booking_slots, ignore_booking):
    out = {}
    active_domains = []
    for domain in DOMAINS:
        out[domain] = []
        for slot in domain_slots[domain]:
            if 'request' not in slot and 'booked' not in slot:
                out[domain].append([SOS]) #text value 
            else:
                out[domain].append(0) #binary value
        active_domains.append(0)
    for s,v in inform_slots.items():
        domain,slot = s.split('_',1)
        slot_idx = domain_slots[domain].index(slot)
        out[domain][slot_idx].extend(v.split())
        active_domains[DOMAINS.index(domain)] = 1
    for s in request_slots:
        domain,slot = s.split('_',1)
        slot_idx = domain_slots[domain].index(slot)
        out[domain][slot_idx] = 1 
        active_domains[DOMAINS.index(domain)] = 1
    if not ignore_booking:
        for s,v in booking_slots.items():
            domain,slot = s.split('_',1)
            slot_idx = domain_slots[domain].index(slot)
            if 'booked' in slot:
                out[domain][slot_idx] = 1 
            else:
                out[domain][slot_idx].extend(v.split())
            active_domains[DOMAINS.index(domain)] = 1 
    for domain in DOMAINS:
        for idx, slot in enumerate(domain_slots[domain]):
            if 'request' not in slot and 'booked' not in slot: 
                out[domain][idx].append(EOS) #text value 
    return out, active_domains 

def sample_dials(dials, num_samples):
    dialogue_names = dials.keys()
    samples = random.sample(dialogue_names, num_samples)
    out = {k:dials[k] for k in samples}
    return out 

def make_bs_txt(bs, domain_slots):
    out = [SOS]
    domain_token_out = [SOS]
    for domain, states in bs.items():
        active_domain = False
        for idx, state in enumerate(states):
            slot = domain_slots[domain][idx]
            if type(state) == list: 
                if state[0] == SOS and (len(state)>1 and state[1] == EOS): continue 
                out += state[1:-1] 
                domain_token_out += state[1:-1]
            else:
                if state == 0: continue 
            active_domain = True
            out += ["<{}_{}>".format(domain, slot)]
            domain_token_out += ["<{}>".format(slot)]
        if active_domain: domain_token_out += ["<{}>".format(domain)]
    out += [EOS]
    domain_token_out += [EOS]
    return out, domain_token_out

def divide_data(data, domain_slots, num_samples=-1, ignore_booking=False, save_data=False, save_prefix='', all_vocab=False, sys_act=False):
    """Given test and validation sets, divide
    the data for three different sets"""
    test_dials = {}
    val_dials = {}
    train_dials = {}
    word_freqs_in = OrderedDict()
    word_freqs_bs = OrderedDict()
    word_freqs_domain_token_bs = OrderedDict()
    word_freqs_out = OrderedDict()
    word_sys_act = set()
    test_word_in = set()
    test_word_domain_token_bs = set()
    test_word_out = set()
    val_word_in = set()
    val_word_domain_token_bs = set()
    val_word_out = set()
    for dialogue_name in tqdm(data):
        dialogue = data[dialogue_name]['dialogue']
        for num_turn in range(1, len(dialogue)+1):
            in_txt = []
            out_txt = []
            in_txt_unique = []
            for t in range(num_turn):
                turn = dialogue[t]
                in_txt += [SOS] + turn['human_txt'].split() + [EOS]
                if t != num_turn-1:
                    in_txt += [SOS] + turn['system_txt'].split() + [EOS]
            in_txt_unique = [SOS] + dialogue[num_turn-1]['human_txt'].split() + [EOS]
            if num_turn > 1: 
                in_txt_unique += [SOS] + dialogue[num_turn-2]['system_txt'].split() + [EOS]
            if dialogue_name in TEST_LS:
                for w in in_txt_unique:
                    test_word_in.add(w)
                    if all_vocab:
                        if w not in word_freqs_in:
                            word_freqs_in[w] = 0 
                        word_freqs_in[w] += 1
            elif dialogue_name in VAL_LS:
                for w in in_txt_unique:
                    val_word_in.add(w)
                    if all_vocab:
                        if w not in word_freqs_in:
                            word_freqs_in[w] = 0 
                        word_freqs_in[w] += 1
            else:
                for w in in_txt_unique:
                    if w not in word_freqs_in:
                        word_freqs_in[w] = 0 
                    word_freqs_in[w] += 1 
            out_txt = [SOS] + turn['delex_system_txt'].split() + [EOS]
            if dialogue_name in TEST_LS:
                for w in out_txt:
                    test_word_out.add(w)
            elif dialogue_name in VAL_LS:
                for w in out_txt:
                    val_word_out.add(w)
            else:
                for w in out_txt:
                    if w not in word_freqs_out:
                        word_freqs_out[w] = 0 
                    word_freqs_out[w] += 1 
            pointer_vector = turn['pointer_vector']
            if num_turn == 1:
                previous_bs, previous_domain_token_bs = [SOS, EOS], [SOS, EOS]
                previous_active_domains = None
            else:
                previous_bs, previous_domain_token_bs = make_bs_txt(bs, domain_slots)
                previous_active_domains = active_domains
            if dialogue_name in TEST_LS:
                for w in previous_domain_token_bs:
                    test_word_domain_token_bs.add(w)
            elif dialogue_name in VAL_LS:
                for w in previous_domain_token_bs:
                    val_word_domain_token_bs.add(w)
            else: 
                for w in previous_bs:
                    if w not in word_freqs_bs:
                        word_freqs_bs[w] = 0
                    word_freqs_bs[w] += 1 
                for w in previous_domain_token_bs:
                    if w not in word_freqs_domain_token_bs:
                        word_freqs_domain_token_bs[w] = 0
                    word_freqs_domain_token_bs[w] += 1 
            bs, active_domains = summarize_bs(domain_slots, turn['human_inform_slots'], turn['human_request_slots'], turn['human_booking_slots'], ignore_booking) 
            current_bs, current_domain_token_bs = make_bs_txt(bs, domain_slots)
            if sys_act:
                if turn['original_act'] is not None:
                    act = [i.strip().lower() for i in turn['original_act'].keys()]
                else:
                    act = ['noact']
                for a in act:
                    word_sys_act.add(a) 
            else:
                act = []
            dial = {
                'dialouge_id': dialogue_name,
                'turn_id': num_turn,
                'in_txt': in_txt,
                'out_txt_in': out_txt[:-1],
                'out_txt_out': out_txt[1:],
                'pointer_vector': pointer_vector,
                'state': bs,
                'previous_state': previous_bs,
                'previous_domain_token_state': previous_domain_token_bs,
                'current_state': current_bs,
                'current_domain_token_state': current_domain_token_bs,
                'active_domains': active_domains,
                'previous_active_domains': previous_active_domains,
                'act': act 
            }
            dial_key = '_'.join([dialogue_name, str(num_turn)])
            if dialogue_name in TEST_LS:
                test_dials[dial_key] = dial
            elif dialogue_name in VAL_LS:
                val_dials[dial_key] = dial
            else:
                train_dials[dial_key] = dial

    # save all dialogues
    if num_samples > 0:  
        train_dials = sample_dials(train_dials, num_samples)
        test_dials = sample_dials(test_dials, num_samples)
        val_dials = sample_dials(val_dials, num_samples)
    dials = {
        'train': train_dials,
        'test': test_dials,
        'val': val_dials
    }
    if save_data:
        if num_samples <= 0:
            pkl.dump(dials, open('data{}/multi-woz/{}dials.pkl'.format(DATA_VERSION, save_prefix), 'wb'))
        else:
            pkl.dump(dials, open('data{}/multi-woz/{}dials_small.pkl'.format(DATA_VERSION, save_prefix), 'wb'))
    return dials, word_freqs_in, word_freqs_bs, word_freqs_domain_token_bs, word_freqs_out, word_sys_act

def combine_lang(lang1, lang2):
    word2idx = copy.deepcopy(lang1)
    idx = len(lang1)
    for ww, ii in lang2.items():
        if ww not in word2idx:
            word2idx[ww] = idx
            idx += 1 
    idx2word = {v:k for k,v in word2idx.items()}
    out = {
        'word2idx': word2idx,
        'idx2word': idx2word
    }
    return out 

def build_dicts(word_freqs_in, word_freqs_bs, word_freqs_domain_token_bs, word_freqs_out, domain_slot_values, save_data=False, save_prefix='', sys_act=False, word_sys_act=set(), share_inout=False):
    lang_in = create_dict(word_freqs_in, IN_WORDS_FREQ_MIN)
    lang_out  = create_dict(word_freqs_out, OUT_WORDS_FREQ_MIN)
    lang_bs = create_dict(word_freqs_bs)
    lang_domain_token_bs = create_dict(word_freqs_domain_token_bs)
    lang_in_bs = combine_lang(lang_in['word2idx'], lang_bs['word2idx'])
    lang_in_domain_bs = combine_lang(lang_in['word2idx'], lang_domain_token_bs['word2idx'])
    if share_inout:
        combined_lang = combine_lang(lang_in_domain_bs['word2idx'], lang_out['word2idx'])
        lang_in_domain_bs = combined_lang
        lang_out = combined_lang
    lang_dst = create_dst_dict(domain_slot_values)
    if sys_act:
        word2idx = {}
        idx2word = {}
        lang_act = {}
        for idx, act in enumerate(sorted(word_sys_act)):
            word2idx[act] = idx
            idx2word[idx] = act
        lang_act['word2idx'] = word2idx
        lang_act['idx2word'] = idx2word
    else:
        lang_act = None
    lang = {
      'in': lang_in,
      'out': lang_out,
      'bs': lang_bs, 
      'domain_token_bs': lang_domain_token_bs,
      'in+bs': lang_in_bs,
      'in+domain+bs': lang_in_domain_bs,
      'dst': lang_dst,
      'act': lang_act
    }
    if save_data: pkl.dump(lang, open('data{}/multi-woz/{}lang.pkl'.format(DATA_VERSION, save_prefix), 'wb'))
    return lang

def encode(word2idx, txt, no_unk=False):
    out = []
    for token in txt: 
        if token in word2idx:
            out.append(word2idx[token])
        else:
            if no_unk: pdb.set_trace()
            out.append(word2idx[UNK])
    return out 

def vocab_encode(dic, vocab):
    out = [None]*len(vocab)
    for k,v in vocab.items():
        out[v] = k
    return encode(dic, out)

def dst_encode(word2idx, in_txt, state, domain_slots, word2idx_in=None, booking_ptr=False, shared_dst=False):
    out = {}
    for domain, slot_values in state.items():
        out_state = []
        for idx, slot_value in enumerate(slot_values): 
            slot_name = domain_slots[domain][idx]
            if 'request' in slot_name or 'booked' in slot_name: 
                out_state.append(slot_value)
                continue 
            if word2idx_in is not None:
                if booking_ptr:
                    if shared_dst:
                        slot_vocab = word2idx_in
                    else:
                        if 'booking_' in slot_name:
                            slot_vocab = word2idx_in
                        else:
                            slot_vocab = word2idx[domain][slot_name]
                else:
                    slot_vocab = word2idx_in
            else:
                slot_vocab = word2idx[domain][slot_name]
            temp = encode(slot_vocab, slot_value)
            out_state.append(temp)
        out[domain] = out_state
    return out

def encode_dials(dials, lang, domain_slots, all_vocab, booking_ptr, shared_dst, sys_act):
    out_dial = {}
    for dial_key, dial in dials.items():
        dial['encoded_in_txt'] = encode(lang['in']['word2idx'], dial['in_txt'])
        dial['encoded_out_txt_in'] = encode(lang['out']['word2idx'], dial['out_txt_in'])
        dial['encoded_out_txt_out'] = encode(lang['out']['word2idx'], dial['out_txt_out'])
        if all_vocab:
            dial['encoded_state'] = dst_encode(lang['dst']['word2idx'], dial['in_txt'], dial['state'], domain_slots, lang['in+domain+bs']['word2idx'], booking_ptr, shared_dst)
        else:
            dial['encoded_state'] = dst_encode(lang['dst']['word2idx'], dial['in_txt'], dial['state'], domain_slots)
        dial['encoded_previous_state'] = encode(lang['bs']['word2idx'], dial['previous_state'])
        dial['encoded_previous_domain_token_state'] = encode(lang['domain_token_bs']['word2idx'], dial['previous_domain_token_state'])
        dial['encoded_in_txt_by_in+bs'] = encode(lang['in+bs']['word2idx'], dial['in_txt'])
        dial['encoded_previous_state_by_in+bs'] = encode(lang['in+bs']['word2idx'], dial['previous_state'])
        dial['encoded_in_txt_by_in+domain+bs'] = encode(lang['in+domain+bs']['word2idx'], dial['in_txt'])
        dial['encoded_previous_domain_token_state_by_in+domain+bs'] = encode(lang['in+domain+bs']['word2idx'], dial['previous_domain_token_state'])
        dial['encoded_current_domain_token_state_by_in+domain+bs'] = encode(lang['in+domain+bs']['word2idx'], dial['current_domain_token_state'])
        if sys_act:
            dial['encoded_act'] = encode(lang['act']['word2idx'], dial['act'], no_unk=True)
        out_dial[dial_key] = dial
    return out_dial 

def remove_domains(delex_data, excluded_domains):
    new_dials = {}
    for dial_name, dial in delex_data['delex_dialogues'].items():
        flag = False
        for domain in excluded_domains:
            if domain in dial['goal'].keys():
                flag = True
                break 
        if not flag:
            turn_flag = False
            for turn in dial['dialogue']:
                state_flag = False
                for domain in excluded_domains:
                    if domain in turn['state'].keys():
                        state_flag = True
                        break
                if state_flag:
                   turn_flag = True
                   break
            if not turn_flag:
                new_dials[dial_name] = dial 
    print(len(new_dials))
    delex_data['delex_dialogues'] = new_dials
    new_dsv_dict = {}
    for domain, dic in delex_data['dsv_dict'].items():
        if domain not in excluded_domains:
            new_dsv_dict[domain] = dic
    delex_data['dsv_dict'] = new_dsv_dict
    return delex_data

def encode_data(dials, lang, domain_slots, num_samples=-1, save_data=False, save_prefix='', all_vocab=False, booking_ptr=False, shared_dst=False, sys_act=False):
    out = {}
    out['train'] = encode_dials(dials['train'], lang, domain_slots, all_vocab, booking_ptr, shared_dst, sys_act)
    out['test'] = encode_dials(dials['test'], lang, domain_slots, all_vocab, booking_ptr, shared_dst, sys_act)
    out['val'] = encode_dials(dials['val'], lang, domain_slots, all_vocab, booking_ptr, shared_dst, sys_act)
    if save_data:
        if num_samples > 0:
            pkl.dump(out, open('data{}/multi-woz/{}encoded_data_small.pkl'.format(DATA_VERSION, save_prefix), 'wb'))
        else:
            pkl.dump(out, open('data{}/multi-woz/{}encoded_data.pkl'.format(DATA_VERSION, save_prefix), 'wb'))
    return out

DATA_VERSION='2.1'
TEST_LS = load_dialogues_ls('data{}/MULTIWOZ2/testListFile.json'.format(DATA_VERSION))
VAL_LS = load_dialogues_ls('data{}/MULTIWOZ2/valListFile.json'.format(DATA_VERSION))

if not os.path.exists('data{}/multi-woz'.format(DATA_VERSION)):
    os.mkdir('data{}/multi-woz'.format(DATA_VERSION))

def main():  
    ignore_booking = False
    save_data = True 
    num_samples = -1 #number of dialogues in each set  
    #num_samples = 1000
    #save_prefix = ''
    save_prefix = 'updated_'
    if os.path.exists('data{}/multi-woz/{}delex_data.pkl'.format(DATA_VERSION, save_prefix)):
        print("Found saved delex data")
        delex_data = pkl.load(open('data{}/multi-woz/{}delex_data.pkl'.format(DATA_VERSION, save_prefix), 'rb'))
    else:
        delex_data = delexicalize_data(ignore_booking, save_data, save_prefix, excluded_domains)
    #if len(excluded_domains) > 0:
    #    save_prefix += '_'.join(excluded_domains) + '_'
    #    delex_data = remove_domains(delex_data, excluded_domains)
    if os.path.exists('data{}/multi-woz/{}slots.pkl'.format(DATA_VERSION, save_prefix)):
        print("Found saved slots") 
        slots = pkl.load(open('data{}/multi-woz/{}slots.pkl'.format(DATA_VERSION, save_prefix), 'rb'))
    else:
        slots = get_all_slots(delex_data['delex_dialogues'], delex_data['dsv_dict'], ignore_booking=ignore_booking, save_data=save_data, save_prefix=save_prefix)
    
    all_vocab = True
    booking_ptr = True
    shared_dst = True
    sys_act = True
    share_inout = 0
    
    #save_prefix = 'updated_shareinout{}_infreq{}_outfreq{}_'.format(
    #    share_inout, IN_WORDS_FREQ_MIN, OUT_WORDS_FREQ_MIN)
    #if not os.path.exists('data{}/multi-woz/{}slots.pkl'.format(DATA_VERSION, save_prefix)):
    #    pkl.dump(slots, open('data{}/multi-woz/{}slots.pkl'.format(DATA_VERSION, save_prefix), 'wb'))
    
    dials, word_freqs_in, word_freqs_bs, word_freqs_domain_token_bs, word_freqs_out, word_sys_act = divide_data(
        delex_data['delex_dialogues'], slots['domain_slots'], 
        num_samples, ignore_booking=ignore_booking, 
        save_data=save_data, save_prefix=save_prefix, all_vocab=all_vocab, sys_act=sys_act)
    lang = build_dicts(word_freqs_in, word_freqs_bs, 
                       word_freqs_domain_token_bs, word_freqs_out, 
                       slots['domain_slot_values'], save_data=save_data, 
                       save_prefix=save_prefix, sys_act=sys_act, 
                       word_sys_act=word_sys_act, share_inout=share_inout)
    encoded_dials = encode_data(dials, lang, slots['domain_slots'], num_samples, save_data=save_data, save_prefix=save_prefix, all_vocab=all_vocab, booking_ptr=booking_ptr, shared_dst=shared_dst, sys_act=sys_act)

if __name__ == "__main__":
    main()
