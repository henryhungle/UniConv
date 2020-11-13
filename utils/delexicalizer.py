import re

import simplejson as json
from collections import defaultdict
from utils.nlp import normalize
import pdb 
import copy
digitpat = re.compile('\d+')
timepat = re.compile("\d{1,2}[:]\d{1,2}")
pricepat2 = re.compile("\d{1,3}[.]\d{1,2}")
DOMAINS = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']
DB_TAGS = {
  'restaurant': ['address', 'area', 'food', 'id', 'introduction', 'name', 'phone', 'postcode', 'pricerange', 'type', 'signature'],
  'hotel': ['address', 'area', 'internet', 'parking', 'name', 'phone', 'postcode', 'pricerange', 'type'],
  'attraction': ['address', 'area', 'name', 'phone', 'postcode', 'type', 'entrancefee', 'pricerange'],
  'train': ['arriveBy', 'day', 'departure', 'destination', 'duration', 'leaveAt', 'price', 'trainID'],
  'taxi': ['taxi_colors', 'taxi_types'],
  'hospital': ['department', 'address', 'postcode', 'phone', 'name'],
  'police': ['name', 'address', 'phone', 'postcode']
}
BOOKING_SLOTS = ['booking_booked', 'booking_day', 'booking_people', 'booking_stay', 'booking_time']

def augment_db_price(prices):
    new_prices = set()
    for price in prices:
        tokens = price.split()
        if len(tokens) ==2 and tokens[1] == 'pounds':
            str_price = str(float(tokens[0]))
            new_prices.add(' '.join([str_price,tokens[1]]))
            new_prices.add(' '.join([str_price,'gbp']))
    return prices.union(new_prices)

def prepare_slot_values_independent():
    domains = DOMAINS
    requestables = ['phone', 'address', 'postcode', 'reference', 'id']
    dic = []
    dsv_dict = {}
    for domain in domains:
        dsv_dict[domain] = defaultdict(set) 
    address_alternatives={
        'road': 'rd',
        'rd': 'road',
        'street': 'st',
        'st': 'street'
    }
    name_alternatives ={
        'b & b': 'bed and breakfast',
        'bed and breakfast': 'b & b',
        'hotel': '',
        'restaurant': ''
    }
    # additional attributes for hospital kb and police kb (provided by Budzianowski's published codes)
    additional_kb = {}
    additional_kb['hospital'] =[
      ('address', 'Hills Rd'),
      ('address', 'Hills Road'),
      ('postcode', 'CB20QQ'),
      ('phone', '01223245151'),
      ('phone', '1223245151'),
      ('phone', '0122324515'),
      ('name', 'Addenbrookes Hospital')
    ]
    additional_kb['police'] =[
      ('address', 'Parkside'),
      ('postcode', 'CB11JG'),
      ('phone', '01223358966'),
      ('phone', '1223358966'),
      ('name', 'Parkside Police Station')
    ]

    # read databases
    for domain in domains:
        if True:
            fin = open('data2.0/db/' + domain + '_db.json')
            if domain == 'taxi': continue 
            db_json = json.load(fin)
            fin.close()
            if domain == 'taxi': 
                for key,val in db_json.items():
                    for v in val:
                        dic.append((v, '[' + key + ']'))
                        dsv_dict[domain][key].add(v)
                continue 
            for ent in db_json: 
                for key, val in ent.items():
                    if  type(val) == list: #(longituide,latitude)
                        assert key == 'location'
                        continue 
                    if type(val) == dict: #{single:.., double:...,....}
                        assert key == 'price'
                        for p_k, p_v in val.items():
                            dic.append((p_v, '[' + domain + '_' + key + ']'))
                            dsv_dict[domain][key].add(p_v)
                        continue 
                    if type(val) == str:
                        val = val.lower()
                    elif type(val) == int: #hospital id 
                        val = str(val)  
                    if val == '?':
                        continue 
                    #for slot in delex_slot:
                    if  True:
                        if key == 'phone':
                            dic.append((val, '[' + domain + '_' + key + ']'))
                            dsv_dict[domain][key].add(val)
                        else:
                            dic.append((normalize(val), '[' + domain + '_' + key + ']'))
                            dsv_dict[domain][key].add(normalize(val))
                        if key == 'address':
                            for add1, add2 in address_alternatives.items():
                                if add1 in val: 
                                    val = val.replace(add1, add2)
                                    dic.append((normalize(val), '[' + domain + '_' + key + ']'))
                                    dsv_dict[domain][key].add(normalize(val))
                                    break 
                        elif key == 'name':
                            dic.append((normalize(val), '[' + domain + '_' + key + ']'))
                            dsv_dict[domain][key].add(normalize(val))
                            for name1, name2 in name_alternatives.items():
                                if name1 in val: 
                                    val = val.replace(name1, name2)
                                    dic.append((normalize(val), '[' + domain + '_' + 'name' + ']'))
                                    dsv_dict[domain][key].add(normalize(val))
                                    break 

        if domain in additional_kb.keys():
            for item in additional_kb[domain]: 
                if item[0] == 'phone': 
                    dic.append((item[1], '[' + domain + '_' + item[0] + ']'))
                    dsv_dict[domain][item[0]].add(item[1])
                else:
                    dic.append((normalize(item[1]), '[' + domain + '_' + item[0] + ']'))
                    dsv_dict[domain][item[0]].add(normalize(item[1]))

    dsv_dict['train']['price'] = augment_db_price(dsv_dict['train']['price'])
    dsv_dict['attraction']['entrance fee'] = augment_db_price(dsv_dict['attraction']['entrance fee'])
    #dsv_dict['hotel']['wifi'] = copy.deepcopy(dsv_dict['hotel']['internet'])
    dsv_dict['attraction']['entrancefee'] = copy.deepcopy(dsv_dict['attraction']['entrance fee'])

    return dic, dsv_dict 

def replace_slots(utt, slot, value, tag):
    if slot == 'wifi':
        tag = tag.replace('wifi', 'internet')
    placeholder1 =  ' ' + ' '.join(len(value.split()) * [tag]) + ' '
    placeholder2 =  ' ' + tag + ' '
    if value in ['yes', 'no']:
        placeholder3 = ' [' + '_'.join(slot.split()) + '] '
        out1 = (' ' + utt[0] + ' ').replace(' ' + slot + ' ', placeholder1)
        out2 = (' ' + utt[1] + ' ').replace(' ' + slot + ' ', placeholder2)
        out3 = (' ' + utt[2] + ' ').replace(' ' + slot + ' ', placeholder3)
    else:
        placeholder3 = ' [' + '_'.join(value.split()) + '] '
        out1 = (' ' + utt[0] + ' ').replace(' ' + value + ' ', placeholder1)
        out2 = (' ' + utt[1] + ' ').replace(' ' + value + ' ', placeholder2)
        out3 = (' ' + utt[2] + ' ').replace(' ' + value + ' ', placeholder3)
    return out1[1:-1], out2[1:-1], out3[1:-1]

def tag_slots(utt, dic, prefix='', request_only=False, is_system_turn=False, ref_only=False):
    for domain, domain_dic in dic.items():
        for s,v in domain_dic.items():
            if 'fail' in s: continue 
            if s == 'booked':
                if request_only: continue 
                for booking in v:
                    for b_k,b_v in booking.items():
                        tag = '_'.join([domain, prefix, b_k])
                        if b_k == 'reference': 
                            for ref_form in ['','#','ref#']:
                                utt = replace_slots(utt, b_k, ref_form+b_v, tag) 
                            continue 
                        if ref_only: continue 
                        utt = replace_slots(utt, b_k, b_v, tag)
            elif s == 'request' and not is_system_turn and not ref_only:
                for r_s in v:
                    tag = '_'.join([domain, 'request', r_s])
                    utt = replace_slots(utt, r_s, r_s, tag)
            else:
                if request_only: continue
                if ref_only: continue 
                tag = '_'.join([domain, prefix, s])
                utt = replace_slots(utt, s, v, tag)
                if s == 'internet':
                    utt = replace_slots(utt, 'wifi', v, tag)
    return utt 

def get_active_domains(utt, state, booking_state):
    active_domains = set()
    for domain in DOMAINS:
        if domain + '_' in utt:
            active_domains.add(domain)
            continue 
        #if (domain in goal and len(goal[domain])>0) or \
        #    (domain in booking_goal and len(booking_goal[domain])>0) or \
        if (domain in state and len(state[domain])>0) or \
            (domain in booking_state and len(booking_state[domain])>0):
            active_domains.add(domain)
            continue
    active_domains.add('police')
    active_domains.add('hospital')
    return active_domains

def tag_slots_by_db(utt, dsv_dict, goal, booking_goal, state, booking_state, act):
    if act is None: return utt
    active_domains = get_active_domains(utt[0], state, booking_state)
    act_attributes = set()
    for domain, domain_act in act.items():
        if domain in active_domains:
            for act, sv in domain_act.items():
                if act != 'inform': continue 
                for s,v in sv.items():
                    act_attributes.add(s)
    for domain in active_domains:
        domain_sv = dsv_dict[domain]
        for s,v_ls in domain_sv.items():
            if s not in act_attributes and s not in DB_TAGS[domain]: continue 
            tag = '_'.join([domain, 'value', s])
            for v in v_ls:
                utt = replace_slots(utt, s, v, tag)
    return utt 

def find_tagged_slots(utt, delex_utt):
    utt_tokens = utt.split()
    delex_utt_tokens = delex_utt.split()
    slots = set()
    for idx, token in enumerate(utt_tokens):
        if token != delex_utt_tokens[idx]:
            slots.add(delex_utt_tokens[idx])
    return slots 
    
def check_inform_slots(tagged_slots, state, prefix=''):
    missed_slots = []
    additional_slots = []
    inform_slots = dict()
    for domain, sv in state.items():
        for s,v in sv.items():
            tag = '_'.join([domain,prefix, s]) 
            inform_slots[tag] = v
            if tag not in tagged_slots:
                missed_slots.append(tag)
    for slot in tagged_slots:
        if 'booking' not in slot and 'value' not in slot and 'request' not in slot and slot not in inform_slots:
            additional_slots.append(slot)
    #if len(missed_slots)>0: print("Missed tagged inform slots: {}".format(missed_slots))
    #if len(additional_slots)>0: print("New tagged inform slots: {}".format(additional_slots))
    return inform_slots, missed_slots, additional_slots

def check_request_slots(utt, tagged_slots, goal, is_last_turn):
    request_slots = set()
    missed_slots = []
    for domain, domain_goal in goal.items():
        for k,v in domain_goal.items():
            if k != 'request': continue 
            for r_s in v:
                tag = '_'.join([domain,'request',r_s])
                value_tag = '_'.join([domain,'value',r_s])
                if r_s in utt:
                    request_slots.add(tag)
                if is_last_turn and tag not in tagged_slots and value_tag not in tagged_slots:
                    missed_slots.append(tag)
    additional_slots = []
    for slot in tagged_slots:
        if 'booking' not in slot and 'value' not in slot and 'inform' not in slot and slot not in request_slots:
            additional_slots.append(slot)
    #if len(missed_slots)>0: print("Missed tagged request slots: {}".format(missed_slots))
    #if len(additional_slots)>0: print("New tagged request slots: {}".format(additional_slots))
    return request_slots, missed_slots, additional_slots

def clean_sys_delex_utt(utt, delex_utt, ignore_booking):
    original_tokens = utt.split()
    delex_tokens = delex_utt.split()
    new_delex_tokens = []
    new_sketch_tokens = []
    sketch_token = []
    assert len(original_tokens) == len(delex_tokens)
    for idx, token in enumerate(delex_tokens):
        if '_request_' in token: pdb.set_trace() 
        if '_inform_' in token or '_value_' in token or '_booking_' in token:
            if ignore_booking:
                is_booking_slot = False
                for booking_slot in BOOKING_SLOTS:
                    if booking_slot in token:
                        is_booking_slot = True
                        break
                if is_booking_slot:
                    new_delex_tokens.append(original_tokens[idx])
                    continue         
            sketch_token.append(original_tokens[idx])
            if idx == len(delex_tokens)-1 or token != delex_tokens[idx+1]:
                sketch_token = '[' + '_'.join(sketch_token) + ']' 
                new_sketch_tokens.append(sketch_token)
                sketch_token = []
            if idx > 0 and token == delex_tokens[idx-1]: continue 
            tokenized_tokens = token.split('_')
            new_token = '_'.join([tokenized_tokens[0]] + tokenized_tokens[2:])
            new_delex_tokens.append(new_token)
        else:
            new_delex_tokens.append(token)
            new_sketch_tokens.append(original_tokens[idx])
    if len(new_delex_tokens) != len(new_sketch_tokens): pdb.set_trace()
    out = ' '.join(new_delex_tokens)
    sketch_out = ' '.join(new_sketch_tokens)
    return out, sketch_out

def fix_phone_req_slots(out_sketch_utt, goal):
    for domain, domain_slots in goal.items():
        if domain != 'taxi': continue 
        for slot, values in domain_slots.items():
            if slot != 'request': continue
            if 'phone' in values and 'number' in out_sketch_utt:
                #print(out_sketch_utt)
                results = digitpat.findall(out_sketch_utt)
                for result in results:
                    out_sketch_utt = out_sketch_utt.replace('{}'.format(result), '{}_phone'.format(domain))
                #print(out_sketch_utt)
                #pdb.set_trace()
    return out_sketch_utt

def delexicalize(utt, dsv_dict, goal, booking_goal, state, booking_state, act, is_last_turn, is_system_turn, ignore_booking, for_glmp=False):
    utt = normalize(utt)
    utt = ' '.join(utt.split())
    delex_utt = [copy.deepcopy(utt)]*3
    delex_utt1 = tag_slots(delex_utt, state, 'inform', False, is_system_turn)
    delex_utt2 = tag_slots(delex_utt1, booking_state, 'booking', False, is_system_turn,  True)
    delex_utt3 = tag_slots(delex_utt2, goal, 'inform', True, is_system_turn)
    delex_utt4 = tag_slots(delex_utt3, booking_goal, 'booking', False, is_system_turn, True)
    delex_utt5 = tag_slots_by_db(delex_utt4, dsv_dict, goal, booking_goal, state, booking_state, act)
    tagged_slots = find_tagged_slots(utt, delex_utt4[0])
    inform_slots, _, _ = check_inform_slots(tagged_slots, state, 'inform') 
    booking_slots, _, _ = check_inform_slots(tagged_slots, booking_state, 'booking')
    request_slots, _, _ = check_request_slots(utt, tagged_slots, goal, is_last_turn)
    if is_system_turn: 
        out_sketch_utt, out_delex_utt = clean_sys_delex_utt(utt, delex_utt5[0], ignore_booking)
        out_sketch_utt = fix_phone_req_slots(out_sketch_utt, goal)
    else:
        out_delex_utt = delex_utt5[2]
        out_sketch_utt = delex_utt5[1] 
    if for_glmp:
        if len(out_sketch_utt.split()) != len(out_delex_utt.split()): pdb.set_trace()
        return utt, out_delex_utt, out_sketch_utt 
    return utt, out_sketch_utt, inform_slots, request_slots, booking_slots

