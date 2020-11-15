import json 
import argparse 
import pdb
import pickle as pkl
import copy 
import random 
from tqdm import tqdm

from models.utils import *
from utils.nlp import *
from utils.db_pointer import *
from utils.data_handler import * 
import glob 

from configs.test_configs import *

REQUESTABLES = ['phone', 'address', 'postcode', 'reference', 'id', 'trainID']
DOMAINS = ['restaurant', 'hotel', 'attraction', 'taxi', 'train']

def parseGoal(d_goal):
    """Parses user goal into dictionary format."""
    goal = {}
    for domain in DOMAINS:
        goal[domain] = {}
        goal[domain] = {'informable': [], 'requestable': [], 'booking': []}
        if 'info' in d_goal[domain]:
            if domain == 'train':
                # we consider dialogues only where train had to be booked!
                if 'book' in d_goal[domain]:
                    goal[domain]['requestable'].append('reference')
                if 'reqt' in d_goal[domain]:
                    if 'trainID' in d_goal[domain]['reqt']:
                        goal[domain]['requestable'].append('trainID')
            else:
                if 'reqt' in d_goal[domain]:
                    for s in d_goal[domain]['reqt']:  # addtional requests:
                        if s in ['phone', 'address', 'postcode', 'reference', 'id']:
                            # ones that can be easily delexicalized
                            goal[domain]['requestable'].append(s)
                if 'book' in d_goal[domain]:
                    goal[domain]['requestable'].append("reference")
            goal[domain]["informable"] = d_goal[domain]['info']
            if 'book' in d_goal[domain]:
                goal[domain]["booking"] = d_goal[domain]['book']
    return goal

def get_gt_requestables(goal):
    temp = parseGoal(goal)
    real_requestables = {}
    for domain in goal.keys():
        if domain not in DOMAINS: continue 
        real_requestables[domain] = temp[domain]['requestable']
    return real_requestables

print('Loading training params from ' + args.out_dir + '/' + args.model + '.conf')
train_args = pkl.load(open(args.out_dir + '/' + args.model + '.conf', 'rb'))

output = "{}/{}".format(args.out_dir, args.output)
result = json.load(open(output, 'r'))        
delex_dir = 'data{}/multi-woz/updated_delex_data.pkl'.format(train_args.data_version)
data = pkl.load(open(delex_dir, 'rb'))

original_res = []
predicted_res = [] 

original_turn_res = {}
predicted_turn_res = {}

domain_original_res = {}
domain_predicted_res = {}

single_original_res = []
single_predicted_res = []

multi_original_res = []
multi_predicted_res = []

predicted_dials = {}

for dial_id, dial_result in result.items():
    dial_name, turn = dial_id.split('_')
    delex_dial = data['delex_dialogues'][dial_name]['dialogue']
    if dial_name not in predicted_dials:
        predicted_dials[dial_name] = [None] * len(delex_dial)
    res = [dial_result['response']]
    pred_res = [dial_result['predicted_response']['0']['txt']]
    active_domains = list(data['delex_dialogues'][dial_name]['goal'].keys())
    original_res.append(res)
    predicted_res.append(pred_res)
    if turn not in original_turn_res: original_turn_res[turn] = []
    if turn not in predicted_turn_res: predicted_turn_res[turn] = []
    original_turn_res[turn].append(res)
    predicted_turn_res[turn].append(pred_res)
    predicted_dials[dial_name][int(turn)-1] = pred_res[0]
    if len(active_domains) == 1:
        domain = active_domains[0]
        if domain not in domain_original_res: domain_original_res[domain] = []
        if domain not in domain_predicted_res: domain_predicted_res[domain] = []
        domain_original_res[domain].append(res)
        domain_predicted_res[domain].append(pred_res)
    if len(active_domains) > 1: 
        multi_original_res.append(res)
        multi_predicted_res.append(pred_res)
    else:
        single_original_res.append(res)
        single_predicted_res.append(pred_res)

out = {}
out['all_bleu'] = BLEUScorer().score(predicted_res, original_res)
out['single_bleu'] = BLEUScorer().score(single_predicted_res, single_original_res)
out['multi_bleu'] = BLEUScorer().score(multi_predicted_res, multi_original_res)
out['domain_bleu'] = {}
for domain, res_ls in domain_original_res.items():
     out['domain_bleu'][domain] = BLEUScorer().score(domain_predicted_res[domain], res_ls)
out['turn_bleu'] = {}
for turn_id, turn_res in original_turn_res.items():
      out['turn_bleu'][turn_id] = BLEUScorer().score(predicted_turn_res[turn_id], turn_res)
        
successes = 0
matches = 0 

oracle_matches = 0
oracle_successes = 0

domain_successes = {}
domain_matches = {}

domain_num_dials = {}

multi_successes = 0
multi_matches = 0

single_successes = 0
single_matches = 0

num_multi_dialogues = 0
num_single_dialogues = 0

for domain in DOMAINS:
    domain_successes[domain] = 0 
    domain_matches[domain] = 0 
    domain_num_dials[domain] = 0 

for dial_name, dial in tqdm(predicted_dials.items()):
    goal = data['delex_dialogues'][dial_name]['goal']
    provided_requestables = {}
    gt_requestables = get_gt_requestables(data['delex_dialogues'][dial_name]['original_goal'])
    goal_for_query, _, goal_for_request = get_goal_for_query_request(goal)
    temp = {}
    for k,v in goal_for_request.items():
        requestables = v
        if k in gt_requestables:
            requestables = list(set(v + gt_requestables[k]))
        temp[k] = requestables
    goal_for_request = temp 

    venue_offered = {}
    oracle_venue_offered = {}

    for domain in goal.keys():
        venue_offered[domain] = []
        oracle_venue_offered[domain] = []
        provided_requestables[domain] = []

    for t, sent_t in enumerate(dial):
        if '{}_{}'.format(dial_name, t+1) not in result: pdb.set_trace()  
        if train_args.setting in ['c2t']:
            predicted_bs = result['{}_{}'.format(dial_name, t+1)]['state']
        else:
            predicted_bs = result['{}_{}'.format(dial_name, t+1)]['predicted_state']
        predicted_bs = get_bs_for_query(predicted_bs)
        for domain in goal.keys():
            if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                if '{}_name'.format(domain) in sent_t or \
                  '{}_id'.format(domain) in sent_t or \
                  '{}_trainID'.format(domain) in sent_t or \
                  '{}_reference'.format(domain) in sent_t:
                    if train_args.setting in ['c2t']:
                        pred_venues, _ = query_result(domain, goal_for_query , True)
                    else:
                        pred_venues, _ = query_result(domain, predicted_bs , True)
                    gt_venues, _ = query_result(domain, goal_for_query, True)
                    if len(pred_venues)>0:
                        venue_offered[domain] = random.sample(pred_venues, 1) 
                    if len(gt_venues)>0:
                        oracle_venue_offered[domain] = random.sample(gt_venues, 1)
            provided_requestables[domain] += get_requestables(domain, sent_t)

    for domain in goal.keys():
        if 'name' in goal[domain]:
            venue_offered[domain] = 1
            continue 
        if domain == 'train':
            if not venue_offered[domain]:
                if 'request' in goal[domain] and 'trainID' not in goal[domain]: venue_offered[domain] = 1

    all_venues_gt = {}      
    match = 0
    oracle_match = 0
    for domain in goal.keys():
        if domain in ['restaurant', 'hotel', 'attraction', 'train']:
            if type(venue_offered[domain]) == int and  venue_offered[domain] == 1:
                match += 1
                oracle_match += 1 
            else:
                all_venues_gt[domain], _ = query_result(domain, goal_for_query, True)
                if len(all_venues_gt[domain]) == 0: pdb.set_trace()
                if len(venue_offered[domain]) > 0 and venue_offered[domain][0] in all_venues_gt[domain]:
                    match += 1      
                if len(oracle_venue_offered[domain]) > 0 and oracle_venue_offered[domain][0] in all_venues_gt[domain]:
                    oracle_match += 1
        else:
            match += 1
            oracle_match += 1

    if len(goal.keys()) == 1:
        domain = list(goal.keys())[0]
        domain_num_dials[domain] += 1
        num_single_dialogues += 1
    if len(goal.keys()) > 1:
        num_multi_dialogues += 1

    if match == len(goal.keys()): 
        matches += 1 
        if len(goal.keys()) == 1:
            domain = list(goal.keys())[0]
            domain_matches[domain] += 1
            single_matches += 1  
        if len(goal.keys()) > 1:
            multi_matches += 1 

        success = 0
        for domain in goal.keys():
            if len(goal_for_request[domain]) == 0:
                success += 1 
                continue

            domain_success = 0
            for request in set(provided_requestables[domain]):
                if request in goal_for_request[domain]:
                    domain_success += 1 
            if domain_success == len(goal_for_request[domain]):
                success += 1 

        if success == len(goal_for_request):
            successes += 1
            if len(goal.keys()) == 1:
                domain = list(goal.keys())[0]
                domain_successes[domain] += 1 
                single_successes += 1 
            if len(goal.keys()) > 1:
                multi_successes += 1 

    if oracle_match == len(goal.keys()): 
        oracle_matches += 1 

        success = 0
        for domain in goal.keys():
            if len(goal_for_request[domain]) == 0:
                success += 1 
                continue

            domain_success = 0
            for request in set(provided_requestables[domain]):
                if request in goal_for_request[domain]:
                    domain_success += 1 

            if domain_success == len(goal_for_request[domain]):
                success += 1 

        if success == len(goal_for_request):
            oracle_successes += 1

out['nb_dials'] = {}
out['nb_dials']['all'] = len(predicted_dials)
out['nb_dials']['domain'] = domain_num_dials
out['nb_dials']['multi'] = num_multi_dialogues
out['nb_dials']['single'] = num_single_dialogues
            
successes /= len(predicted_dials)
matches /= len(predicted_dials) 

oracle_successes /= len(predicted_dials)
oracle_matches /= len(predicted_dials) 

domain_successes = {k:v/domain_num_dials[k] for k,v in domain_successes.items()}
domain_matches = {k:v/domain_num_dials[k] for k,v in domain_matches.items()}

multi_successes /= num_multi_dialogues
multi_matches /= num_multi_dialogues

single_successes /= num_single_dialogues
single_matches /= num_single_dialogues

out['inform'] = matches
out['success'] = successes
out['oracle_inform'] = oracle_matches
out['oracle_success'] = oracle_successes
out['domain_inform'] = domain_matches
out['domain_success'] = domain_successes
out['single_inform'] = single_matches
out['single_success'] = single_successes
out['multi_inform'] = multi_matches
out['multi_success'] = multi_successes
print(out)

output_file = args.output.split('.')[0] + '_c2t_eval.json'
print('writing results to ' + args.out_dir + '/' + output_file)
json.dump(out, open(args.out_dir + '/' + output_file, 'w'), indent=4)
