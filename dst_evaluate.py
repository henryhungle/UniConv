#!/usr/bin/env python

import json
import pdb 
import pickle as pkl 
import glob 

from configs.test_configs import *

SLOTS = ['hotel-pricerange', 'hotel-type', 'hotel-parking', 'hotel-bookstay', 'hotel-bookday', 'hotel-bookpeople', 'hotel-area', 'hotel-stars', 'hotel-internet', 'train-destination', 'train-day', 'train-departure', 'train-arriveby', 'train-bookpeople', 'train-leaveat', 'attraction-area', 'restaurant-food', 'restaurant-pricerange', 'restaurant-area', 'attraction-name', 'restaurant-name', 'attraction-type', 'hospital-department', 'hotel-name', 'taxi-leaveat', 'taxi-destination', 'taxi-departure', 'restaurant-booktime', 'restaurant-bookday', 'restaurant-bookpeople', 'taxi-arriveby', 'bus-departure', 'bus-destination', 'bus-leaveat', 'bus-day']
DOMAINS = ['hotel', 'train', 'attraction', 'restaurant', 'taxi']

class Evaluator:
    "Optim wrapper that implements rate."
    def __init__(self):
        self.slots = SLOTS

    def evaluate_metrics(self, all_prediction, turn=-1):
        total, turn_acc, joint_acc, F1_pred, F1_count = 0, 0, 0, 0, 0
        for d, v in all_prediction.items():
            for t in range(len(v)):
                if turn>-1 and t!=turn: continue 
                cv = v[t]
                if set(cv["turn_belief"]) == set(cv['predicted_belief']):
                    joint_acc += 1
                total += 1

                # Compute prediction slot accuracy
                temp_acc = self.compute_acc(set(cv["turn_belief"]), set(cv['predicted_belief']), self.slots)
                turn_acc += temp_acc

                # Compute prediction joint F1 score
                temp_f1, temp_r, temp_p, count = self.compute_prf(set(cv["turn_belief"]), set(cv['predicted_belief']))
                F1_pred += temp_f1
                F1_count += count

        joint_acc_score = joint_acc / float(total) if total!=0 else 0
        turn_acc_score = turn_acc / float(total) if total!=0 else 0
        F1_score = F1_pred / float(F1_count) if F1_count!=0 else 0
        return joint_acc_score, F1_score, turn_acc_score

    def compute_acc(self, gold, pred, slot_temp):
        miss_gold = 0 
        miss_slot = []
        for g in gold:
            if g not in pred:
                miss_gold += 1
                miss_slot.append(g.rsplit("-", 1)[0])
        wrong_pred = 0 
        for p in pred:
            if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
                wrong_pred += 1
        ACC_TOTAL = len(slot_temp)
        ACC = len(slot_temp) - miss_gold - wrong_pred
        ACC = ACC / float(ACC_TOTAL)
        return ACC 

    def compute_prf(self, gold, pred):
        TP, FP, FN = 0, 0, 0
        if len(gold)!= 0:
            count = 1 
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
            for p in pred:
                if p not in gold:
                    FP += 1
            precision = TP / float(TP+FP) if (TP+FP)!=0 else 0
            recall = TP / float(TP+FN) if (TP+FN)!=0 else 0
            F1 = 2 * precision * recall / float(precision + recall) if (precision+recall)!=0 else 0
        else:
            if len(pred)==0:
                precision, recall, F1, count = 1, 1, 1, 1
            else:
                precision, recall, F1, count = 0, 0, 0, 1
        return F1, recall, precision, count

def process_state(state):
    out = []
    for d, sv in state.items():
        for s, v in sv.items():
            if 'request' not in s and s!='booking_booked':
                s = s.replace('inform_', '')
                s = s.replace('leaveAt', 'leaveat')
                s = s.replace('arriveBy', 'arriveby')
                if 'booking_' in s:
                    s = s.replace('booking_', '')
                    s = 'book' + s
                    #continue 
                if '-'.join([d,s]) not in SLOTS: pdb.set_trace()
                state = '-'.join([d,s,v])
                out.append(state)
    return out

def evaluate(predictions):
    evaluator = Evaluator()
    joint_acc_score, F1_score, turn_acc_score = evaluator.evaluate_metrics(predictions)
    print("joint acc {} f1 {} slot acc {}".format(joint_acc_score, F1_score, turn_acc_score))
    out = {}
    out['joint_acc'] = joint_acc_score
    out['acc'] = turn_acc_score
    out['f1'] = F1_score
    return out 

print('Loading training params from ' + args.out_dir + '/' + args.model + '.conf')
train_args = pkl.load(open(args.out_dir + '/' + args.model + '.conf', 'rb'))
prefix = train_args.prefix
slots_dir = 'data{}/multi-woz/{}slots.pkl'.format(train_args.data_version, prefix)
print('Extracting slots from ' + slots_dir)
slots_dic = pkl.load(open(slots_dir, 'rb')) 
output = "{}/{}".format(args.out_dir, args.output)
predictions = json.load(open(output, 'r'))

new_predictions = {}
for k,v in predictions.items():
    dial_id, turn_id = k.split('_')
    turn_id = int(turn_id) - 1
    if dial_id not in new_predictions: new_predictions[dial_id] = {}
    if turn_id not in new_predictions[dial_id]: new_predictions[dial_id][turn_id] = {}
    turn_belief = process_state(predictions[k]['state'])
    predicted_belief = process_state(predictions[k]['predicted_state'])
    new_predictions[dial_id][turn_id]['turn_belief'] = turn_belief
    new_predictions[dial_id][turn_id]['predicted_belief'] = predicted_belief

single_predictions = {}
multi_predictions = {}
for k,v in new_predictions.items():
    domains = set()
    for k2,v2 in v.items():
        bs = v2['turn_belief']
        turn_domains = [i.split('-')[0] for i in bs]
        domains.update(turn_domains)
    if len(domains)==1:
        single_predictions[k] = v
    if len(domains)>1:
        multi_predictions[k] = v

out = {}
out['all'] = evaluate(new_predictions)
out['multi'] = evaluate(multi_predictions)
out['single'] = evaluate(single_predictions)

for domain in DOMAINS:
    print("Domain: {}".format(domain))
    domain_predictions = {}
    count = 0
    for dial_id, dial in new_predictions.items():
        found_domain = False
        for turn_id, turn in dial.items():
            gt_bs = turn['turn_belief']
            pred_bs = turn['predicted_belief']
            gt_bs = [b for b in gt_bs if b.split('-')[0] == domain]
            pred_bs = [b for b in pred_bs if b.split('-')[0] == domain]
            if len(gt_bs) == 0: continue 
            if len(gt_bs) > 0: found_domain = True
            if dial_id not in domain_predictions: domain_predictions[dial_id] = {}
            turn_id = len(domain_predictions[dial_id])
            domain_predictions[dial_id][turn_id] = {}
            domain_predictions[dial_id][turn_id]['turn_belief'] = gt_bs
            domain_predictions[dial_id][turn_id]['predicted_belief'] = pred_bs
        if found_domain: count += 1
    out[domain] = evaluate(domain_predictions)
    
output_file = args.output.split('.')[0] + '_dst_eval.json'
print('writing results to ' + args.out_dir + '/' + output_file)
json.dump(out, open(args.out_dir + '/' + output_file, 'w'), indent=4)
