import argparse
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
# train, dev and test data
parser.add_argument('--data-version', default='', type=str, help='data version')
parser.add_argument('--small-data', default=0, type=int, help='using small data for testing')
#parser.add_argument('--cut-a-p', default=0, type=float, help='randomly cut responses to simulate bs') 
#parser.add_argument('--merge-source', default=0, type=int, help='merge all source sequences into one') 
#parser.add_argument('--pretrained-word-emb', default=None, type=str,help='')
#parser.add_argument('--pretrained-word-emb-cache', default=None, type=str, help='')
#parser.add_argument('--fixed-word-emb', default=0, type=int, help='')
#parser.add_argument('--pretrained-weights', default=None, type=str,help='')
parser.add_argument('--out-dir', default=None, type=str, help='output path')  
parser.add_argument('--model', default=None, type=str, help='output model name')
#parser.add_argument('--active-domain-slots', default=0, type=int, help='add slot for active domains') 
parser.add_argument('--detach-dial-his', default=0, type=int, help='seperate dialogue history from input text')
parser.add_argument('--incl-sys-utt', default=0, type=int, help='')
parser.add_argument('--only-system-utt', default=0, type=int, help='Only applicable for max_dial_his_len=1')
parser.add_argument('--max-dial-his-len', default=-1, type=int, help='Maximum dialogue turns in dialogue history') 
parser.add_argument('--add-prev-dial-state', default=0, type=int, help='add previous dialogue state into input') 
#parser.add_argument('--domain-token-state', default=0, type=int, help='') 
#parser.add_argument('--ignore-booking', default=0, type=int, help='')
#parser.add_argument('--normalized-db-pointer', default=0, type=int, help='')
parser.add_argument('--prefix', default='', type=str, help='')
parser.add_argument('--num-workers', default=0, type=int, help='')
parser.add_argument('--sys-act', default=0, type=int, help='')
#parser.add_argument('--c2t', default=0, type=int, help='') 
#parser.add_argument('--remove-domain', default='', type=str, help='')

# Model 
parser.add_argument('--nb-blocks-res-dec', default=4, type=int, help='number of transformer blocks')
parser.add_argument('--nb-blocks-slot-dst', default=4, type=int, help='')
parser.add_argument('--nb-blocks-domain-dst', default=4, type=int, help='')
parser.add_argument('--nb-blocks-domain-slot-dst', default=1, type=int, help='') 
#parser.add_argument('--domain-nb-blocks', default=1, type=int, help='number of transformer blocks for domain dst') 
parser.add_argument('--d-model', default=512, type=int, help='dimension of model tensors') 
parser.add_argument('--d-ff', default=2048, type=int, help='dimension of feed forward') 
parser.add_argument('--att-h', default=8, type=int, help='number of attention heads') 
#parser.add_argument('--gated-conn', default=0, type=int, help='')
#parser.add_argument('--dst-gen-d-hidden', default=-1, type=int, help='') 
parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate')  
#parser.add_argument('--dst-gen-dropout', default=0, type=float, help='')
#parser.add_argument('--mode', default=0, type=int, help='0: query only, 1:  query and context')
#parser.add_argument('--share-in-bs-vocab', default=0, type=int, help='share embedding between input text and bs')
#parser.add_argument('--share-slot-bs-vocab', default=0, type=int, help='')
#parser.add_argument('--binary-output-dst', default=0, type=int, help='')
parser.add_argument('--setting', default='dst', type=str, help='')
#parser.add_argument('--add-in-txt', default=0, type=int, help='')
#parser.add_argument('--add-in-his', default=0, type=int, help='')
#parser.add_argument('--merge-dst', default=0, type=int, help='')
parser.add_argument('--domain-flow', default=0, type=int, help='') 
parser.add_argument('--domain-slot-dst', default=0, type=int, help='') 
#parser.add_argument('--combine-domain-slot-state', default=0, type=int, help='')
#parser.add_argument('--all-vocab', default=0, type=int, help='')
#parser.add_argument('--booking-ptr', default=0, type=int, help='')
parser.add_argument('--share-dst-gen', default=0, type=int, help='')
parser.add_argument('--ds-fusion', default='product', type=str, help='')
parser.add_argument('--norm-ds', default=0, type=int, help='')
parser.add_argument('--dst-classify', default=0, type=int, help='') 
parser.add_argument('--pretrained-dst', default=None, type=str, help='')
parser.add_argument('--share-inout', default=0, type=int, help='')
parser.add_argument('--fixed-dst', default=0, type=int, help='')
parser.add_argument('--literal-bs', default=0, type=int, help='')

# Training 
parser.add_argument('--num-epochs', '-e', default=15, type=int,help='Number of epochs')
parser.add_argument('--rand-seed', '-s', default=1, type=int, help="seed for generating random numbers")
parser.add_argument('--batch-size', '-b', default=32, type=int,help='Batch size in training')
#parser.add_argument('--max-length', default=20, type=int,help='Maximum length for controling batch size')
#parser.add_argument('--max-history-length', default=-1, type=int, help='Maximum past history length to consider')
parser.add_argument('--report-interval', default=100, type=int,help='report interval to log training results')
parser.add_argument('--warmup-steps', default=4000, type=int,help='warm up steps for optimizer') 
# others
parser.add_argument('--verbose', '-v', default=0, type=int,help='verbose level')
args = parser.parse_args()
'''
args.merge_source = bool(args.merge_source)
args.fixed_word_emb = bool(args.fixed_word_emb)
args.small_data = bool(args.small_data)
args.active_domain_slots = bool(args.active_domain_slots)
args.gated_conn = bool(args.gated_conn)
args.detach_dial_his = bool(args.detach_dial_his)
args.add_prev_dial_state = bool(args.add_prev_dial_state)
args.share_in_bs_vocab = bool(args.share_in_bs_vocab)
args.only_system_utt = bool(args.only_system_utt)
args.binary_output_dst = bool(args.binary_output_dst)
args.domain_token_state = bool(args.domain_token_state)
args.share_slot_bs_vocab = bool(args.share_slot_bs_vocab)
args.add_in_txt = bool(args.add_in_txt)
args.merge_dst = bool(args.merge_dst)
args.ignore_booking = bool(args.ignore_booking)
args.normalized_db_pointer = bool(args.normalized_db_pointer)
args.add_in_his = bool(args.add_in_his)
args.domain_flow = bool(args.domain_flow) 
args.domain_slot_dst = bool(args.domain_slot_dst) 
args.combine_domain_slot_state = bool(args.combine_domain_slot_state)
'''

# Presetting
random.seed(args.rand_seed)
np.random.seed(args.rand_seed)
for arg in vars(args):
    print("{}={}".format(arg, getattr(args, arg)))