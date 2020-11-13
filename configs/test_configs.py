import argparse
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--out-dir', default='', type=str, help='') 
parser.add_argument('--model', '-m', default='', type=str,
                    help='Attention model to be output')
parser.add_argument('--dst-max-len', default=10, type=int, help='')
parser.add_argument('--res-max-len', default=20, type=int,
                    help='Max-length of output sequence')
parser.add_argument('--beam', default=5, type=int, help='Beam width')
parser.add_argument('--penalty', default=2.0, type=float, help='Insertion penalty')
parser.add_argument('--nbest', default=1, type=int, help='Number of n-best hypotheses')
parser.add_argument('--res-min-len', default=1, type=int, help='')
parser.add_argument('--output', '-o', default='', type=str, help='Output generated responses in a json file')
parser.add_argument('--gt-db-pointer', default=0, type=int, help='')
parser.add_argument('--gt-previous-bs', default=0, type=int, help='')
parser.add_argument('--small-data', default=0, type=int, help='')
parser.add_argument('--verbose', default=0, type=int, help='')
parser.add_argument('--num-workers', default=0, type=int, help='')
parser.add_argument('--tep', default='best', type=str, help='')
#parser.add_argument('--te2e', default=-1, type=int, help='')

args = parser.parse_args()
for arg in vars(args):
    print("{}={}".format(arg, getattr(args, arg)))