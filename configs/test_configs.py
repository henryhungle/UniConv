import argparse
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--out-dir', default='', type=str, 
                    help='directory of experiment') 
parser.add_argument('--model', '-m', default='', type=str,
                    help='model version to be used')
parser.add_argument('--dst-max-len', default=10, type=int, 
                    help='maximum length of output dst value')
parser.add_argument('--res-max-len', default=20, type=int,
                    help='Max-length of output sequence')
parser.add_argument('--beam', default=5, type=int, help='Beam width')
parser.add_argument('--penalty', default=2.0, type=float, help='Insertion penalty')
parser.add_argument('--nbest', default=1, type=int, help='Number of n-best hypotheses')
parser.add_argument('--res-min-len', default=1, type=int, help='output dialogue response minimum length')
parser.add_argument('--output', '-o', default='', type=str, help='Output generated responses in a json file')
parser.add_argument('--gt-db-pointer', default=0, type=int, help='using ground-truth database pointer, in context-to-text setting')
parser.add_argument('--gt-previous-bs', default=0, type=int, help='using ground-truth prior dialogue state, in context-to-text setting')
parser.add_argument('--small-data', default=0, type=int, help='use a small version of dataset for testing purpose')
parser.add_argument('--verbose', default=0, type=int, help='print out option if verbose is True')
parser.add_argument('--num-workers', default=0, type=int, help='number of worker to process data')
parser.add_argument('--tep', default='best', type=str, help='a specific epoch checkpoint for testing')

args = parser.parse_args()
for arg in vars(args):
    print("{}={}".format(arg, getattr(args, arg)))
