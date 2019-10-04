import embeddings
# from cupy_utils import *

import argparse
import numpy as np
import re
import sys
import time

def dropout(m, p):
    if p <= 0.0:
        return m
    else:
        xp = get_array_module(m)
        mask = xp.random.rand(*m.shape) >= p
        return m*mask


def topk_mean(m, k, inplace=False):  # TODO Assuming that axis is 1
    xp = get_array_module(m)
    n = m.shape[0]
    ans = xp.zeros(n, dtype=m.dtype)
    if k <= 0:
        return ans
    if not inplace:
        m = xp.array(m)
    ind0 = xp.arange(n)
    ind1 = xp.empty(n, dtype=int)
    minimum = m.min()
    for i in range(k):
        m.argmax(axis=1, out=ind1)
        ans += m[ind0, ind1]
        m[ind0, ind1] = minimum
    return ans / k

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('src_input', help='the input source embeddings')
	parser.add_argument('trg_input', help='the input target embeddings')
	parser.add_argument('src_output', help='the output source embeddings')
	parser.add_argument('trg_output', help='the output target embeddings')
	parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
	parser.add_argument('--cuda', action='store_true', help='use cuda(requires cupy)')
	args = parser.parse_args()
	if args.cuda:
		print('there is cuda')
	print(args.src_input)
	print(args.trg_input)
	print(args.src_output)
	print(args.trg_output)
	print('complete')
	
	# Read input embeddings
    srcfile = open(args.src_input, encoding=args.encoding, errors='surrogateescape')
    trgfile = open(args.trg_input, encoding=args.encoding, errors='surrogateescape')
    src_words, x = embeddings.read(srcfile, dtype=dtype)
    trg_words, z = embeddings.read(trgfile, dtype=dtype)
    print('complete')

if __name__ == '__main__':
	main()