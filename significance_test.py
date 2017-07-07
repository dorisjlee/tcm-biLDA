### Author: Edward Huang

import argparse
import numpy as np
from scipy.stats import ttest_rel
import sys
import time

### This script evaluates the NDCG's of retrieval systems by performing the
### paired t-test on them.

def read_ndcg_dct(method):
    '''
    Reads the NDCG results dictionary and gets the results for each k.
    '''
    ndcg_dct = {}
    f = open('./results/%s.txt' % (method), 'r')
    for line in f:
        ndcg, k = line.split()
        if k not in ndcg_dct:
            ndcg_dct[k] = []
        ndcg_dct[k] += [float(ndcg)]
    f.close()
    return ndcg_dct

def parse_args():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', choices=['no', 'synonym', 'lda',
        'bilda', 'dca', 'med2vec', 'pmi', 'cooccurrence', 'prosnet', 'bilda-dca',
        'bilda-synonym'], required=True, help='Type of query expansion to test.')
    parser.add_argument('-r', '--rank_metric', choices=['ndcg', 'precision',
        'recall', 'f1', 'map'], required=True, help='Type of rank metric to use.')
    parser.add_argument('-t', '--term_type', choices=['herbs', 'symptoms',
        'mixed'], help='Type of query expansion terms.')
    args = parser.parse_args()
    if args.method in ['no', 'synonym']:
        assert args.term_type == None
        fname = args.method
    else:
        fname = '%s_%s' % (args.method, args.term_type)
    return fname

def main():
    fname = parse_args()

    # TODO
    baseline_ndcg_dct = read_ndcg_dct('dca_herbs_expansion_%s' % args.rank_metric)
    method_ndcg_dct = read_ndcg_dct('%s_expansion_%s' % (fname, args.rank_metric))
    for k in sorted(baseline_ndcg_dct.keys()):
        baseline_ndcg_list = baseline_ndcg_dct[k]
        lda_ndcg_list = method_ndcg_dct[k]
        print 'k =', k
        print 'No expansion: %.6f; %s: %.6f' % (np.mean(baseline_ndcg_list),
            fname, np.mean(lda_ndcg_list))
        print ttest_rel(baseline_ndcg_list, lda_ndcg_list)
        print ''

if __name__ == '__main__':
    main()