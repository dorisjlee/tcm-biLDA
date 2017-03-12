### Author: Edward Huang

import numpy as np
from scipy.stats import ttest_rel
import sys
import time

### This script evaluates the NDCG's of retrieval systems by performing the
### paired t-test on them.

def read_ndcg_dct(method):
    ndcg_dct = {}
    f = open('./results/%s.txt' % (method), 'r')
    for line in f:
        ndcg, k = line.split()
        if k not in ndcg_dct:
            ndcg_dct[k] = []
        ndcg_dct[k] += [float(ndcg)]
    f.close()
    return ndcg_dct

def main():
    if len(sys.argv) not in [3, 4]:
        print 'Usage: python %s method rank_metric term_type<optional>' % sys.argv[0]
        exit()
    method = sys.argv[1]
    assert (method in ['no', 'synonym', 'lda', 'bilda', 'dca', 'med2vec',
        'pmi', 'cooccurrence', 'prosnet'])
    rank_metric = sys.argv[2]
    assert rank_metric in ['ndcg', 'precision', 'recall']
    if len(sys.argv) == 4:
        term_type = sys.argv[3]
        assert term_type in ['herbs', 'symptoms', 'mixed']
        method += '_%s' % term_type

    baseline_ndcg_dct = read_ndcg_dct('no_expansion_%s' % rank_metric)
    lda_ndcg_dct = read_ndcg_dct('%s_expansion_%s' % (method, rank_metric))
    for k in sorted(baseline_ndcg_dct.keys()):
        baseline_ndcg_list = baseline_ndcg_dct[k]
        lda_ndcg_list = lda_ndcg_dct[k]
        print k
        print 'baseline %.6f %s %.6f' % (np.mean(baseline_ndcg_list), method, np.mean(
            lda_ndcg_list))
        print ttest_rel(baseline_ndcg_list, lda_ndcg_list)
        print ''

if __name__ == '__main__':
    # start_time = time.time()
    main()
    # print "---%f seconds---" % (time.time() - start_time)