#!/usr/bin/python
# -*- coding: utf-8 -*-

import collections
from multiprocessing import Pool
import os 
import numpy as np
import sys
import time

### This script rewrites the test files, except with query expansion performed
### on each query patient's list of symptoms.

def flatten_list(sub_topic_words, sympt_cooc_count):
    assert len(sub_topic_words) == len(sympt_cooc_count)
    # top_topic_words = np.array(sub_topic_words)[top_topic_indices]
    flattened_lst = []
    for topic_idx, cooc_count in enumerate(sympt_cooc_count):
        for i in range(cooc_count):
            flattened_lst += sub_topic_words[topic_idx]
    return flattened_lst

def get_highest_cooccuring_words(symptom_set, run_num):
    '''
    Given the symptom list, find the highest co-occuring topics
    then find the highest co-occuring words in those topics.
    '''
    # Read the PLTM output.
    f = open('./data/bilda_output/pltm_output_topics%s.txt' % run_num, 'r')
    herb_topic_words, symp_topic_words = [], []
    for line in f:
        line = line.split('\t')
        if len(line) == 2:
            continue
        word_lst = line[3].split()
        if line[0] == ' 0':
            herb_topic_words += [word_lst]
        elif line[0] == ' 1':
            symp_topic_words += [word_lst]
    # Get the indices of the topics that match the query best.
    # sympt_cooccurence = [filter(lambda x: x in symptom_set, topic_lst
    #     ) for topic_lst in symp_topic_words]
    sympt_cooc_count = [len(symptom_set.intersection(topic_symptoms)
        ) for topic_symptoms in symp_topic_words]
    sympt_cooc_count = [count if count > 1 else 0 for count in sympt_cooc_count]
    # TODO: change the number of top topics.
    # top_topic_indices = np.argsort(sympt_cooc_count)[::-1][:3*num_symptoms+1]
    # print top_topic_indices
    # print sympt_cooc_count
    # top_topic_indices = [i for i, e in enumerate(sympt_cooc_count) if e >= 1]
    # Remove zeros from the counts, so that we can multiply the co-occurrence.
    # sympt_cooc_count = filter(lambda a: a != 0, sympt_cooc_count)
    # Get the symptoms and herbs from the best topics.
    if expansion_type == 'mixed':
        flattened_lst = flatten_list(herb_topic_words, sympt_cooc_count)
        flattened_lst += flatten_list(symp_topic_words, sympt_cooc_count)
    elif expansion_type == 'symptoms':
        flattened_lst = flatten_list(symp_topic_words, sympt_cooc_count)
    elif expansion_type == 'herbs':
        flattened_lst = flatten_list(herb_topic_words, sympt_cooc_count)

    expansion_terms = []
    word_counter = collections.Counter(flattened_lst)
    for term, count in word_counter.most_common(10):
        if term not in symptom_set:
            expansion_terms += [term]
    return expansion_terms

def query_expansion(run_num):
    '''
    Goes through the basic test queries created by train_test_split.py, and adds
    on the words that most co-occur with the query symptoms. Co-occurrence is
    computed by appearances in the topics output by the different LDA models.
    '''
    out_fname = './data/train_test/test_bilda_%s_expansion_%d.txt' % (
        expansion_type, run_num)

    out = open(out_fname, 'w')
    f = open('./data/train_test/test_no_expansion_%d.txt' % run_num, 'r')
    num_bad = 0
    for query in f:
        # Split by tab, fifth element, split by comma, take out trailing comma.
        query = query.split('\t') 
        symptom_set = set(query[4].split(':')[:-1])

        expansion_terms = get_highest_cooccuring_words(symptom_set, run_num)

        # # # TODO: Not expanding on patients that have at least 5 symptoms.
        if len(symptom_set) >= num_terms or len(expansion_terms) == 0:
        # # if len(expansion_terms) == 0:
            out.write('\t'.join(query))
            continue
        num_bad += 1
        # Only taking the top 5 symptoms. TODO.
        expansion_terms = expansion_terms[:num_terms-len(symptom_set)]
        # expansion_terms = expansion_terms[:num_terms]

        # Write expanded query to file
        expanded_query = query[:]
        expanded_query[4] += ':'.join(expansion_terms) + ':'

        out.write('\t'.join(expanded_query))
    f.close()
    out.close()
    print num_bad

def main():
    if len(sys.argv) != 3:
        print 'Usage: python %s herbs/symptoms/mixed num_terms' % sys.argv[0]
        exit()
    global expansion_type, num_terms
    expansion_type = sys.argv[1]
    assert expansion_type in ['herbs', 'symptoms', 'mixed']
    num_terms = int(sys.argv[2])

    pool = Pool(processes=10)
    for run_num in range(10):
        pool.apply_async(query_expansion, (run_num,))
    pool.close()
    pool.join()

if __name__ == '__main__':
    # start_time = time.time()
    main()
    # print "---%f seconds---" % (time.time() - start_time)