#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import collections
from itertools import izip
from multiprocessing import Pool
from operator import itemgetter

### This script rewrites the test files, except with query expansion performed
### on each query patient's list of symptoms.

# def flatten_list(sub_topic_words, sympt_cooc_count):
#     assert len(sub_topic_words) == len(sympt_cooc_count)
#     flattened_lst = []
#     for topic_idx, cooc_count in enumerate(sympt_cooc_count):
#         for i in range(cooc_count):
#             flattened_lst += sub_topic_words[topic_idx]
#     return flattened_lst

def get_highest_cooccuring_words(symptom_set, run_num):
    '''
    Given the symptom list, find the highest co-occuring topics
    then find the highest co-occuring words in those topics.
    '''
    # Read the PLTM output.
    f = open('./data/bilda_output/pltm_output_%d_topics_%s.txt' % (
        args.num_topics, run_num), 'r')
    topic_herb_dct, topic_sympt_dct = {}, {}
    # for line in f:
    while True:
        topic = f.readline() # Read the topic number and alpha parameter.
        if not topic:
            break
        herb_line = f.readline().split('\t')
        symptom_line = f.readline().split('\t')
        # Remove language number and beta parameter.
        herb_lst = herb_line[3].split()
        symptom_lst = symptom_line[3].split()        
        # Get number of query terms in symptom topic.
        num_inter = len(symptom_set.intersection(symptom_lst))
        # Skip a topic if there are no query expansion terms in it.
        if num_inter == 0:
            continue
        # Process herbs and their counts.
        h = iter(herb_lst)
        for herb, herb_count in izip(h, h):
            if herb not in topic_herb_dct:
                topic_herb_dct[herb] = 0.0
            # TODO. not multipying by num_inter.
            topic_herb_dct[herb] += float(herb_count) * num_inter
            # topic_herb_dct[herb] += float(herb_count)
        # Process symptoms and their counts.
        s = iter(symptom_lst)
        for symptom, symptom_count in izip(s, s):
            if symptom not in topic_sympt_dct:
                topic_sympt_dct[symptom] = 0.0                
            topic_sympt_dct[symptom] += float(symptom_count) * num_inter * 6
            # topic_sympt_dct[symptom] += float(symptom_count) * num_inter * 100000000000
            # topic_sympt_dct[symptom] += float(symptom_count)
            # # Check if it's a weight.
            # if line[0] == ' 0':
            #     if word not in topic_herb_dct:
            #         topic_herb_dct[word] = 0
            #     topic_herb_dct[word] += float(word_count) * len(symptom_set.intersection(word_lst))
            # elif line[0] == ' 1':
            #     if word not in topic_sympt_dct:
            #         topic_sympt_dct[word] = 0
            #     topic_sympt_dct[word] += float(word_count) * len(symptom_set.intersection(word_lst))
    # Get the indices of the topics that match the query best.
    # sympt_cooc_count = [len(symptom_set.intersection(topic_symptoms)
    #     ) for topic_symptoms in symp_topic_words]
    # sympt_cooc_count = [count if count > 1 else 0 for count in sympt_cooc_count]

    # # Get the symptoms and herbs from the best topics.
    if args.term_type == 'mixed':
        word_count_dct = topic_sympt_dct
        word_count_dct.update(topic_herb_dct)
    #     flattened_lst = flatten_list(herb_topic_words, sympt_cooc_count)
    #     flattened_lst += flatten_list(symp_topic_words, sympt_cooc_count)
    elif args.term_type == 'symptoms':
        word_count_dct = topic_sympt_dct
    #     flattened_lst = flatten_list(symp_topic_words, sympt_cooc_count)
    elif args.term_type == 'herbs':
        word_count_dct = topic_herb_dct
    #     flattened_lst = flatten_list(herb_topic_words, sympt_cooc_count)

    expansion_terms = []
    # word_counter = collections.Counter(flattened_lst)
    # for term, count in word_counter.most_common(10):
    for term, count in sorted(word_count_dct.items(), key=itemgetter(1), reverse=True):
        if term not in symptom_set:
            if term in topic_herb_dct:
                print ','.join(symptom_set)
            expansion_terms += [term]
        if len(expansion_terms) == args.num_terms:
            break
    return expansion_terms

def query_expansion(run_num):
    '''
    Goes through the basic test queries created by train_test_split.py, and adds
    on the words that most co-occur with the query symptoms. Co-occurrence is
    computed by appearances in the topics output by the different LDA models.
    '''
    out_fname = './data/train_test/test_bilda_%s_expansion_%d.txt' % (
        args.term_type, run_num)

    out = open(out_fname, 'w')
    f = open('./data/train_test/test_no_expansion_%d.txt' % run_num, 'r')
    for query in f:
        # Split by tab, fifth element, split by comma, take out trailing comma.
        query = query.split('\t')
        symptom_set = set(query[4].split(':'))

        expansion_terms = get_highest_cooccuring_words(symptom_set, run_num)

        if len(expansion_terms) == 0:
            out.write('\t'.join(query))
            continue

        # expansion_terms = expansion_terms[:args.num_terms]

        # Write expanded query to file
        expanded_query = query[:]
        expanded_query[4] += ':' + ':'.join(expansion_terms)

        out.write('\t'.join(expanded_query))
    f.close()
    out.close()

def parse_args():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--term_type', choices=['herbs', 'symptoms',
        'mixed'], required=True, help='Type of query expansion terms.')
    parser.add_argument('-n', '--num_topics', required=True, type=int,
        help='Number of topics to train on.')
    parser.add_argument('-e', '--num_terms', required=True, type=int,
        help='Number of expansion terms to add.')
    args = parser.parse_args()

def main():
    parse_args()

    # query_expansion(0)
    pool = Pool(processes=10)
    pool.map(query_expansion, range(10))

if __name__ == '__main__':
    main()