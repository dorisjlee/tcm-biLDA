### Author: Edward Huang

import argparse
import math
from multiprocessing import Pool
import numpy as np

### This script rewrites the test files, except with query expansion performed
### on each query patient's list of symptoms.

def read_code_list(run_num):
    code_list = []
    f = open('./data/code_lists/code_list_%d.txt' % run_num, 'r')
    for line in f:
        code_list += [line.strip()]
    f.close()
    return code_list

def get_symptom_count_dct(run_num):
    '''
    We only use this to determine if a code is a symptom or not.
    '''
    symptom_count_dct = {}
    f = open('./data/count_dictionaries/symptom_count_dct_%d.txt' % run_num,
        'r')
    for line in f:
        symptom, count = line.split()
        symptom_count_dct[symptom] = count
    f.close()
    return symptom_count_dct

def get_scaled_topic(symptom_list, word_distr, code_list):
    '''
    Given a symptom list (i.e., query), and the word distributions output by
    some LDA run, we want to recompute the topic probabilities. For each topic,
    we multiply the word probabilities of that topic by the number of query 
    terms that appear in the top 100 words. Add together these topics
    elementwise.
    '''
    # Number of top words to define a topic.
    n_top_words = 100
    scaled_topic = np.zeros(len(code_list))

    for topic_dist in word_distr:
        # Get the topic's top 100 words by word probability.
        topic_words = np.array(code_list)[np.argsort(topic_dist)][:-(
            n_top_words + 1):-1]
        # Number of query terms in the top n words.
        num_shared_terms = len(set(topic_words).intersection(symptom_list))
        # Scale each topic by the number of terms it shares with the query.
        scaled_topic += topic_dist * num_shared_terms

    return scaled_topic

def get_highest_prob_words(symptom_list, scaled_topic, code_list,
    symptom_count_dct):
    '''
    Given the scaled topic, find the top words to add to the given query. Add
    on twice as many expansion terms as there are symptoms.
    '''
    expansion_terms = []
    # TODO: I reversed this.
    highest_prob_words = np.array(code_list)[np.argsort(scaled_topic)][::-1]
    for candidate in highest_prob_words:
        # Decide whether to add a candidate based on args.term_type.
        candidate_is_symptom = candidate in symptom_count_dct

        if args.term_type == 'herbs' and candidate_is_symptom:
            continue
        elif args.term_type == 'symptoms' and not candidate_is_symptom:
            continue
        if candidate not in symptom_list:
            expansion_terms += [candidate]
    return expansion_terms

def query_expansion(run_num):
    '''
    Goes through the basic test queries created by train_test_split.py, and adds
    on the words that most co-occur with the query symptoms. Co-occurrence is
    computed by appearances in the topics output by the different LDA models.
    '''
    code_list = read_code_list(run_num)
    word_distr = np.loadtxt('./results/lda_word_distributions/lda_word_distrib'
        'ution_%d_%d.txt' % (args.num_topics, run_num))
    symptom_count_dct = get_symptom_count_dct(run_num)

    # Process filename.
    out_fname = './data/train_test/test_lda_%s_expansion_%d.txt' % (
        args.term_type, run_num)

    out = open(out_fname, 'w')
    f = open('./data/train_test/test_no_expansion_%d.txt' % run_num, 'r')
    for query in f:
        # Split by tab, fifth element, split by colon.
        query = query.split('\t')

        symptom_list = query[4].split(':')

        scaled_topic = get_scaled_topic(symptom_list, word_distr, code_list)

        expansion_terms = get_highest_prob_words(symptom_list, scaled_topic,
            code_list, symptom_count_dct)

        # Adding up to num_terms expansion terms.
        expansion_terms = expansion_terms[:args.num_terms]
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
        help='Number of monolingual LDA topics to train on.')
    parser.add_argument('-e', '--num_terms', required=True, type=int,
        help='Number of expansion terms to add.')
    args = parser.parse_args()

def main():
    parse_args()

    pool = Pool(processes=10)
    pool.map(query_expansion, range(10))

if __name__ == '__main__':
    main()