#!/usr/bin/python
# -*- coding: utf-8 -*-

### Author: Edward Huang

import argparse
import math
from multiprocessing import Pool
import numpy as np
import operator
from rank_metrics import ndcg_at_k, precision_at_k
import sys

# This script opens the test file, and based on the queries inside the test file,
# returns a set of most similar patients to each query.
# Run time: 9 minutes.

# k_list = [5, 10, 15, 20, 30]
k_list = [5, 10, 15, 20]

def read_input_file(fname):
    '''
    Returns a dictionary of a set of patient records, either training or test.
    Key: (name, dob, visit_date) -> (str, str, str)
    Value: (disease list, symptom list, herb list) -> (list(str), list(str),
            list(str))
    '''
    record_dct = {}
    f = open(fname, 'r')
    for i, line in enumerate(f):
        diseases, name, dob, visit_date, symptoms, herbs = line.strip().split('\t')
        
        disease_set = set(diseases.split(':'))
        symptom_set = set(symptoms.split(':'))
        herb_set = set(herbs.split(':'))

        key = (name, dob, visit_date)
        while key in record_dct:
            key = (name, dob, visit_date + '1')
        record_dct[key] = (disease_set, symptom_set, herb_set)
    f.close()
    return record_dct

def get_inverted_index(corpus_dct):
    '''
    Given the corpus dictionary, build the inverted dictionary.
    Key: herb or symptom -> str
    Value: number of patient visits in which the key occurs -> int
    '''
    inverted_index, avg_doc_len = {}, 0.0
    for key in corpus_dct:
        disease_set, symptom_set, herb_set = corpus_dct[key]

        # Update the total document length of the corpus.
        avg_doc_len += len(symptom_set)
        # Increment the number of documents that contain each symptom.
        for symptom in symptom_set:
            if symptom not in inverted_index:
                inverted_index[symptom] = 0
            inverted_index[symptom] += 1

        # Mixed, herb, and synonym expansions all have herbs.
        if args.method == 'synonym' or args.term_type in ['herbs', 'mixed']:
            avg_doc_len += len(herb_set)
            for herb in herb_set:
                if herb not in inverted_index:
                    inverted_index[herb] = 0
                inverted_index[herb] += 1
    # Caclulate the average document length.
    avg_doc_len /= float(len(corpus_dct))
    return inverted_index, avg_doc_len

def okapi_bm25(query, document, inverted_index, corpus_size, avg_doc_len):
    '''
    Given a query and a document, compute the Okapi BM25 score. Returns a float.
    '''
    score = 0.0
    k_1, b = 2, 0.75
    # TF term is the same since frequency is always 1.
    tf = (k_1 + 1) / (1 + k_1 * (1 - b + b * len(document) / avg_doc_len))

    shared_terms = query.intersection(document)
    for term in shared_terms:
        # Number of documents containing the current shared term.
        n_docs_term = 0
        if term in inverted_index:
            n_docs_term = inverted_index[term]
        else:
            print 'what'
            exit()
        # Compute the inverse document frequency.
        idf = math.log((corpus_size - n_docs_term + 0.5) / (n_docs_term + 0.5),
            math.e)
        # Add on the score for the current shared term.
        score += tf * idf
    return score

def get_rel_score(query_disease_set, doc_disease_set):
    '''
    This function determines how we compute a relevance score between a query's
    diseases and the document's diseases.
    '''
    # Computing the intersection between the two for gain.
    size_inter = len(query_disease_set.intersection(doc_disease_set))
    if args.rank_metric == 'ndcg':
        return size_inter / float(len(query_disease_set) * len(doc_disease_set))
    elif size_inter > 0:
        return 1
    return 0

def evaluate_retrieval(query_dct, corpus_dct):
    '''
    Given a query dictionary and a corpus dictionary, go through each query and
    determine the NDCG for its retrieval with the disease labels as relevance
    measures.
    '''
    # Map each symptom and herb to the number of patient visits it appears in.
    inverted_index, avg_doc_len = get_inverted_index(corpus_dct)
    corpus_size = len(corpus_dct)

    metric_dct = {}
    for query_key in query_dct:
        doc_score_dct = {}
        # Ignore the query herb set. q_disease is label, q_symptom is query.
        q_disease_set, q_symptom_set, q_herb_set = query_dct[query_key]

        for doc_key in corpus_dct:
            d_disease_set, d_symptom_set, d_herb_set = corpus_dct[doc_key]

            # With no query expansion, our document is just the set of symptoms.
            document = d_symptom_set
            # If synonym or herbs/mixed expansions, add herb list into document.
            if args.method == 'synonym' or args.term_type in ['herbs', 'mixed']:
                document = document.union(d_herb_set)

            # Get the score between the query and the document.
            doc_score = okapi_bm25(q_symptom_set, document, inverted_index,
                corpus_size, avg_doc_len)
            # Compute the relevance judgement.
            relevance = get_rel_score(q_disease_set, d_disease_set)
            doc_score_dct[(doc_key, relevance)] = doc_score

        sorted_scores = sorted(doc_score_dct.items(),
            key=operator.itemgetter(1), reverse=True)
        # Get the relevance rankings.
        rel_list = [pair[0][1] for pair in sorted_scores]

        # Compute different rank metrics for different values of k.
        for k in k_list:
            if k not in metric_dct:
                metric_dct[k] = []
            if args.rank_metric == 'ndcg':
                metric_dct[k] += [ndcg_at_k(rel_list, k)]
            elif args.rank_metric == 'precision':
                # metric_dct[k] += [precision_at_k(rel_list, k)]
                metric_dct[k] += [sum(rel_list[:k]) / float(k)]
            elif args.rank_metric == 'recall':
                metric_dct[k] += [sum(rel_list[:k]) / float(sum(rel_list))]
            elif args.rank_metric == 'f1':
                precision = sum(rel_list[:k]) / float(k)
                recall = sum(rel_list[:k]) / float(sum(rel_list))
                if precision == 0:
                    metric_dct[k] += [0]
                else:
                    metric_dct[k] += [2 * precision * recall / (precision + recall)]
            elif args.rank_metric == 'map':

                r = np.asarray(rel_list[:k]) != 0
                out = [precision_at_k(r, i + 1) for i in range(r.size) if r[i]]
                if not out:
                    metric_dct[k] += [0.0]
                else:
                    metric_dct[k] += [sum(out) / sum(rel_list)]
    return metric_dct

def perform_retrieval(run_num):
    # Read the test file containing the query patients.
    test_fname = './data/train_test/test_%s_%d.txt' % (query_fname, run_num)
    query_dct = read_input_file(test_fname)
    # Training set is always with no expansion.
    train_fname = './data/train_test/train_no_expansion_%d.txt' % run_num
    corpus_dct = read_input_file(train_fname)

    metric_dct = evaluate_retrieval(query_dct, corpus_dct)
    return metric_dct

def parse_args():
    global args, query_fname
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
        query_fname = '%s_expansion' % args.method
    else:
        query_fname = '%s_%s_expansion' % (args.method, args.term_type)

def main():
    parse_args()

    pool = Pool(processes=10)
    metric_dct_lst = pool.map(perform_retrieval, range(10))

    out = open('./results/%s_%s.txt' % (query_fname, args.rank_metric), 'w')
    for metric_dct in metric_dct_lst:
        for k in k_list:
            for metric in metric_dct[k]:
                out.write('%g\t%d\n' % (metric, k))
    out.close()

if __name__ == '__main__':
    main()