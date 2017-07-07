### Author: Edward Huang

import argparse
from monolingual_lda_baseline import get_patient_dct
from multiprocessing import Pool
import numpy as np
import operator
import os
import cPickle
from scipy.spatial.distance import pdist, squareform
import subprocess

### Rewrites the query test files by adding on the most similar medical codes
### to the symptom section. Query expansion is done by med2vec.
### Run time:

n_iterations = 100

def generate_directories():
    data_dir = './data/med2vec/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

def read_code_list(run_num):
    code_list = []
    f = open('./data/code_lists/code_list_%s.txt' % run_num, 'r')
    for line in f:
        code_list += [line.strip()]
    f.close()
    return code_list

def create_med2vec_input(run_num, pickle_fname):
    '''
    Generates the pickle lists for running with med2vec. Writes it out to
    file. Returns the codes for this training set.
    '''
    patient_dct = get_patient_dct(run_num)
    code_list = read_code_list(run_num)
    # Make the pickle list.
    pickle_list = []
    for key in patient_dct:
        symptom_list, herb_list = patient_dct[key]
        # Convert each symptom/herb to their index in the code list.
        symptom_list = [code_list.index(symp) for symp in symptom_list]
        herb_list = [code_list.index(herb) for herb in herb_list]
        # pickle_list is where each visit is all symptoms and herbs.
        pickle_list += [symptom_list + herb_list]
        # A [-1] is the delimiter between patients.
        pickle_list += [[-1]]
    # Remove the trailing delimiter.
    pickle_list = pickle_list[:-1]
    with open(pickle_fname, 'wb') as out:
        cPickle.dump(pickle_list, out)
    return code_list

def run_med2vec(run_num):
    '''
    Calls the external med2vec.py script.
    '''
    pickle_fname = './data/med2vec/train_%s.pickle' % run_num
    code_list = create_med2vec_input(run_num, pickle_fname)

    emb_fname = './data/med2vec/embeddings_%s' % run_num
    command = '/usr/local/bin/python2.7 med2vec.py %s --n_epoch %d %d %s' % (pickle_fname,
        n_iterations, len(code_list), emb_fname)
    subprocess.call(command, shell=True)
    return emb_fname

def get_count_dct(code_type, run_num):
    code_count_dct = {}
    f = open('./data/count_dictionaries/%s_count_dct_%d.txt' % (code_type,
        run_num), 'r')
    for line in f:
        code, count = line.split()
        code_count_dct[code] = count
    f.close()
    return code_count_dct

def get_similarity_dct(emb_fname, code_list):
    data = np.load('%s.%d.npz' % (emb_fname, n_iterations - 1))
    embedding_matrix = data['W_emb']
    # Compute pairwise cosine similarity.
    similarity_matrix = squareform(pdist(embedding_matrix, 'cosine'))
    assert len(similarity_matrix) == len(embedding_matrix)
    similarity_dct = {}
    for row_i, row in enumerate(similarity_matrix):
        row_code = code_list[row_i]
        for col_i in range(row_i + 1, len(row)):
            col_code = code_list[col_i]
            # 1 - since we're using cosine distance. TODO: abs.
            similarity_dct[(row_code, col_code)] = abs(1 - row[col_i])
            # similarity_dct[(row_code, col_code)] = 1 - row[col_i]
    return similarity_dct

def get_expansion_terms(symptom_list, similarity_dct, code_list,
    training_code_list):
    '''
    Given a query list, find 10 terms that have the highest similarity scores
    to the symptoms in symptom_list.
    '''
    candidate_term_dct = {}
    for query_symptom in symptom_list:
        # Skip a query if it isn't in the dictionary.
        if query_symptom not in code_list:
            continue
        for training_code in training_code_list:
            # Skip candidates that are already in the query.
            if training_code in symptom_list:
                continue
            # Skip candidates that aren't in the dictionary.
            if training_code not in code_list:
                continue

            if (query_symptom, training_code) in similarity_dct:
                score = similarity_dct[(query_symptom, training_code)]
            else:
                score = similarity_dct[(training_code, query_symptom)]
                
            if training_code not in candidate_term_dct:
                candidate_term_dct[training_code] = 0.0
            candidate_term_dct[training_code] += score
            
    expansion_terms = sorted(candidate_term_dct.items(),
        key=operator.itemgetter(1), reverse=True)
    # Extract just the terms from the sorted list.
    expansion_terms = [term[0] for term in expansion_terms]
    return expansion_terms

def query_expansion(run_num):
    '''
    Gets the top 10 most similar codes to each query's symptom set, based on
    the embeddings computed by med2vec.
    '''
    code_list = read_code_list(run_num)
    emb_fname = './data/med2vec/embeddings_%s' % run_num
    similarity_dct = get_similarity_dct(emb_fname, code_list)
    # The list of medical codes in the training set.
    if args.term_type == 'symptoms':
        training_code_list = get_count_dct('symptom', run_num).keys()
    elif args.term_type == 'herbs':
        training_code_list = get_count_dct('herb', run_num).keys()
    else:
        training_code_list = get_count_dct('symptom', run_num
            ).keys() + get_count_dct('herb', run_num).keys()

    # Process output filename.
    out_fname = './data/train_test/test_med2vec_%s_expansion_%d.txt' % (
        args.term_type, run_num)

    out = open(out_fname, 'w')
    f = open('./data/train_test/test_no_expansion_%d.txt' % run_num, 'r')
    for query in f:
        # Split by tab, fifth element, split by comma, take out trailing comma.
        query = query.split('\t')
        symptom_list = query[4].split(':')

        expansion_terms = get_expansion_terms(symptom_list, similarity_dct,
            code_list, training_code_list)

        # TODO: only getting the top 5 terms.
        expansion_terms = expansion_terms[:5]

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
    args = parser.parse_args()

def main():
    parse_args()

    generate_directories()

    pool = Pool(processes=10)

    ### TODO HERE.
    # pool.map(run_med2vec, range(10))

    pool.map(query_expansion, range(10))

if __name__ == '__main__':
    main()