### Author: Edward Huang

from multiprocessing import Pool
import numpy as np
import operator
from sklearn.metrics.pairwise import cosine_similarity
import sys
import time

### This script rewrites the test files, except with query expansion performed
### on each query patient's list of symptoms. Query expansion is done by
### word embeddings.
### Run time: 73 minutes.

def get_similarity_code_list():
    '''
    Returns the mappings for the columns and rows in the similarity matrix.
    '''
    similarity_code_list = []
    f = open('./data/herb_symptom_dictionary.txt', 'r')
    for i, line in enumerate(f):
        if i == 0:
            continue
        line = line.strip().split('\t')

        line_length = len(line)
        # Some symptoms don't have good English translations.
        assert line_length == 2 or line_length == 5
        if line_length == 2:
            herb, symptom = line
        elif line_length == 5:
            herb, symptom, english_symptom, db_src, db_src_id = line
        if herb not in similarity_code_list:
            similarity_code_list += [herb]
        if symptom not in similarity_code_list:
            similarity_code_list += [symptom]
    f.close()
    return similarity_code_list

def read_similarity_matrix(similarity_code_list):
    '''
    Returns a dictionary of similarity scores.
    Key: (code_a, code_b) -> tuple(str, str)
    Value: cosine similarity -> float
    '''
    similarity_dct = {}
    f = open('./data/similarity_matrix.txt', 'r')
    for i, line in enumerate(f):
        index_a, index_b, score = line.split()
        # Map the indices to real medical codes.
        code_a = similarity_code_list[int(index_a) - 1]
        code_b = similarity_code_list[int(index_b) - 1]
        if (code_b, code_a) in similarity_dct:
            continue
        score = abs(float(score))
        similarity_dct[(code_a, code_b)] = score
    f.close()
    return similarity_dct

def read_prosnet_matrix(run_num):
    '''
    Creates the similarity matrix generated by prosnet.
    '''
    symptom_herb_list = get_count_dct('symptom',
            run_num).keys()[:] + get_count_dct('herb', run_num).keys()[:]
    num_dim = 50
    f = open('./data/prosnet_data/prosnet_node_vectors_%d_dims_%s.vec' % (
        num_dim, run_num))
    node_list, vector_matrix = [], []
    for i, line in enumerate(f):
        if i == 0:
            continue
        line = line.split()
        node = line[0]
        # Skip codes that aren't in the expansion list.
        if node not in symptom_herb_list:
            continue
        node_list += [node]
        vector_matrix += [map(float, line[1:])]
    f.close()
    # TODO: absolute value or not?
    # similarity_matrix = np.abs(cosine_similarity(np.array(vector_matrix)))
    similarity_matrix = cosine_similarity(np.array(vector_matrix))

    similarity_dct = {}
    for a, node_a in enumerate(node_list):
        for b, node_b in enumerate(node_list):
            if a == b:
                continue
            similarity_dct[(node_a, node_b)] = similarity_matrix[a][b]
    return node_list, similarity_dct

def get_count_dct(code_type, run_num):
    code_count_dct = {}
    f = open('./data/count_dictionaries/%s_count_dct_%d.txt' % (code_type,
        run_num), 'r')
    for line in f:
        code, count = line.split()
        code_count_dct[code] = count
    f.close()
    return code_count_dct

def get_expansion_terms(symptom_list, similarity_dct, similarity_code_list,
    training_code_list):
    '''
    Given a query list, find 10 terms that have the highest similarity scores
    to the symptoms in symptom_list.
    '''
    candidate_term_dct = {}
    for query_symptom in symptom_list:
        # Skip a query if it isn't in the dictionary.
        if query_symptom not in similarity_code_list:
            continue
        for training_code in training_code_list:
            # Skip candidates that are already in the query.
            if training_code in symptom_list:
                continue
            # Skip candidates that aren't in the dictionary.
            if training_code not in similarity_code_list:
                continue

            if (query_symptom, training_code) in similarity_dct:
                score = similarity_dct[(query_symptom, training_code)]
            else:
                score = similarity_dct[(training_code, query_symptom)]
            # Keep only terms that have a score above a threshold.
            if embed_type == 'dca' and score < sim_thresh:
                continue
            # TODO: similarity threshold.
            elif embed_type == 'prosnet' and score < 0.3:
                continue
            candidate_term_dct[training_code] = score
    # Get the top 10 terms.
    expansion_terms = sorted(candidate_term_dct.items(),
        key=operator.itemgetter(1), reverse=True)[:10]
    # Extract just the terms from the sorted list.
    expansion_terms = [term[0] for term in expansion_terms]
    return expansion_terms

def query_expansion(run_num, similarity_dct, similarity_code_list):
    '''
    Runs the query expansion.
    '''
    # The list of medical codes in the training set.
    if expansion_type == 'symptoms':
        training_code_list = get_count_dct('symptom', run_num).keys()[:]
    elif expansion_type == 'herbs':
        training_code_list = get_count_dct('herb', run_num).keys()[:]
    else:
        training_code_list = get_count_dct('symptom',
            run_num).keys()[:] + get_count_dct('herb', run_num).keys()[:]
        
    # Process output filename.
    out_fname = './data/train_test/test_%s_%s_expansion_%d.txt' % (embed_type,
        expansion_type, run_num)

    out = open(out_fname, 'w')
    f = open('./data/train_test/test_no_expansion_%d.txt' % run_num, 'r')
    for query in f:
        # Split by tab, fifth element, split by comma, take out trailing comma.
        query = query.split('\t')
        symptom_list = query[4].split(':')[:-1]

        # # TODO: Not expanding on patients that have at least 5 symptoms.
        # if len(symptom_list) >= 10:
        #     out.write('\t'.join(query))
        #     continue

        expansion_terms = get_expansion_terms(symptom_list, similarity_dct,
            similarity_code_list, training_code_list)

        # TODO: fill expansion terms up to 10 terms?
        # expansion_terms = expansion_terms[:10-len(symptom_list)]

        # Write expanded query to file
        expanded_query = query[:]
        if expansion_terms != []:
            expanded_query[4] += ':'.join(expansion_terms) + ':'
        
        out.write('\t'.join(expanded_query))
    f.close()
    out.close()

def main():
    if len(sys.argv) != 4:
        print ('Usage: python %s dca/prosnet herbs/symptoms/mixed s' %
            sys.argv[0])
        exit()
    # This variable determines what types of medical codes to add to the query.
    global expansion_type, embed_type, sim_thresh
    expansion_type, embed_type = sys.argv[2], sys.argv[1]
    assert expansion_type in ['herbs', 'symptoms', 'mixed']
    assert embed_type in ['dca', 'prosnet']
    sim_thresh = float(sys.argv[3])

    # The keys will become the mappings for the similarity matrix.
    if embed_type == 'dca':
        similarity_code_list = get_similarity_code_list()
        similarity_dct = read_similarity_matrix(similarity_code_list)

    pool = Pool(processes=10)
    for run_num in range(10):
        if embed_type == 'prosnet':
            similarity_code_list, similarity_dct = read_prosnet_matrix(run_num)
        # query_expansion(run_num, similarity_dct, similarity_code_list)
        pool.apply_async(query_expansion, (run_num, similarity_dct,
            similarity_code_list))
    pool.close()
    pool.join()

if __name__ == '__main__':
    # start_time = time.time()
    main()
    # print "---%f seconds---" % (time.time() - start_time)