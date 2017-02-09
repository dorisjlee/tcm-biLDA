# Author: Edward Huang

from med2vec_query_expansion import read_code_list, get_count_dct
from monolingual_lda_baseline import get_patient_dct
import numpy as np
import operator
from scipy.linalg import svd
from scipy.spatial.distance import pdist, squareform
import sys
import time

# This script adds highest PMI pairs to queries as expansion methods.

def build_pmi_dct(co_occ_dct, run_num):
    '''
    Given the co-occurrence dictionary, simply divide each value by the counts
    of its keys to get the pointwise mutual information.
    '''
    code_count_dct = get_count_dct('herb', run_num)
    code_count_dct.update(get_count_dct('symptom', run_num))

    pmi_dct = {}
    for code_a, code_b in co_occ_dct:
        if co_occ_dct[(code_a, code_b)] == 0:
            continue
        pmi_dct[(code_a, code_b)] = np.log2(co_occ_dct[(code_a, code_b)] /
            (float(code_count_dct[code_a]) * float(code_count_dct[code_b])))
    return pmi_dct

def get_svd_pmi_matrix(co_occ_matrix, code_list, run_num):
    '''
    Given a co-occurrence matrix, perform SVD and return the dictionary of 
    cosine similarity terms between the low-dimensional vectors.
    '''
    pmi_matrix = np.array(co_occ_matrix)

    code_count_dct = get_count_dct('herb', run_num)
    code_count_dct.update(get_count_dct('symptom', run_num))

    # Divide the rows.
    for code_i, code in enumerate(code_list):
        code_count = float(code_count_dct[code])
        pmi_matrix[code_i] /= code_count
        pmi_matrix[:,code_i] /= code_count
    pmi_matrix[pmi_matrix == 0] = 1
    return np.log2(pmi_matrix)

def reduce_matrix(matrix, code_list):
    '''
    Generically performs SVD on any co-occurrence matrix. Argument 'matrix' can
    either be the co-occurrence matrix or the PMI matrix.
    '''
    U, s, Vh = svd(matrix)
    # for k in [50, 100, 150]:
    # TODO: Try different k's.
    k = 100

    top_indices = sorted(range(len(s)), key=lambda i: s[i])[-k:]
    # Get the top singular values.
    top_s = [s[i] for i in top_indices]

    reduced_Vh = []
    for row in Vh:
        row = [row[i] for i in top_indices]
        # Multiply the top singular values by each row in V_h.
        reduced_Vh += [np.array(row) * np.sqrt(top_s)]

    # Compute the pairwise cosine similarity.
    similarity_matrix = squareform(pdist(reduced_Vh, 'cosine'))

    similarity_dct = {}
    for row_i, row in enumerate(similarity_matrix):
        row_code = code_list[row_i]
        for col_i in range(row_i + 1, len(row)):
            col_code = code_list[col_i]
            similarity_dct[(row_code, col_code)] = 1 - row[col_i]

    # Sort the pairs of codes by their cosine simliarity.
    return similarity_dct

def get_similarity_dct(run_num):
    '''
    If code_list has n elements, build an n x n matrix of co-occurrence values.
    Also writes out the top pairs as scored by co-occurrence counts.
    '''
    patient_fname = './data/train_test/train_no_expansion_%d.txt' % run_num
    patient_dct, disease_set = get_patient_dct(patient_fname)
    # Pulled from med2vec_query_expansion.py
    code_list = read_code_list(run_num)

    # Also initialize the matrix for the SVD approaches.
    if 'svd' in similarity_metric:
        co_occ_matrix = [[0.0 for i in range(len(code_list))] for j in range(
            len(code_list))]

    # The dictionary is for co-occurrence/PMI purposes.
    co_occ_dct = {}

    for key in patient_dct:
        visit_dct = patient_dct[key]
        # Skip patients that only had one visit.
        if len(visit_dct) == 1:
            continue
        for date in sorted(visit_dct.keys()):
            disease_list, symptom_list, herb_list = visit_dct[date]
            combined_list = symptom_list + herb_list

            # For the dictionary, only count each pair once.
            if 'svd' not in similarity_metric:
                for a in range(len(combined_list)):
                    code_a = combined_list[a]
                    for b in range(a + 1, len(combined_list)):
                        code_b = combined_list[b]
                        # Update the dictionary.
                        if (code_a, code_b) not in co_occ_dct:
                            co_occ_dct[(code_a, code_b)] = 0.0
                        co_occ_dct[(code_a, code_b)] += 1
            # We count each pair twice for the matrix.
            else:
                int_code_lst = [code_list.index(code) for code in combined_list]
                for code_a in int_code_lst:
                    for code_b in int_code_lst:
                        co_occ_matrix[code_a][code_b] += 1

    # If we're computing the co-occurrence as similarity score, return early.
    if similarity_metric == 'cooccurrence':
        return co_occ_dct
    elif similarity_metric == 'pmi':
        return build_pmi_dct(co_occ_dct, run_num)
    elif similarity_metric == 'svd_pmi':
        # This is the PMI matrix.
        pmi_matrix = get_svd_pmi_matrix(co_occ_matrix, code_list, run_num)
        return reduce_matrix(pmi_matrix, code_list)
    elif similarity_metric == 'svd_cooccurrence':
        return reduce_matrix(co_occ_matrix, code_list)        

def get_expansion_terms(symptom_list, sim_lst, run_num):
    '''
    Given a symptom list and a sorted similarity list, get the ten words with
    the highest score to all symptoms in symptom_list.
    '''
    if expansion_type == 'symptoms':
        candidate_list = get_count_dct('symptom', run_num).keys()
    elif expansion_type == 'herbs':
        candidate_list = get_count_dct('herb', run_num).keys()
    else:
        candidate_list = get_count_dct('symptom', run_num
            ).keys() + get_count_dct('herb', run_num).keys()

    expansion_terms = []
    for (code_a, code_b), score in sim_lst:
        a_in_list = code_a in symptom_list
        b_in_list = code_b in symptom_list
        # Case where pair is already in symptom list.
        if a_in_list and b_in_list:
            continue
        elif a_in_list and code_b in candidate_list:
            expansion_terms += [code_b]
        elif b_in_list and code_a in candidate_list:
            expansion_terms += [code_a]
        if len(expansion_terms) == 10:
            break
    return expansion_terms

def query_expansion(run_num):
    '''
    Finds the top PMI pairs to add as query expansion terms.
    '''
    # Process filename.
    out_fname = './data/train_test/test_%s_%s_expansion_%d.txt' % (
        similarity_metric, expansion_type, run_num)

    out = open(out_fname, 'w')
    f = open('./data/train_test/test_no_expansion_%d.txt' % run_num, 'r')
    for query in f:
        # Split by tab, fifth element, split by comma, take out trailing comma.
        query = query.split('\t')
        symptom_list = query[4].split(':')[:-1]

        similarity_dct = get_similarity_dct(run_num)
        # Sort the similarity dictionary.
        sim_lst = sorted(similarity_dct.items(), key=operator.itemgetter(1),
            reverse=True)
        expansion_terms = get_expansion_terms(symptom_list, sim_lst, run_num)

        # Write expanded query to file
        expanded_query = query[:]
        expanded_query[4] += ':'.join(expansion_terms) + ':'
        
        out.write('\t'.join(expanded_query))
    f.close()
    out.close()

def main():
    if len(sys.argv) != 3:
        print ('Usage: python %s pmi/cooccurrence/svd_pmi/svd_cooccurrence '
            'herbs/symptoms/mixed' % sys.argv[0])
        exit()
    # This variable determines what types of medical codes to add to the query.
    global expansion_type, similarity_metric
    similarity_metric, expansion_type = sys.argv[1:]
    assert (similarity_metric in ['pmi', 'cooccurrence', 'svd_pmi',
        'svd_cooccurrence'])
    assert expansion_type in ['herbs', 'symptoms', 'mixed']

    for run_num in range(10):
        query_expansion(run_num)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print "---%f seconds---" % (time.time() - start_time)