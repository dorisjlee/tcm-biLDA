import argparse
from multiprocessing import Pool
import numpy as np
import os
import lda

### This script runs regular LDA on a patient record training set (90% of the
### original data). Writes out the herb counts, symptom counts, code list (for
### mapping symptoms/herbs to integers), and the word distributions for each
### topic. The number of topics will match the number of unique diseases.

def generate_folders():
    for folder in ('./results', './data/count_dictionaries',
        './data/code_lists', './results/lda_word_distributions'):
        if not os.path.exists(folder):
            os.makedirs(folder)

def get_patient_dct(run_num):
    '''
    Returns dictionary
    Key: (name, date of birth) -> (str, str)
    Value: dictionary, where keys are (name, DOB) pairs and values are tuples
    containing the diseases, diagnosis dates, symptoms, and herbs of each visit.
    '''
    # Keep track of the unique set of diseases so we know n_topics.
    patient_dct = {}

    f = open('./data/train_test/train_no_expansion_%s.txt' % run_num, 'r')
    for line in f:
        diseases, name, dob, visit_date, symptoms, herbs = line.strip().split('\t')

        symptom_list = symptoms.split(':')
        herb_list = herbs.split(':')
        patient_dct[(name, dob)] = (symptom_list, herb_list)
    f.close()
    return patient_dct

def get_symptom_and_herb_counts(patient_dct, run_num):
    '''
    Given the patient dictionary, count the symptom and herb occurrences in
    patients with more than one visit. Writes the counts out to file.
    Returns the list of unique medical codes.
    '''
    symptom_count_dct, herb_count_dct = {}, {}
    for key in patient_dct:
        symptom_list, herb_list = patient_dct[key]

        # Update the counts of each symptom and herb.
        for symptom in symptom_list:
            if symptom not in symptom_count_dct:
                symptom_count_dct[symptom] = 0
            symptom_count_dct[symptom] += 1
        for herb in herb_list:
            if herb not in herb_count_dct:
                herb_count_dct[herb] = 0
            herb_count_dct[herb] += 1

    dct_folder = './data/count_dictionaries'

    # Write out the unique symptoms and herbs to file.
    herb_out = open('%s/herb_count_dct_%s.txt' % (dct_folder, run_num), 'w')
    for herb in herb_count_dct:
        herb_out.write('%s\t%d\n' % (herb, herb_count_dct[herb]))
    herb_out.close()

    symptom_out = open('%s/symptom_count_dct_%s.txt' % (dct_folder, run_num), 'w')
    for symptom in symptom_count_dct:
        symptom_out.write('%s\t%d\n' % (symptom, symptom_count_dct[symptom]))
    symptom_out.close()

    return list(set(symptom_count_dct.keys()).union(herb_count_dct.keys()))

def write_code_list(code_list, run_num):
    '''
    Writes the code list out to file.
    '''
    out = open('./data/code_lists/code_list_%s.txt' % run_num, 'w')
    out.write('\n'.join(code_list))
    out.close()

def get_matrix_from_dct(patient_dct, code_list):
    '''
    Convert the patient dictionary to a document-term matrix.
    '''
    patient_matrix = []
    for key in patient_dct:
        symptom_list, herb_list = patient_dct[key]

        curr_code_list = symptom_list + herb_list
        # Create binary vectors for each patient visit.
        curr_row = [1 if c in curr_code_list else 0 for c in code_list]
        patient_matrix += [curr_row]
    return np.array(patient_matrix)

def run_baseline_lda(patient_matrix):
    model = lda.LDA(n_topics=args.num_topics, n_iter=5000, random_state=1)
    model.fit(patient_matrix)
    topic_word = model.topic_word_
    return topic_word

def lda_pipeline(run_num):
    patient_dct = get_patient_dct(run_num)

    # code_list is the vocabulary list.
    code_list = get_symptom_and_herb_counts(patient_dct, run_num)
    write_code_list(code_list, run_num)

    # Convert the patient dictionary to a matrix for LDA.
    patient_matrix = get_matrix_from_dct(patient_dct, code_list)

    # Run LDA.
    topic_word = run_baseline_lda(patient_matrix)
    np.savetxt('./results/lda_word_distributions/lda_word_distribution_%s_%s.txt' %
        (args.num_topics, run_num), topic_word)

def parse_args():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_topics', required=True, type=int,
        help='Number of monolingual LDA topics to train on.')
    args = parser.parse_args()

def main():
    generate_folders()
    parse_args()

    pool = Pool(processes=10)
    pool.map(lda_pipeline, range(10))

if __name__ == '__main__':
    main()