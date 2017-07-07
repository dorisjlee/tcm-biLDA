#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
from multiprocessing import Pool
import os

if not os.path.exists('./data/sequence'):
    os.makedirs('./data/sequence')
if not os.path.exists('./data/bilda_output'):
    os.makedirs('./data/bilda_output')

def make_mallet_herb_symptom_files(fold_num):
    f = open('./data/train_test/train_no_expansion_%s.txt' % fold_num, 'r')
    symptom_corpus = open('./data/sequence/symptom%s.txt' % fold_num, 'w')
    herb_corpus = open('./data/sequence/herb%s.txt' % fold_num, 'w')
    for line in f:
        diseases, name, dob, visit_date, symptoms, herbs = line.strip().split('\t')

        # Format symptom and herb lists and remove duplicates.
        symptom_list = symptoms.split(':')
        sdoc = '%s\tSYMPT\t%s\n' % (diseases, ' '.join(symptom_list))
        symptom_corpus.write(sdoc)

        herb_list = herbs.split(':')
        hdoc = '%s\tHERB\t%s\n' % (diseases, ' '.join(herb_list))
        herb_corpus.write(hdoc)
    herb_corpus.close()
    symptom_corpus.close()

    # Convert formatted text file to MALLET sequence files
    for entity in ['herb', 'symptom']:
        os.system("../mallet-2.0.7/bin/mallet import-file --input "
            "./data/sequence/%s%s.txt --output ./data/sequence/%s%s.sequences"
            " --keep-sequence --token-regex '\p{L}+'" % (entity, fold_num,
                entity, fold_num))

def train_mallet(fold_num):
    os.system("../mallet-2.0.7/bin/mallet run cc.mallet.topics.PolylingualTopic"
        "Model --output-topic-keys ./data/bilda_output/pltm_output_{1}_topics_{0}.txt "
        "--num-top-words 100 --language-inputs ./data/sequence/herb{0}.sequences "
        "./data/sequence/symptom{0}.sequences "
        "--num-topics {1} --optimize-interval 10".format(fold_num, args.num_topics))

def parse_args():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_topics', required=True, type=int,
        help='Number of BiLDA topics.')
    args = parser.parse_args()

def main():
    parse_args()
    pool = Pool(processes=10)
    pool.map(make_mallet_herb_symptom_files, range(10))

    pool = Pool(processes=10)
    pool.map(train_mallet, range(10))

if __name__ == '__main__':
    main()