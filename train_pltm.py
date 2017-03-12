#!/usr/bin/python
# -*- coding: utf-8 -*-

from datetime import datetime
import os
import glob

if not os.path.exists('./data/sequence'):
    os.makedirs('./data/sequence')
if not os.path.exists('./data/bilda_output'):
    os.makedirs('./data/bilda_output')

def make_mallet_herb_symptom_files(fold_num):
    f = open('./data/train_test/train_no_expansion_%s.txt' % fold_num, 'r')
    symptom_corpus = open('./data/sequence/symptom%s.txt' % fold_num, 'w')
    herb_corpus = open('./data/sequence/herb%s.txt' % fold_num, 'w')
    # TODO: adding in diseases.
    # disease_corpus = open('./data/sequence/disease%s.txt' % fold_num, 'w')
    for i, line in enumerate(f):
        diseases, name, dob, visit_date, symptoms, herbs = line.split('\t')

        # Format symptom and herb lists and remove duplicates.
        symptom_list = list(set(symptoms.split(':')[:-1]))
        sdoc = '%s\tSYMPT\t%s\n' % (diseases, ' '.join(symptom_list))
        symptom_corpus.write(sdoc)

        herb_list = list(set(herbs.split(':')[:-1]))
        hdoc = '%s\tHERB\t%s\n' % (diseases, ' '.join(herb_list))
        herb_corpus.write(hdoc)

        disease_list = list(set(diseases.split(':')[:-1]))
        ddoc = '%s\tDIS\t%s\n' % (diseases, ' '.join(disease_list))
        # disease_corpus.write(ddoc)

    # disease_corpus.close()
    herb_corpus.close()
    symptom_corpus.close()
    # Convert formatted text file to MALLET sequence files
    # for entity in ['herb', 'symptom', 'disease']:
    for entity in ['herb', 'symptom']:
        os.system("../mallet-2.0.7/bin/mallet import-file --input "
            "./data/sequence/%s%s.txt --output ./data/sequence/%s%s.sequences"
            " --keep-sequence --token-regex '\p{L}+'" % (entity, fold_num,
                entity, fold_num))

for fold_num in range(10):
    make_mallet_herb_symptom_files(fold_num)

for fold_num in range(10):
    os.system("../mallet-2.0.7/bin/mallet run cc.mallet.topics.PolylingualTopic"
        "Model --output-topic-keys ./data/bilda_output/pltm_output_topics{0}.txt "
        "--num-top-words 100 --language-inputs ./data/sequence/herb{0}.sequences "
        # "./data/sequence/symptom{0}.sequences ./data/sequence/disease{0}.sequences "
        "./data/sequence/symptom{0}.sequences "
        "--num-topics 96 --optimize-interval 10".format(fold_num))
