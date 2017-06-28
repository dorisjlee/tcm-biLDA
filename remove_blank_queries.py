### Author: Edward Huang

from datetime import datetime

### This script rewrites the input file, removing any patient records that have
### blank symptom or herb lists.

def rewrite_input_file():
    # Map each first visit to the line to be written out.
    first_visit_dct, key_line_dct = {}, {}
    f = open('./data/HIS_tuple_word.txt', 'r')
    for line in f:
        diseases, name, dob, visit_date, symptoms, herbs = line.split('\t')
        # Skip null keys.
        if name == 'null' or dob == 'null' or visit_date == 'null':
            continue

        disease_list = list(set(diseases.split(':')[:-1]))
        symptom_list = list(set(symptoms.split(':')[:-1]))
        herb_list = list(set(herbs.split(':')[:-1]))
        # Skip empty labels, symptoms, or herbs.
        if 0 in (len(disease_list), len(symptom_list), len(herb_list)):
            continue

        # We exclude dosages. Dosages are in grams, so they have a 'G'.
        filtered_herb_list = [herb for herb in herb_list if 'G' not in herb]
        visit_line = '%s\t%s\t%s\t%s\t%s\t%s:\n' % (diseases, name, dob,
            visit_date, symptoms, ':'.join(filtered_herb_list))
        # Check to make sure that this is indeed the first visit.
        converted_date = datetime.strptime(visit_date[7:visit_date.index('(')],
            '%Y-%m-%d')
        key = (name, dob)
        # Update dictionaries if patient isn't in yet or if current date is earlier
        # than visit already in the dictionary.
        if key not in first_visit_dct or converted_date < first_visit_dct[key]:
            first_visit_dct[key] = converted_date
            key_line_dct[key] = visit_line
    f.close()

    out = open('./data/clean_HIS_tuple_word_first_visit.txt', 'w')
    for key in key_line_dct:
        out.write(key_line_dct[key])
    out.close()

def main():
    rewrite_input_file()

if __name__ == '__main__':
    main()