# tcm-biLDA

### Preprocessing
    
1.  Removes blank records, as well as records that have null name/dob.

    ```bash
    $ python remove_blank_queries.py
    ```

2.  Splits the HIS_tuple_word.txt data file into 10 different pairs of train/
    test data sets.

    ```bash
    $ python train_test_split.py
    ```

### LDA training

1.  Runs regular LDA for each of the ten training sets. Writes out the word
    distributions in nxm format. n is the number of unique diseases (number of
    topics), and m is the number of codes. Code mappings are under ./data/code_lists/

    ```bash
    $ python monolingual_lda_baseline.py
    ```

2.  Runs PLTM for two languages. Reduces to BiLDA.

    ```bash
    python train_pltm.py
    ```

### Query Expansions

1.  Adds the appropriate query expansion terms to each patient record's list of
    symptoms. Add 10 expansion terms. Requires monolingual_lda_baseline.py.

    ```bash
    $ python lda_query_expansion.py herbs/symptoms/mixed
    ```

2.  Adds the synonyms to each query based on the herb-symptom dictionary. Can
    only add herbs. Adds all synonyms, since we have no measure of similarity.
    Does not require monolingual_lda_baseline.py

    ```bash
    $ python synonym_query_expansion.py
    ```

3.  Adds the most similar terms by word embedding. Run from PaReCat's
    embeddings on the network constructed by the herb-symptom dictionary.

    ```bash
    $ python dca_query_expansion.py herbs/symptoms/mixed
    ```

4.  med2vec query expansion. Must first grab the med2vec.py file from the
    https://github.com/mp2893/med2vec.

    ```bash
    $ python med2vec_query_expansion.py
    ```

5.  PMI or co-occurrence-based query expansion.

    ```bash
    $ python cooccurrence_query_expansion.py pmi/cooccurrence herbs/symptoms/mixed
    ```

### Method Evaluations

1.  Evaluates the retrievals using Okapi BM25. Relevant documents are patient
    records that share diseases with the query patient. Currently using NDCG 
    as the rank metric, where gain is calculated by the number of intersecting
    diseases between the query and document.

    ```bash
    $ python retrieval_evaluation.py no/lda_symptoms/lda_herbs/lda_mixed/bilda_symptoms/bilda_herbs/bilda_mixed/dca_symptoms/dca_herbs/dca_mixed/med2vec_symptoms/med2vec_herbs/med2vec_mixed/synonym/pmi_herbs/pmi_symptoms/pmi_mixed/cooccurrence_herbs/cooccurrence_symptoms/cooccurrence_mixed rank_metric
    ```

    rank_metric in ['ndcg', 'precision', 'recall']

2.  Runs the paired t-tests comparing each of the topic model-based query
    expansion to the baseline without expansion.

    ```bash
    $ python significance_test.py no/lda_symptoms/lda_herbs/lda_mixed/bilda_symptoms/bilda_herbs/bilda_mixed/dca_symptoms/dca_herbs/dca_mixed/med2vec_symptoms/med2vec_herbs/med2vec_mixed/synonym/pmi_herbs/pmi_symptoms/pmi_mixed/cooccurrence_herbs/cooccurrence_symptoms/cooccurrence_mixed rank_metric
    ```