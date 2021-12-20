from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from tqdm import tqdm
import os
from local_functions import seg_eval

# -------------------------------------
# --- Parameters
# -------------------------------------

# Input folder
input_text_folder = "corpora/manifesto_pp"
# Take stopwords
stop_words = False
# Output file name
output_file_name = "results/3.1_clust_results/LDA_clust_manifesto.csv"

# N groups (if None, extracted from data)
n_groups = None

# Number of tests
n_tests = 20

# Hyperparameters
chunk_size = 2250
use_prior = False

# -------------------------------------
# --- Computations
# -------------------------------------

# List files
file_list = os.listdir(input_text_folder)

# Restrict them to those with or without stopwords
file_list = [file for file in file_list if ("wostw" in file) ^ stop_words]

# Sort the list
file_list.sort()

# Split groups and non-groups file
text_file_list = [file for file in file_list if "groups" not in file]
input_text_file_list = [f"{input_text_folder}/{file}" for file in file_list if "groups" not in file]
input_group_file_list = [f"{input_text_folder}/{file}" for file in file_list if "groups" in file]

# Make results file
with open(output_file_name, "w") as output_file:
    output_file.write("input_file,n_groups,n_tests,use_prior,chunk_size,"
                      "mean_nmi,mean_pk,mean_rdm_pk,mean_wd,mean_rdm_wd\n")

for index_file in range(len(input_text_file_list)):

    # Get text file associated files
    input_text_file = input_text_file_list[index_file]
    input_group_file = input_group_file_list[index_file]

    # Get real groups
    with open(input_group_file) as ground_truth:
        real_group_vec = ground_truth.read()
        real_group_vec = np.array([int(element) for element in real_group_vec.split(",")])
    if n_groups is None:
        n_groups = len(set(real_group_vec))

    # The number of Topic and a-priori probabality
    topic_distrib = [np.sum(real_group_vec == topic_id) / len(real_group_vec) for topic_id in set(real_group_vec)]

    # Get tokens
    with open(input_text_file) as text_file:
        text = text_file.read()
        token_list = text.split()

    # Divide the corpus by chunk
    n_chunk = int(np.ceil(len(token_list) / chunk_size))
    token_list_list = []
    for i in range(n_chunk):
        token_list_list.append(token_list[i * chunk_size:(i + 1) * chunk_size])

    # The common voc
    lda_voc = Dictionary(token_list_list)

    # Make the corpus
    lda_corpus = [lda_voc.doc2bow(token_list) for token_list in token_list_list]

    # -- Make the n tests
    nmi_list, pk_list, pk_rdm_list, wd_list, wd_rdm_list = [], [], [], [], []
    for _ in tqdm(range(n_tests)):

        # LDA
        if use_prior:
            lda = LdaModel(lda_corpus, num_topics=n_groups, alpha=topic_distrib)
        else:
            lda = LdaModel(lda_corpus, num_topics=n_groups)

        # Id doc
        algo_group_vec = []
        for id_doc in range(len(token_list_list)):
            topic_per_type = lda.get_document_topics(lda_corpus[id_doc], per_word_topics=True)[1]
            type_list = []
            topic_list = []
            for type_topic_elem in topic_per_type:
                type_list.append(lda_voc.get(type_topic_elem[0]))
                topic_list.append(type_topic_elem[1][0])

            algo_group_vec.extend([topic_list[type_list.index(token)] for token in token_list_list[id_doc]])

        # Save nmi
        nmi_list.append(normalized_mutual_info_score(real_group_vec, algo_group_vec))

        # Segmentation evaluation
        pk, wd, pk_rdm, wd_rdm = seg_eval(algo_group_vec, real_group_vec)
        pk_list.append(pk)
        pk_rdm_list.append(pk_rdm)
        wd_list.append(wd)
        wd_rdm_list.append(wd_rdm)

    # Writing results
    with open(output_file_name, "a") as output_file:
        output_file.write(f"{input_text_file},{n_groups},{n_tests},{use_prior},{chunk_size},{np.mean(nmi_list)},"
                          f"{np.mean(pk_list)},{np.mean(pk_rdm_list)},{np.mean(wd_list)},{np.mean(wd_rdm_list)}\n")
