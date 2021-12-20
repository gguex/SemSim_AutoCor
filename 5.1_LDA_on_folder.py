import os
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
import numpy as np
from sklearn.metrics import normalized_mutual_info_score

# -------------------------------------
# --- Parameters
# -------------------------------------

# Input folder
input_text_folder = "corpora/manifesto_pp"
# Take stopwords
stop_words = False
# Output file name
output_file = "results/LDA_nmi_manifesto.csv"

# Number of test
n_test = 50

# Number of groups (if none, extracted from data)
n_groups = None

# -------------------------------------
# --- Loading and preprocessing
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

# Make the corpus as list of list of tokens
token_list_list = []
real_group_vec_list = []
for i, input_text_file in enumerate(input_text_file_list):
    with open(input_text_file) as text_file:
        text = text_file.read()
        token_list_list.append(text.split())
    with open(input_group_file_list[i]) as group_file:
        real_group_vec = group_file.read()
        real_group_vec_list.append(np.array([int(element) for element in real_group_vec.split(",")]))

# The number of Topics
if n_groups is None:
    all_group_vec = [gr for real_gr_vec in real_group_vec_list for gr in real_gr_vec]
    n_groups = len(set(all_group_vec))

# The common voc
lda_voc = Dictionary(token_list_list)

# Make the corpus
lda_corpus = [lda_voc.doc2bow(token_list) for token_list in token_list_list]

nmi_vec = []
for _ in range(n_test):
    # LDA
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

    nmi_vec.append(normalized_mutual_info_score(real_group_vec, algo_group_vec))

print(np.mean(nmi_vec))