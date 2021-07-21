from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from code_python.local_functions import get_all_paths
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from itertools import chain

# -------------------------------------
# --- Parameters
# -------------------------------------

# File name to explore
input_file_list = ["mix_word1.txt"]
# Similarity tag
sim_tag = "w2v"
# Number of test
n_test = 50
# Number of token for chunk
n_token_per_chunk = 2

# -------------------------------------
# --- Loading and preprocessing
# -------------------------------------

# Input file
input_file = input_file_list[0]

# Get paths
text_file_path, _, _, ground_truth_path = get_all_paths(input_file, sim_tag)

# Get tokens
with open(text_file_path) as text_file:
    text = text_file.read()
    token_list = text.split()

n_chunk = int(np.ceil(len(token_list) / n_token_per_chunk))
token_list_list = []
for i in range(n_chunk):
    token_list_list.append(token_list[i*n_token_per_chunk:(i+1)*n_token_per_chunk])

# Get real groups
with open(ground_truth_path) as ground_truth:
    real_group_vec = ground_truth.read()
    real_group_vec = np.array([int(element) for element in real_group_vec.split(",")])

# The number of Topic and a-priori probabality
n_group = len(set(real_group_vec))
topic_distrib = [np.sum(real_group_vec == topic_id) / len(real_group_vec) for topic_id in set(real_group_vec)]

# The common voc
lda_voc = Dictionary(token_list_list)

# Make the corpus
lda_corpus = [lda_voc.doc2bow(token_list) for token_list in token_list_list]

nmi_vec = []
for _ in range(n_test):
    # LDA
    lda = LdaModel(lda_corpus, num_topics=n_group, alpha=topic_distrib)

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