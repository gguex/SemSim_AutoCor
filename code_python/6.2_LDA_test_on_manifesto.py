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
input_file_list = ["61320_199211_pp.txt",
                   "61320_200411_pp.txt",
                   "61320_201211_pp.txt",
                   "61320_201611_pp.txt",
                   "61620_200411_pp.txt",
                   "61620_200811_pp.txt",
                   "61620_201211_pp.txt",
                   "61620_201611_pp.txt"]
# Similarity tag
sim_tag = "w2v"
# Number of test
n_test = 50

# -------------------------------------
# --- Loading and preprocessing
# -------------------------------------

# Loop on files
token_list_list = []
real_group_vec_list = []
for input_file in input_file_list:
    # Get the file paths
    text_file_path, _, _, ground_truth_path = get_all_paths(input_file, sim_tag)

    # Get tokens
    with open(text_file_path) as text_file:
        text = text_file.read()
        token_list = text.split()
        token_list_list.append(token_list)

    # Get real groups
    with open(ground_truth_path) as ground_truth:
        real_group_vec = ground_truth.read()
        real_group_vec = np.array([int(element) for element in real_group_vec.split(",")])
        real_group_vec_list.append(real_group_vec)

# The number of Topic and a-priori probabality
full_topic = list(chain.from_iterable(real_group_vec_list))
n_group = len(set(full_topic))
topic_distrib = [np.sum(full_topic == topic_id) / len(full_topic) for topic_id in set(full_topic)]

# The common voc
lda_voc = Dictionary(token_list_list)

# Make the corpus
lda_corpus = [lda_voc.doc2bow(token_list) for token_list in token_list_list]

# LDA
lda = LdaModel(lda_corpus, num_topics=n_group, alpha=topic_distrib)

# Id doc
nmi_list = []
for id_doc in range(len(token_list_list)):
    topic_per_type = lda.get_document_topics(lda_corpus[id_doc], per_word_topics=True)[1]
    type_list = []
    topic_list = []
    for type_topic_elem in topic_per_type:
        type_list.append(lda_voc.get(type_topic_elem[0]))
        topic_list.append(type_topic_elem[1][0])

    algo_group_res = [topic_list[type_list.index(token)] for token in token_list_list[id_doc]]
    nmi_list.append(normalized_mutual_info_score(real_group_vec_list[id_doc], algo_group_res))

print(nmi_list)