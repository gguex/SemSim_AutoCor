from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from code_python.local_functions import get_all_paths
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from tqdm import tqdm
from local_functions import write_groups_in_html_file

# -------------------------------------
# --- Parameters
# -------------------------------------

# File name to explore
# input_file_list = ["mix_word1.txt",
#                    "mix_word5.txt",
#                    "mix_sent1.txt",
#                    "mix_sent5.txt"] * 7
# input_file_list = ["61320_199211_pp.txt",
#                    "61320_200411_pp.txt",
#                    "61320_201211_pp.txt",
#                    "61320_201611_pp.txt",
#                    "61620_200411_pp.txt",
#                    "61620_200811_pp.txt",
#                    "61620_201211_pp.txt",
#                    "61620_201611_pp.txt"]
input_file_list = ["61620_200411_pp.txt"]
# Use prior distrib for topics list
prior_distrib_list = [False] * len(input_file_list)
# Number of tests list
n_test_list = [1] * len(input_file_list)
# Number of tokens for chunk
# n_token_per_chunk_list = [2, 7, 17, 85, 2, 7, 17, 85]
# n_token_per_chunk_list = [20] * 4 + [50] * 4 + [100] * 4 + [200] * 4 + [300] * 4 + [400] * 4 + [500] * 4
n_token_per_chunk_list = [300] * len(input_file_list)

# Results file name
results_file_name = "../results/6_lda_results/lda_mix_fixed_size.csv"

# -------------------------------------
# --- Loading and preprocessing
# -------------------------------------

# Make results file
with open(results_file_name, "w") as output_file:
    output_file.write("input_file,use_prior,n_test,n_token_per_chunk,mean_nmi,ci95_nmi\n")

# Loop on files
for i, input_file in enumerate(input_file_list):

    # Get parameters
    prior_distrib = prior_distrib_list[i]
    n_test = n_test_list[i]
    n_token_per_chunk = n_token_per_chunk_list[i]

    # Print
    print(f"File: {input_file}, Prior: {prior_distrib}, Nb tests: {n_test}, Nb tokens per chunk: {n_token_per_chunk}")

    # Get paths
    text_file_path, _, _, ground_truth_path = get_all_paths(input_file, "w2v")

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
    for _ in tqdm(range(n_test)):

        # LDA
        if prior_distrib:
            lda = LdaModel(lda_corpus, num_topics=n_group, alpha=topic_distrib)
        else:
            lda = LdaModel(lda_corpus, num_topics=n_group)

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

    # Writing results
    nmi_mean = np.mean(nmi_vec)
    nmi_std = np.std(nmi_vec)
    with open(results_file_name, "a") as output_file:
        output_file.write(f"{input_file},{prior_distrib},{n_test},{n_token_per_chunk},{nmi_mean},"
                          f"{nmi_std * 1.96 / np.sqrt(n_test)}\n")


# Compute the real membership matrix
z_algo_mat = np.zeros((len(token_list), n_group))
for i, label in enumerate(algo_group_vec):
    if label != 0:
        z_algo_mat[i, :] = 0
        z_algo_mat[i, label - 1] = 1

write_groups_in_html_file("lda.html", token_list, z_algo_mat, comment_line=f"nmi = {nmi_mean}")
