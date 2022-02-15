from local_functions import *
import numpy as np
import random as rdm
from sklearn.metrics import normalized_mutual_info_score
from sentence_transformers import SentenceTransformer, util

# -------------------------------------
# --- Parameters
# -------------------------------------

# Input folder
input_text_folder = "corpora/wiki50_pp"
# Take stopwords
stop_words = False
# Output file name
output_file = "results/segm_sent_wiki50.csv"

# ---

# Fixed number of groups (if none, extracted from data)
fixed_n_groups = None

# Algo hyperparameters
dist_option = "max_minus"
exch_mat_opt = "u"
exch_range = 5
alpha = 30
beta = 100
kappa = 1
known_label_ratio = 0.1  # if > 0, semi-supervised model

# Number of times algo is run
n_tests = 1

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

# Load sentence model
sbert_model = SentenceTransformer("all-mpnet-base-v2")

# Create output file
with open(output_file, "w") as res_file:
    res_file.write(f"file,nmi,pk,pk_rdm,wd,wd_rdm,ext_nmi,ext_pk,ext_pk_rdm,ext_wd,ext_wd_rdm\n")

for index_file in range(len(input_text_file_list)):

    # Get text file associated files
    input_text_file = input_text_file_list[index_file]
    input_group_file = input_group_file_list[index_file]

    # Print loop status
    print(f"Computing results for {input_text_file}")

    # Load corpus
    with open(input_text_file, "r") as text_file:
        sent_list = text_file.readlines()

    # Load ground truth
    with open(input_group_file, "r") as group_file:
        group_list = group_file.read().split(",")

    # Transform the vector to get 1 group by sentence
    ind_1 = 0
    real_sent_group_vec = []
    for sent in sent_list:
        sent_token = nltk.word_tokenize(sent)
        token_group = group_list[ind_1:(ind_1 + len(sent_token))]
        real_sent_group_vec.append(int(max(set(token_group), key=token_group.count)))
        ind_1 = ind_1 + len(sent_token)
    real_sent_group_vec = np.array(real_sent_group_vec)

    # Get the number of groups if there is no group defined
    if fixed_n_groups is None:
        n_groups = len(set(real_sent_group_vec))
    else:
        n_groups = fixed_n_groups

    # Make the sentence vectors
    sentence_embeddings = sbert_model.encode(sent_list)
    # Make sim matrix
    sim_mat = np.array(util.pytorch_cos_sim(sentence_embeddings, sentence_embeddings))

    # Compute the dissimilarity matrix
    d_mat = similarity_to_dissimilarity(sim_mat, dist_option=dist_option)

    # Compute the exchange and transition matrices
    exch_mat, w_mat = exchange_and_transition_matrices(len(sent_list), exch_mat_opt=exch_mat_opt, exch_range=exch_range)

    # Loop on n_tests
    nmi_vec, pk_vec, win_diff_vec, pk_rdm_vec, win_diff_rdm_vec, \
    ext_nmi_vec, ext_pk_vec, ext_win_diff_vec, ext_pk_rdm_vec, ext_win_diff_rdm_vec = [], [], [], [], [], \
                                                                                      [], [], [], [], []
    for id_test in range(n_tests):

        # For semi-supervised results, pick some labels
        if known_label_ratio > 0:
            indices_for_known_label = rdm.sample(range(len(real_sent_group_vec)),
                                                 int(len(real_sent_group_vec) * known_label_ratio))
            known_labels = np.zeros(len(real_sent_group_vec))
            known_labels[indices_for_known_label] = real_sent_group_vec[indices_for_known_label]
            known_labels = known_labels.astype(int)
        else:
            known_labels = None
            indices_for_known_label = []

        # Compute the membership matrix
        result_matrix = spatial_clustering(d_ext_mat=d_mat,
                                           exch_mat=exch_mat,
                                           w_mat=w_mat,
                                           n_groups=n_groups,
                                           alpha=alpha,
                                           beta=beta,
                                           kappa=kappa,
                                           known_labels=known_labels)

        # Compute the groups
        algo_sent_group_vec = np.argmax(result_matrix, 1) + 1

        # Restrained results
        rstr_real_sent_group_vec = np.delete(real_sent_group_vec, indices_for_known_label)
        rstr_algo_sent_group_vec = np.delete(algo_sent_group_vec, indices_for_known_label)

        # Compute the groups on token
        real_group_vec, algo_group_vec, rstr_real_group_vec, rstr_algo_group_vec = [], [], [], []
        for i, sent in enumerate(sent_list):
            real_group_vec.extend([real_sent_group_vec[i]] * len(sent))
            algo_group_vec.extend([algo_sent_group_vec[i]] * len(sent))
            if i not in indices_for_known_label:
                rstr_real_group_vec.extend([real_sent_group_vec[i]] * len(sent))
                rstr_algo_group_vec.extend([algo_sent_group_vec[i]] * len(sent))

        # Compute nmi score
        nmi = normalized_mutual_info_score(rstr_real_sent_group_vec, rstr_algo_sent_group_vec)
        # Segmentation evaluation
        pk_res, win_diff, pk_rdm, win_diff_rdm = seg_eval(algo_sent_group_vec, real_sent_group_vec)

        # Compute token nmi score
        ext_nmi = normalized_mutual_info_score(rstr_real_group_vec, rstr_algo_group_vec)
        # Segmentation token evaluation
        ext_pk_res, ext_win_diff, ext_pk_rdm, ext_win_diff_rdm = seg_eval(algo_group_vec, real_group_vec)

        # Save results
        nmi_vec.append(nmi)
        pk_vec.append(pk_res)
        win_diff_vec.append(win_diff)
        pk_rdm_vec.append(pk_rdm)
        win_diff_rdm_vec.append(win_diff_rdm)
        ext_nmi_vec.append(ext_nmi)
        ext_pk_vec.append(ext_pk_res)
        ext_win_diff_vec.append(ext_win_diff)
        ext_pk_rdm_vec.append(ext_pk_rdm)
        ext_win_diff_rdm_vec.append(ext_win_diff_rdm)

    with open(output_file, "a") as res_file:
        res_file.write(f"{input_text_file},{np.mean(nmi_vec)},{np.mean(pk_vec)},{np.mean(pk_rdm_vec)},"
                       f"{np.mean(win_diff_vec)},{np.mean(win_diff_rdm_vec)},{np.mean(ext_nmi_vec)},"
                       f"{np.mean(ext_pk_vec)},{np.mean(ext_pk_rdm_vec)},{np.mean(ext_win_diff_vec)},"
                       f"{np.mean(ext_win_diff_rdm_vec)}\n")
