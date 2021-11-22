from local_functions import *
import numpy as np
import random as rdm
from sklearn.metrics import normalized_mutual_info_score

# -------------------------------------
# --- Parameters
# -------------------------------------

input_text_folder = "corpora/manifesto_pp"
stop_words = False
output_file = "results/clust_manifesto.csv"

# ---

# N groups (if none, extracted from data)
n_groups = None

# Block size
block_size = None

sim_tag = "w2v"
dist_option = "max_minus"
exch_mat_opt = "u"
exch_range = 15
alpha = 5
beta = 100
kappa = 0.5
known_label_ratio = 0  # if > 0, semi-supervised model

n_tests = 3

# -------------------------------------
# --- Computations
# -------------------------------------

file_list = os.listdir(input_text_folder)

# Restrict them to those with or without stopwords
file_list = [file for file in file_list if ("wostw" in file) ^ stop_words]

# Sort the list
file_list.sort()

# Split groups and non-groups file
text_file_list = [file for file in file_list if "groups" not in file]
input_text_file_list = [f"{input_text_folder}/{file}" for file in file_list if "groups" not in file]
input_group_file_list = [f"{input_text_folder}/{file}" for file in file_list if "groups" in file]

# Vector models
home = os.path.expanduser("~")
if sim_tag == "w2v":
    vector_model_path = f"{home}/Documents/data/pretrained_word_vectors/enwiki.model"
elif sim_tag == "glv":
    vector_model_path = f"{home}/Documents/data/pretrained_word_vectors/glove42B300d.model"
else:
    vector_model_path = f"{home}/Documents/data/pretrained_word_vectors/en_fasttext.model"

# Create output file
with open(output_file, "w") as res_file:
    res_file.write(f"file,nmi,pk,pk_rdm,wd,wd_rdm\n")

for index_file in range(len(input_text_file_list)):

    # Get text file associated files
    input_text_file = input_text_file_list[index_file]
    input_group_file = input_group_file_list[index_file]

    # Loading ground truth
    with open(input_group_file) as ground_truth:
        real_group_nr_vec = ground_truth.read()
        real_group_nr_vec = np.array([int(element) for element in real_group_nr_vec.split(",")])
    if n_groups is None:
        n_groups = len(set(real_group_nr_vec))

    # Loop on n_tests
    nmi_vec = []
    pk_vec = []
    win_diff_vec = []
    pk_rdm_vec = []
    win_diff_rdm_vec = []
    for id_test in range(n_tests):

        # For semi-supervised results, pick some labels
        if known_label_ratio > 0:
            indices_for_known_label = rdm.sample(range(len(real_group_nr_vec)),
                                                 int(len(real_group_nr_vec) * known_label_ratio))
            known_labels = np.zeros(len(real_group_nr_vec))
            known_labels[indices_for_known_label] = real_group_nr_vec[indices_for_known_label]
            known_labels = known_labels.astype(int)
        else:
            known_labels = None
            indices_for_known_label = []

        # Run the algorithm
        result_matrix, existing_token_list, existing_pos_list = \
            token_clustering_on_file(input_text_file, vector_model_path, dist_option,
                                     exch_mat_opt, exch_range, n_groups, alpha, beta,
                                     kappa, block_size=block_size, verbose=True, strong_pass=True)

        # Restrain real group
        real_group_vec = real_group_nr_vec[existing_pos_list]

        # Compute the groups
        algo_group_vec = np.argmax(result_matrix, 1) + 1

        # Restrained results
        rstr_real_group_vec = np.delete(real_group_vec, indices_for_known_label)
        rstr_algo_group_vec = np.delete(algo_group_vec, indices_for_known_label)

        # Compute nmi score
        nmi = normalized_mutual_info_score(rstr_real_group_vec, rstr_algo_group_vec)

        # Segmentation evaluation
        pk_res, win_diff, pk_rdm, win_diff_rdm = seg_eval(algo_group_vec, real_group_vec)

        # Save results
        nmi_vec.append(nmi)
        pk_vec.append(pk_res)
        win_diff_vec.append(win_diff)
        pk_rdm_vec.append(pk_rdm)
        win_diff_rdm_vec.append(win_diff_rdm)

    with open(output_file, "a") as res_file:
        res_file.write(f"{input_text_file},{np.mean(nmi_vec)},{np.mean(pk_vec)},{np.mean(pk_rdm_vec)},"
                       f"{np.mean(win_diff_vec)},{np.mean(win_diff_rdm_vec)}\n")
