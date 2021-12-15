from local_functions import *
import os
import numpy as np
from sklearn.metrics import normalized_mutual_info_score

# -------------------------------------
# --- Parameters
# -------------------------------------

input_text_folder = "corpora/manifesto_pp"
stop_words = False
output_file = "results/clust_ww_manifesto.csv"
output_word_stat_file = "results/word_stat_manifesto.csv"

# ---

# N groups (if none, extracted from data)
n_groups = None

sim_tag = "w2v"
dist_option = "max_minus"
exch_mat_opt = "u"
exch_range = 15
alpha = 5
beta = 100
kappa = 0.5

n_tests = 3

# ---

# List of words for each groups (prop 1)
word_for_group = [["nato", "israel", "korea", "capabilities", "nuclear",
                   "europe", "allies", "north", "peace", "weapons"],
                  ["democracy", "constitutional", "rights", "political", "puerto",
                   "rico", "freedom", "citizens", "civil", "independent"],
                  ["vision", "local", "eliminate", "governments", "government",
                   "state", "administration", "rather", "party"],
                  ["consumers", "small", "growth", "businesses", "tax",
                   "market", "business", "companies", "markets", "gas"],
                  ["school", "discrimination", "students", "conservation", "schools",
                   "quality", "low", "college", "education", "early"],
                  ["crime", "crimes", "faith", "enforcement", "religious",
                   "legal", "values", "law", "seeking", "violence"],
                  ["farmer", "labor", "union", "workers", "class",
                   "join", "middle", "jobs", "built", "good"]]

# List of words for each groups (prop 2)
# word_for_group = [["united", "military", "security", "nuclear", "international", "nations", "peace", "forces",
#                     "allies", "war"],
#                    ["constitution", "constitutional", "vote", "political", "constitutions", "elections", "document",
#                     "separation", "judiciary", "declaration"],
#                    ["local", "state", "washington", "government", "governments", "tribal", "control",
#                     "corruption", "audit", "vision"],
#                    ["tax", "economic", "economy", "energy", "growth", "small", "businesses", "private",
#                     "investment", "trade"],
#                    ["health", "care", "education", "democrats", "students", "schools", "disabilities",
#                     "access", "school", "public"],
#                    ["abortion", "religious", "marriage", "family", "enforcement", "faith", "life", "crime",
#                     "ban", "oppose"],
#                    ["workers", "jobs", "labor", "veterans", "class", "union", "middle", "work",
#                     "farmers", "wages"]]

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
input_sim_file_list = [f"similarity_matrices/{file[:-4]}_{sim_tag}.csv" for file in file_list if "groups" not in file]

with open(output_file, "w") as res_file:
    res_file.write(f"file,nmi,pk,pk_rdm,wd,wd_rdm\n")

with open(output_word_stat_file, "w") as res_file:
    res_file.write(f"file,group,nb,%nb,%right\n")

for index_file in range(len(input_text_file_list)):

    input_text_file = input_text_file_list[index_file]
    input_group_file = input_group_file_list[index_file]
    input_sim_file = input_sim_file_list[index_file]

    # Loading the similarity matrix
    type_list, sim_mat = load_sim_matrix(input_sim_file)

    # Compute the extended version of the similarity matrix
    sim_ext_mat, token_list, existing_index_list = type_to_token_matrix_expansion(input_text_file, sim_mat, type_list)

    # Loading ground truth
    with open(input_group_file) as ground_truth:
        real_group_vec = ground_truth.read()
        real_group_vec = np.array([int(element) for element in real_group_vec.split(",")])
    real_group_vec = real_group_vec[existing_index_list]
    if n_groups is None:
        n_groups = len(set(real_group_vec))

    # Label the words
    known_labels = np.zeros(len(real_group_vec))
    indices_for_known_label = []
    for i in range(len(word_for_group)):
        group_words = word_for_group[i]
        indices_group_words = [token_id for token_id, token in enumerate(token_list) if token in group_words]
        indices_for_known_label.extend(indices_group_words)
        known_labels[indices_group_words] = (i + 1)
    known_labels = known_labels.astype(int)

    # Some stats for the words
    nb_in_group_list = []
    nb_right_group_list = []
    with open(output_word_stat_file, "a") as res_file:
        for i in range(n_groups):
            nb_in_group = sum(known_labels == i + 1)
            nb_right_groups = sum(real_group_vec[known_labels == i + 1] == i + 1)
            nb_in_group_list.append(nb_in_group)
            nb_right_group_list.append(nb_right_groups)
            res_file.write(f"{input_text_file},{i + 1},{nb_in_group},{nb_in_group / sum(real_group_vec == i + 1)},"
                           f"{nb_right_groups / nb_in_group}\n")

        res_file.write(f"{input_text_file},Overall,{sum(nb_in_group_list)},{sum(nb_in_group_list) / len(token_list)},"
                       f"{sum(nb_right_group_list) / sum(nb_in_group_list)}\n")

    # Compute the dissimilarity matrix
    d_ext_mat = similarity_to_dissimilarity(sim_ext_mat, dist_option=dist_option)

    # Compute the exchange and transition matrices
    exch_mat, w_mat = exchange_and_transition_matrices(len(token_list),
                                                       exch_mat_opt=exch_mat_opt,
                                                       exch_range=exch_range)

    # Loop on n_tests
    nmi_vec = []
    pk_vec = []
    win_diff_vec = []
    pk_rdm_vec = []
    win_diff_rdm_vec = []
    for id_test in range(n_tests):
        # Compute the membership matrix
        result_matrix = token_clustering(d_ext_mat=d_ext_mat, exch_mat=exch_mat, w_mat=w_mat, n_groups=n_groups,
                                         alpha=alpha, beta=beta, kappa=kappa, init_labels=known_labels, verbose=True)

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