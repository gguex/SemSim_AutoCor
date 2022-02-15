import os
from local_functions import similarity_to_dissimilarity, \
    exchange_and_transition_matrices, spatial_clustering, seg_eval
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from itertools import product
import multiprocessing as mp
import nltk
from miniutils import parallel_progbar
from sentence_transformers import SentenceTransformer, util

# -------------------------------------
# --- Parameters
# -------------------------------------

base_path = os.getcwd()

# ------------ Options

input_text_file = "corpora/manifesto_pp/61320_201211_pp_wostw.txt"
input_group_file = "corpora/manifesto_pp/61320_201211_pp_wostw_groups.txt"

results_file_name = "results/2_hyperparam_search/SentGn_61320_201211_pp_wostw.csv"

# N groups (if None, extracted from data)
n_groups = None

# Number of tests
n_tests = 3

# Search on
dist_option_vec = ["max_minus"]
exch_mat_opt_vec = ["u"]
exch_range_vec = [5, 10, 15]
alpha_vec = [1, 2, 5, 10, 30]
beta_vec = [5, 10, 50, 100, 200]
kappa_vec = [0, 0.25, 0.5, 0.75, 1]

# Number of cpu to use
n_cpu = mp.cpu_count()

# -------------------------------------
# --- Computation
# -------------------------------------

# Creating hyperparameters for multiproc
hyperp_list = list(product(alpha_vec, beta_vec, kappa_vec))

# Make results file
with open(results_file_name, "w") as output_file:
    output_file.write(f"input_file,n_groups,n_tests,dist_option,exch_mat_opt,exch_range,"
                      f"alpha,beta,kappa,mean_nmi,mean_pk,mean_rdm_pk,mean_wd,mean_rdm_wd,"
                      f"mean_ext_nmi,mean_ext_pk,mean_rdm_ext_pk,mean_ext_wd,mean_rdm_ext_wd\n")

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
if n_groups is None:
    n_groups = len(set(real_sent_group_vec))

# Load sentence model
sbert_model = SentenceTransformer("all-mpnet-base-v2")
# Make the sentence vectors
sentence_embeddings = sbert_model.encode(sent_list)
# Make sim matrix
sim_mat = np.array(util.pytorch_cos_sim(sentence_embeddings, sentence_embeddings))

########################
# -- Loop on dist option

for dist_option in dist_option_vec:

    # Compute the dissimilarity matrix
    d_mat = similarity_to_dissimilarity(sim_mat, dist_option=dist_option)

    ######################
    # -- Loop on exchg opt

    for exch_mat_opt, exch_range in product(exch_mat_opt_vec, exch_range_vec):

        # Compute the exchange and transition matrices
        exch_mat, w_mat = exchange_and_transition_matrices(len(sent_list), exch_mat_opt=exch_mat_opt,
                                                           exch_range=exch_range)

        ########################################
        # -- Creating a function to multiprocess


        def val_computation(alpha, beta, kappa):
            # Compute the matrix and val  n_train time
            nmi_list, pk_list, pk_rdm_list, wd_list, wd_rdm_list = [], [], [], [], []
            ext_nmi_list, ext_pk_list, ext_pk_rdm_list, ext_wd_list, ext_wd_rdm_list = [], [], [], [], []
            for _ in range(n_tests):
                # Compute the membership matrix
                res_matrix = spatial_clustering(d_ext_mat=d_mat,
                                                exch_mat=exch_mat,
                                                w_mat=w_mat,
                                                n_groups=n_groups,
                                                alpha=alpha,
                                                beta=beta,
                                                kappa=kappa)
                # Compute the aglo groups
                algo_sent_group_vec = np.argmax(res_matrix, 1) + 1

                # Compute nmi score
                nmi = normalized_mutual_info_score(real_sent_group_vec, algo_sent_group_vec)
                nmi_list.append(nmi)
                # Segmentation evaluation
                pk, wd, pk_rdm, wd_rdm = seg_eval(algo_sent_group_vec, real_sent_group_vec)
                pk_list.append(pk)
                pk_rdm_list.append(pk_rdm)
                wd_list.append(wd)
                wd_rdm_list.append(wd_rdm)

                # Compute the groups on token
                ext_real_group_vec = []
                ext_algo_group_vec = []
                for i, sent in enumerate(sent_list):
                    ext_real_group_vec.extend([real_sent_group_vec[i]] * len(sent))
                    ext_algo_group_vec.extend([algo_sent_group_vec[i]] * len(sent))

                # Compute nmi score
                ext_nmi = normalized_mutual_info_score(ext_real_group_vec, ext_algo_group_vec)
                ext_nmi_list.append(nmi)
                # Segmentation evaluation
                ext_pk, ext_wd, ext_pk_rdm, ext_wd_rdm = seg_eval(ext_algo_group_vec, ext_real_group_vec)
                ext_pk_list.append(ext_pk)
                ext_pk_rdm_list.append(ext_pk_rdm)
                ext_wd_list.append(ext_wd)
                ext_wd_rdm_list.append(ext_wd_rdm)

            return np.mean(nmi_list), np.mean(pk_list), np.mean(pk_rdm_list), np.mean(wd_list), \
                np.mean(wd_rdm_list), np.mean(ext_nmi_list), np.mean(ext_pk_list), np.mean(ext_pk_rdm_list), \
                np.mean(ext_wd_list), np.mean(ext_wd_rdm_list)


        ##################################
        # -- Computing and writing results

        # Print message
        print(f"Multiprocessing for {dist_option}, {exch_mat_opt}, {exch_range}")
        # Multiprocess
        res_multi = parallel_progbar(val_computation, hyperp_list, starmap=True, nprocs=n_cpu)

        # Writing results
        with open(results_file_name, "a") as output_file:
            for id_hyp, hyperp in enumerate(hyperp_list):
                output_file.write(f"{input_text_file},{n_groups},{n_tests},"
                                  f"{dist_option},{exch_mat_opt},{exch_range},{hyperp[0]},{hyperp[1]},"
                                  f"{hyperp[2]},{res_multi[id_hyp][0]},{res_multi[id_hyp][1]},"
                                  f"{res_multi[id_hyp][2]},{res_multi[id_hyp][3]},{res_multi[id_hyp][4]},"
                                  f"{res_multi[id_hyp][5]},{res_multi[id_hyp][6]},{res_multi[id_hyp][7]},"
                                  f"{res_multi[id_hyp][8]},{res_multi[id_hyp][9]}\n")
