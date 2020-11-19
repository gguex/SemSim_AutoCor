import numpy as np
import nltk
import pandas as pd
import os
import errno
from scipy.linalg import expm
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score


# --- Defining function --- #

def compute_discontinuity_segment_token(working_path, input_file, sim_tag, exch_mat_opt, exch_range, n_groups,
                                        alpha, beta, kappa, conv_threshold=1e-5, max_it=1000):
    """
    :param working_path: working path of the SemSim_AutoCor folder, or a folder above
    :param input_file: name of the text input file
    :param sim_tag: similarity tag
    :param exch_mat_opt: option for the exchange matrix, "u" = uniform, "d" = diffusive
    :param exch_range: range of the exchange matrix
    :param n_groups: number of groups
    :param alpha: alpha parameter
    :param beta: beta parameter
    :param kappa: kappa parameter
    :param conv_threshold: convergence threshold (default = 1e-5)
    :param max_it: maximum iteration (default = 1000)
    :return: A n_token x n_group matrix with group membership for each tokens
    """

    # Saving the SemSim_AutoCor folder, if above
    base_path = str.split(working_path, "SemSim_AutoCor")[0] + "SemSim_AutoCor/"

    # Path of the text file
    file_path = base_path + "corpora/" + input_file
    # Path of the types and frequencies file
    typefreq_file_path = base_path + "similarities_frequencies/" + input_file[:-4] + "_" + sim_tag + "_typefreq.txt"

    # Path of the similarity matrix
    similarities_file_path = base_path + "similarities_frequencies/" + input_file[:-4] + \
                             "_" + sim_tag + "_similarities.txt"

    # Raise errors if files not found
    if not os.path.exists(file_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_path)
    if not os.path.exists(typefreq_file_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), typefreq_file_path)
    if not os.path.exists(similarities_file_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), typefreq_file_path)

    # --- Load the data --- #

    # Import the type freq file
    type_freq_df = pd.read_csv(typefreq_file_path, sep=";", header=None)
    type_list = list(type_freq_df[0])

    # Import the text file and remove non-existing token
    with open(file_path, "r") as text_file:
        text_string = text_file.read()
    raw_token_list = nltk.word_tokenize(text_string)
    token_list = [token for token in raw_token_list if token in type_list]
    n_token = len(token_list)  # The number of tokens

    # Import the similarity matrix
    sim_mat = np.loadtxt(similarities_file_path, delimiter=";")

    # --- Computation before loop --- #

    # Compute the exchange matrix and the markov chain transition matrix
    f_vec = np.ones(n_token) / n_token

    if exch_mat_opt not in ["u", "d"]:
        print("EXCHANGE MATRIX OPTION NOT RECOGNIZED, SETTING IT TO UNIFORM")
        exch_mat_opt = "u"

    if exch_mat_opt == "u":
        adj_mat = np.abs(np.add.outer(np.arange(n_token), -np.arange(n_token))) <= exch_range
        np.fill_diagonal(adj_mat, 0)
        g_vec = np.sum(adj_mat, axis=1) / np.sum(adj_mat)
        k_vec = f_vec / g_vec
        b_mat = np.array([[min(v1, v2) for v2 in k_vec] for v1 in k_vec]) * adj_mat / np.sum(adj_mat)
        exch_mat = np.diag(f_vec) - np.diag(np.sum(b_mat, axis=1)) + b_mat
    else:
        adj_mat = np.abs(np.add.outer(np.arange(n_token), -np.arange(n_token))) <= 1
        np.fill_diagonal(adj_mat, 0)
        l_adj_mat = np.diag(np.sum(adj_mat, axis=1)) - adj_mat
        pi_outer_mat = np.outer(np.sqrt(f_vec), np.sqrt(f_vec))
        phi_mat = (l_adj_mat / pi_outer_mat) / np.trace(l_adj_mat)
        exch_mat = expm(- exch_range * phi_mat) * pi_outer_mat

    w_mat = (exch_mat / np.sum(exch_mat, axis=1)).T

    # Easy computation of dissimilarity matrix
    d_mat = 1 - sim_mat

    # Compute the n_type x n_token presence matrix
    pres_mat = np.empty([0, n_token])
    for type_i in type_list:
        pres_mat = np.append(pres_mat, [[token == type_i for token in token_list]], axis=0)

    # Compute the extended distance matrix
    d_ext_mat = pres_mat.T.dot(d_mat.dot(pres_mat))

    # --- Loop --- #

    # Initialization of Z
    z_mat = np.random.random((n_token, n_groups))
    z_mat = (z_mat.T / np.sum(z_mat, axis=1)).T

    # Control of the loop
    converge = False

    # Loop
    it = 0
    print("Starting loop")
    while not converge:

        # Computation of rho_g vector
        rho_vec = np.sum(z_mat.T * f_vec, axis=1)

        # Computation of f_i^g matrix
        fig_mat = ((z_mat / rho_vec).T * f_vec).T

        # Computation of D_i^g matrix
        dig_mat = fig_mat.T.dot(d_ext_mat).T
        delta_g_vec = 0.5 * np.diag(dig_mat.T.dot(fig_mat))
        dig_mat = dig_mat - delta_g_vec

        # Computation of the epsilon_g vector
        epsilon_g = np.sum(exch_mat.dot(z_mat ** 2), axis=0) - np.diag(z_mat.T.dot(exch_mat.dot(z_mat)))

        # Computation of H_ig
        hig_mat = beta * dig_mat + alpha * (rho_vec ** -kappa) * (z_mat - w_mat.dot(z_mat)) \
                  - (0.5 * alpha * kappa * (rho_vec ** (-kappa - 1)) * epsilon_g)

        # Computation of the new z_mat
        z_new_mat = rho_vec * np.exp(-hig_mat)
        z_new_mat = (z_new_mat.T / np.sum(z_new_mat, axis=1)).T

        # Print diff and it
        diff_pre_new = np.linalg.norm(z_mat - z_new_mat)
        it += 1
        print("Iteration {}: {}".format(it, diff_pre_new))

        # Verification of convergence
        if diff_pre_new < conv_threshold:
            converge = True
        if it > max_it:
            converge = True

        # Saving the new z_mat
        z_mat = z_new_mat

    # Return the result
    return z_mat


# Parameters for the gride

# Real values
with open("/home/gguex/PycharmProjects/SemSim_AutoCor/corpora/mixgroup_sent1_min5.txt") as group_file:
    real_group_vec = group_file.read()
    real_group_vec = np.array([int(element) for element in real_group_vec.split(",")])

# Test the function
result_matrix = compute_discontinuity_segment_token(working_path=os.getcwd(),
                                                    input_file="mix_sent1_min5.txt",
                                                    sim_tag="wesim",
                                                    exch_mat_opt="d",
                                                    exch_range=5,
                                                    n_groups=4,
                                                    alpha=5,
                                                    beta=10,
                                                    kappa=1,
                                                    max_it=40)

algo_group_value = np.argmax(result_matrix, 1) + 1

conf_matrix = confusion_matrix(real_group_vec, algo_group_value)
nmi = normalized_mutual_info_score(real_group_vec, algo_group_value)

print(conf_matrix)
print(f"NMI = {nmi}")