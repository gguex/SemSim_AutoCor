import nltk
import pandas as pd
import os
import errno
from scipy.linalg import expm
import warnings
import numpy as np

def sim_to_dissim_matrix(input_file, sim_tag, dist_option="minus_log", working_path=os.getcwd()):
    """
    :param input_file: name of the text input file
    :param sim_tag: similarity tag
    :param dist_option: transformation parameter from similarity to dissimilarity, eigther "minus_log" or "1_minus"
    :param working_path: a path to the SemSim_AutoCor folder or above (default = os.getcwd())
    :return: the n_token x n_token dissimilarity matrix between text tokens
    """

    # --- Defining paths --- #

    # Getting the SemSim_AutoCor folder, if above
    base_path = str.split(working_path, "SemSim_AutoCor")[0] + "SemSim_AutoCor/"

    # Path of the text file
    file_path = base_path + "corpora/" + input_file
    # Path of the types and frequencies file
    typefreq_file_path = base_path + "similarities_frequencies/" + input_file[:-4] + "_" + sim_tag + "_typefreq.txt"

    # Path of the similarity matrix
    similarities_file_path = base_path + "similarities_frequencies/" + input_file[:-4] \
                             + "_" + sim_tag + "_similarities.txt"

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

    # Computation of dissimilarity matrix with option
    if dist_option == "minus_log":
        d_mat = - np.log(sim_mat - np.min(sim_mat) + 1e-30)
    elif dist_option == "1_minus":
        d_mat = 1 - sim_mat
    else:
        warnings.warn("The parameter 'dist_option' is not recognise, setting it to 'minus_log'")
        d_mat = - np.log(sim_mat + 1e-30)

    # Compute the n_type x n_token presence matrix
    pres_mat = np.empty([0, n_token])
    for type_i in type_list:
        pres_mat = np.append(pres_mat, [[token == type_i for token in token_list]], axis=0)

    # Compute the extended distance matrix
    d_ext_mat = pres_mat.T.dot(d_mat.dot(pres_mat))

    # Return the distance_matrix
    return d_ext_mat


def exchange_and_transition_mat(input_file, sim_tag, exch_mat_opt, exch_range, working_path=os.getcwd()):
    """
    :param input_file: name of the text input file
    :param sim_tag: similarity tag
    :param exch_mat_opt: option for the exchange matrix, "s" = standard, "u" = uniform, "d" = diffusive
    :param exch_range: range of the exchange matrix
    :param working_path: a path to the SemSim_AutoCor folder or above (default = os.getcwd())
    :return: the n_token x n_token exchange matrix and the n_token x n_token markov transition matrix
    """

    # --- Defining paths --- #

    # Getting the SemSim_AutoCor folder, if above
    base_path = str.split(working_path, "SemSim_AutoCor")[0] + "SemSim_AutoCor/"

    # Path of the text file
    file_path = base_path + "corpora/" + input_file
    # Path of the types and frequencies file
    typefreq_file_path = base_path + "similarities_frequencies/" + input_file[:-4] + "_" + sim_tag + "_typefreq.txt"

    # Raise errors if files not found
    if not os.path.exists(file_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_path)
    if not os.path.exists(typefreq_file_path):
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

    # --- Compute the exchange matrix and transition matrix --- #

    # Compute the exchange matrix and the markov chain transition matrix
    f_vec = np.ones(n_token) / n_token

    if exch_mat_opt not in ["s", "u", "d"]:
        warnings.warn("Exchange matrix option ('exch_mat_opt') not recognized, setting it to 's'")
        exch_mat_opt = "s"

    if exch_mat_opt  == "s":
        exch_mat = np.abs(np.add.outer(np.arange(n_token), -np.arange(n_token))) <= exch_range
        np.fill_diagonal(exch_mat, 0)
        exch_mat = exch_mat / np.sum(exch_mat)
    elif exch_mat_opt == "u":
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

    # Return the transition matrix
    return exch_mat, w_mat


def discontinuity_segmentation(d_ext_mat, exch_mat, w_mat, n_groups, alpha, beta, kappa,
                               conv_threshold=1e-5, max_it=100, init_labels=None):
    """
    :param d_ext_mat: the n_token x n_token distance matrix
    :param exch_mat: the n_token x n_token exchange matrix
    :param w_mat: the n_token x n_token Markov chain transition matrix
    :param n_groups: the number of groups
    :param alpha: alpha parameter
    :param beta: beta parameter
    :param kappa: kappa parameter
    :param conv_threshold: convergence threshold (default = 1e-5)
    :param max_it: maximum iterations (default = 100)
    :param init_labels: a vector containing initial labels (default = None)
    :return: the n_tokens x n_groups membership matrix for each token
    """

    # Getting the number of token
    n_token, _ = d_ext_mat.shape

    # Compute the weights of token
    f_vec = np.sum(exch_mat, 0)

    # Initialization of Z
    z_mat = np.random.random((n_token, n_groups))
    z_mat = (z_mat.T / np.sum(z_mat, axis=1)).T

    # Set true labels
    # If init_labels is not None, set known to value
    if init_labels is not None:
        for i, label in enumerate(init_labels):
            if label != 0:
                z_mat[i, :] = 0
                z_mat[i, label - 1] = 1

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
        if np.sum(-hig_mat > 690) > 0:
            warnings.warn("Overflow of exp(-hig_mat)")
            hig_mat[-hig_mat > 690] = -690
        z_new_mat = rho_vec * np.exp(-hig_mat)
        z_new_mat = (z_new_mat.T / np.sum(z_new_mat, axis=1)).T

        # If init_labels is not None, set known to value
        if init_labels is not None:
            for i, label in enumerate(init_labels):
                if label != 0:
                    z_new_mat[i, :] = 0
                    z_new_mat[i, label - 1] = 1

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
