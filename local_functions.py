import os
import nltk
import colorsys
from scipy.linalg import expm
import warnings
import numpy as np
from gensim.models import KeyedVectors
from tqdm import tqdm
from segeval import convert_positions_to_masses, pk, window_diff
import random as rdm


def get_all_paths(input_file, sim_tag, working_path=os.getcwd(), warn=True):
    """
    A function specific to the Semsim_Autocor project that returns all the useful paths
    given a input file name and a similarity tag. If any file is missing, raise a warning and return an empty string
    (the missing of ground truth file don't raise a warning, as it is not always needed)

    :param input_file: the name of the text file
    :type input_file: str
    :param sim_tag: the similarity tag
    :type sim_tag: str
    :param working_path: the initial path of script, must be anywhere in the SemSim_AutoCor project
    :type working_path: str
    :param warn: option to deactivate warnings
    :type warn: bool
    :return: paths of : the corpus file, the similarity file and the ground truth file.
    :rtype: (str, str, str)
    """

    # Getting the SemSim_AutoCor folder, if above
    base_path = str.split(working_path, "SemSim_AutoCor")[0] + "SemSim_AutoCor"

    # Path of the text file
    text_file_path = f"{base_path}/corpora/{input_file}"
    # Path of the similarity matrix
    sim_file_path = f"{base_path}/similarity_matrices/{input_file[:-4]}_{sim_tag}_similarities.txt"
    # Path of the ground true file
    ground_truth_path = f"{base_path}/corpora/{input_file[:-4]}_groups.txt"

    if warn:
        if not os.path.exists(text_file_path):
            warnings.warn(f"The corpus file '{text_file_path}' is missing")
            text_file_path = ""
        if not os.path.exists(sim_file_path):
            warnings.warn(f"The similarity file '{sim_file_path}' is missing")
            sim_file_path = ""
        if not os.path.exists(ground_truth_path):
            ground_truth_path = ""

    # Return paths
    return text_file_path, sim_file_path, ground_truth_path


def build_wv_similarity_matrix(corpus_path, output_file_path, wv_model):
    """
    Build and save a similarity matrix given a file and a word vector model in a gensim format

    :param corpus_path: the path of the corpus file
    :type corpus_path: str
    :param output_file_path: the path of the outputted similarity files
    :type output_file_path: str
    :param wv_model: the word vector model in gensim format
    :type wv_model: gensim.models.keyedvectors.Word2VecKeyedVectors
    """

    # Opening the corpus file
    with open(corpus_path, "r") as text_file:
        text_string = text_file.read()

    # Split by tokens
    token_list = nltk.word_tokenize(text_string)

    # Get type list and frequencies
    type_freq_dict = nltk.FreqDist(token_list)
    vocab_text = set(type_freq_dict.keys())

    # build vocabulary
    vocab_wv = set(wv_model.vocab.keys())

    # Find the common vocabulary
    vocab_common = list(vocab_wv & vocab_text)

    # Write the similarity file
    with open(output_file_path, "w") as sim_matrix_file:
        for type_1 in tqdm(vocab_common):
            sim_matrix_file.write(type_1 + ",")
            for type_2 in vocab_common:
                sim_matrix_file.write(str(wv_model.similarity(type_1, type_2)))
                if type_2 != vocab_common[len(vocab_common) - 1]:
                    sim_matrix_file.write(",")
                else:
                    sim_matrix_file.write("\n")


def load_sim_matrix(sim_matrix_path):
    """
    Load a similarity matrix in csv format with types on the first column

    :param sim_matrix_path: the path of the similarity matrix
    :type sim_matrix_path: str
    :return: the list of token corresponding to the row and columns of the similarity matrix
    and the similarity matrix itself
    :rtype: (list[str], numpy.ndarray)
    """
    # Load read file
    with open(sim_matrix_path, "r", encoding="utf-8", errors="ignore") as csv_sim_file:
        row_list = csv_sim_file.readlines()
        type_list = []
        sim_mat = []
        for row in row_list:
            row_elem = row.strip().split(",")
            type_list.append(row_elem[0])
            sim_mat.append(row_elem[1:])

    # Transform the list of list into numpy array
    sim_mat = np.array(sim_mat)
    sim_mat = sim_mat.astype(np.float)
    np.fill_diagonal(sim_mat, 1)

    # Return elements
    return type_list, sim_mat


def type_to_token_matrix_expansion(text_file_path, type_mat, type_list):
    """
    Transform the (n_type x n_type) similarity of dissimilarity matrix to its extended version,
    with size (n_token x n_token). Return the extended matrix and the list of used tokens in corresponding order.

    :param text_file_path: path of the text file containing tokens from which to build the extended matrix.
    :type text_file_path: str
    :param type_mat: the (n_type x n_type) similarity or dissimilarity matrix between types.
    :type type_mat: numpy.ndarray
    :param type_list: the list of types defining the rows and columns of the (n_type x n_type) matrix.
    :type type_list: list[str]
    :return: the (n_token x n_token) extended matrix between text tokens, the (n_token) list of tokens used and
    the (n_token) indices in the original text of token exisiting in type_list.
    :rtype: (numpy.ndarray, list[str], list[int])
    """

    # Import the text file
    with open(text_file_path, "r") as text_file:
        text_string = text_file.read()
    raw_token_list = nltk.word_tokenize(text_string)

    # Keep only the tokens which are present in the type_list and create indices of existing tokens
    token_list = []
    existing_index_list = []
    for i, token in enumerate(raw_token_list):
        if token in type_list:
            token_list.append(token)
            existing_index_list.append(i)
    n_token = len(token_list)

    # Compute the n_type x n_token presence matrix
    pres_mat = np.empty([0, n_token])
    for type_i in type_list:
        pres_mat = np.append(pres_mat, [[token == type_i for token in token_list]], axis=0)

    # Compute the extended distance matrix
    token_mat = pres_mat.T.dot(type_mat.dot(pres_mat))

    # Return the distance_matrix
    return token_mat, token_list, existing_index_list


def similarity_to_dissimilarity(sim_mat, dist_option="max_minus"):
    """
    Compute the dissimilarity matrix from a similarity matrix with different options.

    :param sim_mat: the similarity matrix.
    :type sim_mat: numpy.ndarray
    :param dist_option: transformation parameter from similarity to dissimilarity, either "minus_log" or "max_minus".
    :type dist_option: str
    """

    # Warning if dist_option is not recognised
    if dist_option not in ["minus_log", "max_minus"]:
        warnings.warn("The parameter 'dist_option' is not recognise, setting it to 'minus_log'")
        dist_option = "minus_log"

    # Computation of dissimilarity matrix regarding dist_option
    if dist_option == "minus_log":
        if np.min(sim_mat) <= 0:
            sim_mat = sim_mat - np.min(sim_mat) + 1e-30
        d_mat = - np.log(sim_mat - np.min(sim_mat) + 1e-30)
    else:
        d_mat = np.max(sim_mat) - sim_mat

    #  Return the dissimilarity matrix
    return d_mat / np.max(d_mat)


def exchange_and_transition_matrices(n_token, exch_mat_opt, exch_range):
    """
    Compute the exchange matrix and the Markov chain transition matrix from given number of tokens

    :param n_token: the number of tokens
    :type n_token: int
    :param exch_mat_opt: option for the exchange matrix, "s" = standard, "u" = uniform, "d" = diffusive, "r" = ring
    :type exch_mat_opt: str
    :param exch_range: range of the exchange matrix
    :type exch_range: int
    :return: the n_token x n_token exchange matrix and the n_token x n_token markov transition matrix
    :rtype: (numpy.ndarray, numpy.ndarray)
    """

    # To manage other options
    if exch_mat_opt not in ["s", "u", "d", "r"]:
        warnings.warn("Exchange matrix option ('exch_mat_opt') not recognized, setting it to 's'")
        exch_mat_opt = "s"

    # Computation regarding options
    if exch_mat_opt == "s":
        exch_mat = np.abs(np.add.outer(np.arange(n_token), -np.arange(n_token))) <= exch_range
        np.fill_diagonal(exch_mat, 0)
        exch_mat = exch_mat / np.sum(exch_mat)
    elif exch_mat_opt == "u":
        f_vec = np.ones(n_token) / n_token
        adj_mat = np.abs(np.add.outer(np.arange(n_token), -np.arange(n_token))) <= exch_range
        np.fill_diagonal(adj_mat, 0)
        g_vec = np.sum(adj_mat, axis=1) / np.sum(adj_mat)
        k_vec = f_vec / g_vec
        b_mat = np.array([[min(v1, v2) for v2 in k_vec] for v1 in k_vec]) * adj_mat / np.sum(adj_mat)
        exch_mat = np.diag(f_vec) - np.diag(np.sum(b_mat, axis=1)) + b_mat
    elif exch_mat_opt == "d":
        f_vec = np.ones(n_token) / n_token
        adj_mat = np.abs(np.add.outer(np.arange(n_token), -np.arange(n_token))) <= 1
        np.fill_diagonal(adj_mat, 0)
        l_adj_mat = np.diag(np.sum(adj_mat, axis=1)) - adj_mat
        pi_outer_mat = np.outer(np.sqrt(f_vec), np.sqrt(f_vec))
        phi_mat = (l_adj_mat / pi_outer_mat) / np.trace(l_adj_mat)
        exch_mat = expm(- exch_range * phi_mat) * pi_outer_mat
    else:
        exch_mat = np.abs(np.add.outer(np.arange(n_token), -np.arange(n_token))) <= exch_range
        if exch_range == 1:
            np.fill_diagonal(exch_mat, 0)
        else:
            to_remove = np.abs(np.add.outer(np.arange(n_token), -np.arange(n_token))) <= (exch_range - 1)
            exch_mat = exch_mat ^ to_remove
        exch_mat = exch_mat / np.sum(exch_mat)

    w_mat = (exch_mat / np.sum(exch_mat, axis=1)).T

    # Return the transition matrix
    return exch_mat, w_mat


def token_clustering(d_ext_mat, exch_mat, w_mat, n_groups, alpha, beta, kappa, init_labels=None, known_labels=None,
                     conv_threshold=1e-5, n_hist=10, max_it=200, learning_rate_init=1, learning_rate_mult=0.9,
                     verbose=False):
    """
    Cluster tokens with cut soft clustering from a dissimilarity matrix, exchange matrix and transition matrix.
    Semi-supervised option available if init_labels is given.

    :param d_ext_mat: the n_token x n_token distance matrix
    :type d_ext_mat: numpy.ndarray
    :param exch_mat: the n_token x n_token exchange matrix
    :type exch_mat: numpy.ndarray
    :param w_mat: the n_token x n_token Markov chain transition matrix
    :type w_mat: numpy.ndarray
    :param n_groups: the number of groups
    :type n_groups: int
    :param alpha: alpha parameter
    :type alpha: float
    :param beta: beta parameter
    :type beta: float
    :param kappa: kappa parameter
    :type kappa: float
    :param init_labels: a vector containing initial labels. 0 = unknown class. (default = None)
    :type init_labels: numpy.ndarray
    :param known_labels: a vector containing fixed labels. 0 = unknown class. (default = None)
    :type known_labels: numpy.ndarray
    :param conv_threshold: convergence threshold (default = 1e-5)
    :type conv_threshold: float
    :param n_hist: number of past iterations which must be under threshold for considering convergence (default = 10)
    :type n_hist: int
    :param max_it: maximum iterations (default = 100)
    :type max_it: int
    :param learning_rate_init: initial value of the learning_rate parameter (>0, default = 1)
    :type learning_rate_init: float
    :param learning_rate_mult: multiplication coefficient for the learning_rate parameter (in ]0,1[, default = 0.9)
    :type learning_rate_mult: float
    :param verbose: turn on messages during computation (default = False)
    :type verbose: bool
    :return: the n_tokens x n_groups membership matrix for each token
    :rtype: numpy.ndarray
    """

    # Get the number of tokens
    n_token, _ = d_ext_mat.shape

    # Get the weights of tokens
    f_vec = np.sum(exch_mat, 0)

    # Initialization of Z
    # z_mat = np.random.random((n_token, n_groups))
    z_mat = np.abs(np.ones((n_token, n_groups)) + np.random.normal(0, 0.001, (n_token, n_groups)))
    z_mat = (z_mat.T / np.sum(z_mat, axis=1)).T

    # If known_labels is not None, pass them to init_labels
    if known_labels is not None:
        if init_labels is None:
            init_labels = known_labels
        else:
            if len(np.array(init_labels).shape) == 1:
                init_labels[known_labels > 1e-2] = known_labels[known_labels > 1e-2]
            else:
                init_labels[np.sum(known_labels, axis=1) > 1e-2, :] = \
                    known_labels[np.sum(known_labels, axis=1) > 1e-2, :]

    # Set true labels
    # If init_labels is not None, set known to value
    if init_labels is not None:
        if len(np.array(init_labels).shape) == 1:
            for i, label in enumerate(init_labels):
                if label != 0:
                    z_mat[i, :] = 0
                    z_mat[i, label - 1] = 1
        else:
            z_mat[np.sum(init_labels, axis=1) > 1e-2, :] = init_labels[np.sum(init_labels, axis=1) > 1e-2, :]

    # Control of the loop
    converge = False

    # Loop
    it = 0
    free_energy = 1e300
    learning_rate = learning_rate_init
    diff_memory = []
    while not converge:

        # Computation of rho_g vector
        rho_vec = np.sum(z_mat.T * f_vec, axis=1)

        # Computation of f_i^g matrix
        fig_mat = ((z_mat / rho_vec).T * f_vec).T

        # Computation of D_i^g matrix
        dig_mat = fig_mat.T.dot(d_ext_mat).T
        delta_g_vec = 0.5 * np.diag(dig_mat.T.dot(fig_mat))
        dig_mat = dig_mat - delta_g_vec

        # Computation of the e_gg vector and c_g vector
        e_gg = np.diag(z_mat.T.dot(exch_mat.dot(z_mat)))
        c_g = (rho_vec ** 2 - e_gg) / (rho_vec ** kappa)

        # Computation of FE standard way
        # free_energy_std = beta * np.sum(rho_vec * delta_g_vec) + alpha / 2 * np.sum(c_g) \
        #    + np.sum((z_mat.T * f_vec).T * np.log(z_mat / rho_vec))

        # Computation of H_ig
        hig_mat = beta * dig_mat + alpha * (rho_vec ** -kappa) * (rho_vec - w_mat.dot(z_mat)) \
                  - (0.5 * alpha * kappa * c_g / rho_vec)

        # Computation of the free energy and the new z_mat
        z_mat_new = rho_vec * np.exp(-hig_mat)
        z_mat_new[z_mat_new > 1e300] = 1e300
        zeta = np.sum(z_mat_new, axis=1)
        free_energy_new = 0.5 * alpha * (kappa - 1) * np.sum(c_g) - np.sum(np.log(zeta) * f_vec)
        z_mat_new = (z_mat_new.T / zeta).T
        z_mat_new = learning_rate * z_mat_new + (1 - learning_rate) * z_mat
        if learning_rate > 1:
            z_mat_new[z_mat_new < 0] = 0
            z_mat_new = (z_mat_new.T / np.sum(z_mat_new, axis=1)).T

        # If known_labels is not None, fix them again
        if known_labels is not None:
            if len(np.array(known_labels).shape) == 1:
                for i, label in enumerate(known_labels):
                    if label != 0:
                        z_mat_new[i, :] = 0
                        z_mat_new[i, label - 1] = 1
            else:
                z_mat_new[np.sum(init_labels, axis=1) > 1e-2, :] = known_labels[np.sum(known_labels, axis=1) > 1e-2, :]

        # Print diff and it
        # diff_pre_new = np.linalg.norm(z_mat - z_mat_new)
        diff_free_energy = (free_energy_new - free_energy) / learning_rate
        diff_memory.append(np.abs(diff_free_energy))
        if diff_free_energy > 0:
            learning_rate *= learning_rate_mult
        it += 1
        if verbose:
            print(f"Iteration {it}: FE = {free_energy_new}, Diff FE = {diff_free_energy}, "
                  f"learning_rate = {learning_rate}")

        # Verification of convergence
        if ((it >= n_hist) and (max(diff_memory[-n_hist:]) < conv_threshold)) or it > max_it:
            converge = True

        # Saving the new z_mat and new free_energy
        z_mat = z_mat_new
        free_energy = free_energy_new

    # Return the result
    return z_mat


def token_clustering_on_file(file_path, word_vector_path, dist_option, exch_mat_opt, exch_range, n_groups, alpha, beta,
                             kappa, block_size=1000, init_labels=None, known_labels=None, strong_pass=False,
                             conv_threshold=1e-5, n_hist=10, max_it=200, learning_rate_init=1, learning_rate_mult=0.9,
                             verbose=False):
    """
    Cluster tokens with cut soft clustering from any file. Uses the block_size to cut the text in smaller segments.
    Semi-supervised option available if init_labels is given.

    :param file_path: The path of the text file
    :type file_path: str
    :param word_vector_path: The path of a word vector model, in a gensim format
    :type word_vector_path: str
    :param dist_option: transformation parameter from similarity to dissimilarity, either "minus_log" or "max_minus".
    :type dist_option: str
    :param exch_mat_opt: option for the exchange matrix, "s" = standard, "u" = uniform, "d" = diffusive, "r" = ring
    :type exch_mat_opt: str
    :param exch_range: range of the exchange matrix
    :type exch_range: int
    :param n_groups: the number of groups
    :type n_groups: int
    :param alpha: alpha parameter
    :type alpha: float
    :param beta: beta parameter
    :type beta: float
    :param kappa: kappa parameter
    :type kappa: float
    :param block_size: The size of the blocks to slice the initial text
    :type block_size: int
    :param init_labels: a vector containing initial labels. 0 = unknown class. (default = None)
    :type init_labels: numpy.ndarray
    :param known_labels: a vector containing fixed labels. 0 = unknown class. (default = None)
    :type known_labels: numpy.ndarray
    :param strong_pass: the way labels are pass between blocks, if true, via known_labels. if false, via init_labels
    :param conv_threshold: convergence threshold (default = 1e-5)
    :type conv_threshold: float
    :param n_hist: number of past iterations which must be under threshold for considering convergence (default = 10)
    :type n_hist: int
    :param max_it: maximum iterations (default = 100)
    :type max_it: int
    :param learning_rate_init: initial value of the learning_rate parameter (>0, default = 1)
    :type learning_rate_init: float
    :param learning_rate_mult: multiplication coefficient for the learning_rate parameter (in ]0,1[, default = 0.9)
    :type learning_rate_mult: float
    :param verbose: turn on messages during computation (default = False)
    :type verbose: bool
    :return: the n_tokens x n_groups membership matrix for each token, the list of token found in the wv model,
             the boolean vector of found tokens
    :rtype: (numpy.ndarray, list[str], list[bool])
    """

    # Opening the file
    with open(file_path, "r") as text_file:
        text_string = text_file.read()
    # Split by tokens
    token_list = nltk.word_tokenize(text_string)
    # Vocabulary of text
    vocab_text = set(token_list)

    # Loading wordvector models
    wv_model = KeyedVectors.load(word_vector_path)
    # Vocabulary of word vectors
    vocab_wv = set(wv_model.vocab.keys())

    # The common vocabulary
    vocab_common = list(vocab_wv & vocab_text)
    # Reducing token_list to existing one
    existing_pos_list = [token in vocab_common for token in token_list]
    existing_token_list = np.array(token_list)[existing_pos_list]

    # Defining blocks indices
    n_token = len(existing_token_list)
    if block_size is None or block_size > n_token:
        n_split = 1
    else:
        n_split = round(n_token / block_size)
    n_block = n_split + (n_split - 1)
    real_split_size = int(n_token / n_split)
    range_list = []
    for i in range(n_block):
        if i == (n_block - 1):
            range_list.append(range(int(i * real_split_size / 2), n_token))
        else:
            range_list.append(range(int(i * real_split_size / 2), int((i / 2 + 1) * real_split_size)))

    # If known_labels is not None, pass them to init_labels
    if known_labels is not None:
        known_labels = known_labels[existing_pos_list]
        if init_labels is None:
            init_labels = known_labels
        else:
            init_labels = init_labels[existing_pos_list]
            if len(np.array(init_labels).shape) == 1:
                init_labels[known_labels > 1e-2] = known_labels[known_labels > 1e-2]
            else:
                init_labels[np.sum(known_labels, axis=1) > 1e-2, :] = \
                    known_labels[np.sum(known_labels, axis=1) > 1e-2, :]

    # Defining the initial matrix
    z_final = np.zeros((n_token, n_groups))
    if init_labels is not None:
        if len(np.array(init_labels).shape) == 1:
            for i, label in enumerate(init_labels):
                if label != 0:
                    z_final[i, :] = 0
                    z_final[i, label - 1] = 1
        else:
            z_final[np.sum(init_labels, axis=1) > 1e-2, :] = init_labels[np.sum(init_labels, axis=1) > 1e-2, :]

    # Loop on blocks
    for i in range(n_block):

        # Get block token and size
        block_token = existing_token_list[range_list[i]]
        n_block_token = len(block_token)

        # Build sim matrix
        sim_matrix = np.eye(n_block_token) / 2
        for ind_token_1 in range(n_block_token):
            for ind_token_2 in range(ind_token_1 + 1, n_block_token):
                sim_matrix[ind_token_1, ind_token_2] = \
                    wv_model.similarity(block_token[ind_token_1], block_token[ind_token_2])
        sim_matrix = sim_matrix + sim_matrix.T

        # Build dist matrix
        dist_matrix = similarity_to_dissimilarity(sim_matrix, dist_option=dist_option)

        # Compute the exchange and transition matrices
        exch_mat, w_mat = exchange_and_transition_matrices(n_block_token, exch_mat_opt=exch_mat_opt,
                                                           exch_range=exch_range)

        # Get the previous labels
        previous_labels = z_final[range_list[i], :]

        # Strong or weak pass
        if strong_pass:
            arg_dict = {"known_labels": previous_labels}
        else:
            arg_dict = {"init_labels": previous_labels}

        # Compute the membership matrix for the block
        z_block = token_clustering(d_ext_mat=dist_matrix, exch_mat=exch_mat, w_mat=w_mat, n_groups=n_groups,
                                   alpha=alpha, beta=beta, kappa=kappa, **arg_dict,
                                   conv_threshold=conv_threshold, n_hist=n_hist, max_it=max_it,
                                   learning_rate_init=learning_rate_init, learning_rate_mult=learning_rate_mult,
                                   verbose=verbose)

        # Put the z_block in z_final
        z_final[range_list[i], :] = z_block

    return z_final, existing_token_list, existing_pos_list


def seg_eval(algo_group_vec, real_group_vec):
    """
    A function computing the Pk and win_diff value for 2 segmentations. Also give random baselines
    :param algo_group_vec: The algorithm result in the form a token group memberships
    :type algo_group_vec: Union[list, numpy.ndarray]
    :param real_group_vec: The real group memberships of tokens
    :type real_group_vec: Union[list, numpy.ndarray]
    :return: Pk value, Win_diff value, Pk random value, Win_diff random value
    :rtype: (float, float, float, float)
    """

    # Transform into segmentation vectors
    real_segm_vec = convert_positions_to_masses(real_group_vec)
    algo_segm_vec = convert_positions_to_masses(algo_group_vec)

    # Make a shuffle group vec
    rdm_group_vec = real_group_vec.copy()
    rdm.shuffle(rdm_group_vec)
    rdm_segm_vec = convert_positions_to_masses(rdm_group_vec)

    # Compute the real value
    pk_res = pk(algo_segm_vec, real_segm_vec)
    try:
        win_diff = window_diff(algo_segm_vec, real_segm_vec)
    except:
        win_diff = 1

    # Compute the random value
    pk_rdm = pk(rdm_segm_vec, real_segm_vec)
    try:
        win_diff_rdm = window_diff(rdm_segm_vec, real_segm_vec)
    except:
        win_diff_rdm = 1

    # Return
    return pk_res, win_diff, pk_rdm, win_diff_rdm


def write_vector_in_html_file(output_file, token_list, vec, comment_line=None):
    """
    Write the token list in html file where colors correspond to value of the vector "vec".

    :param output_file: the name of the html outputted file
    :type output_file: str
    :param token_list: the list of tokens which define z_mat rows
    :type token_list: list[string]
    :param vec: the vector defining color of the tokens
    :type vec: numpy.ndarray
    :param comment_line: an optional comment line to start the file (default=None)
    :type comment_line: str
    """

    # Compute the vector of color
    color_vec = np.copy(vec)
    color_vec[color_vec > 0] = (color_vec[color_vec > 0] - np.min(color_vec[color_vec > 0])) \
                               / (np.max(vec) - np.min(color_vec[color_vec > 0])) * 255
    color_vec[color_vec < 0] = (color_vec[color_vec < 0] - np.max(color_vec[color_vec < 0])) \
                               / np.abs(np.min(vec) - np.max(color_vec[color_vec < 0])) * 255
    color_vec = np.intc(color_vec)

    # Write token in the html file
    with open(output_file, 'w') as html_file:
        html_file.write("<html>\n<head></head>\n")
        if comment_line is None:
            html_file.write("<body><p>")
        else:
            html_file.write(f"<body><p>{comment_line}</p> <p>")

        for i, token in enumerate(token_list):
            if color_vec[i] >= 0:
                html_file.write(f"<span style=\"background-color: "
                                f"rgb({255 - color_vec[i]},255,{255 - color_vec[i]})\">")
            else:
                html_file.write(f"<span style=\"background-color: "
                                f"rgb(255,{255 + color_vec[i]},{255 + color_vec[i]})\">")
            html_file.write(token + " </span>")
        html_file.write("</p></body>\n</html>")

    # Return 0 is all went well
    return 0


def write_groups_in_html_file(output_file, token_list, z_mat, comment_line=None):
    """
    Write the html group color file from an input file and a membership matrix "z_mat".

    :param output_file: the name of the html outputted file
    :type output_file: str
    :param token_list: the list of tokens which define z_mat rows
    :type token_list: list[string]
    :param z_mat: the fuzzy membership matrix Z
    :type z_mat: numpy.ndarray
    :param comment_line: an optional comment line to start the file (default=None)
    :type comment_line: str
    """

    # Getting the number of groups
    _, n_groups = z_mat.shape

    # Creating group colors
    color_rgb_list = []
    for i in range(n_groups):
        color_rgb_list.append(np.array(colorsys.hsv_to_rgb(i * 1 / n_groups, 1, 1)))
    color_rgb_mat = np.array(color_rgb_list)

    # Creating tokens color
    token_color_mat = np.array(255 * z_mat.dot(color_rgb_mat), int)

    # Creating html file
    with open(output_file, 'w') as html_file:

        html_file.write("<html>\n<head></head>\n")

        # Writing comment is present
        if comment_line is None:
            html_file.write("<body><p>")
        else:
            html_file.write(f"<body><p>{comment_line}</p> <p>")

        # Writing tokens with colors
        for i in range(len(token_list)):
            html_file.write(f"<span style=\"background-color: "
                            f"rgb({token_color_mat[i, 0]},{token_color_mat[i, 1]},{token_color_mat[i, 2]})\">")
            html_file.write(token_list[i] + " </span>")

        html_file.write("</p></body>\n</html>")

    # Return 0 is all went well
    return 0


def write_membership_mat_in_csv_file(output_file, z_token_list, z_mat, comment_line=None):
    """
    Write the csv file containing the membership matrix and token list

    :param output_file: the name of the csv outputted file
    :type output_file: str
    :param z_token_list: the list of tokens which define z_mat rows
    :type z_token_list: list[string]
    :param z_mat: the fuzzy membership matrix Z
    :type z_mat: numpy.ndarray
    :param comment_line: an optional comment line to start the file (default=None)
    :type comment_line: str
    """

    # Getting the number of groups
    _, n_groups = z_mat.shape

    # Creating csv file
    with open(output_file, 'w') as text_file:

        # Write comment_line if exists
        if comment_line is not None:
            text_file.write(comment_line + "," * (n_groups + 1) + "\n")

        # Write head
        text_file.write("token,id")
        for i in range(n_groups):
            text_file.write(f",G{i + 1}")
        text_file.write("\n")

        # Write matrix
        for i, token in enumerate(z_token_list):
            text_file.write(f"{token},{i + 1}")
            for j in range(n_groups):
                text_file.write(f",{z_mat[i, j]:.20f}")
            text_file.write("\n")

    # Return 0 is all went well
    return 0


def autocorrelation_index(d_ext_mat, exch_mat, w_mat):
    """
    Compute the autocorrelation index regarding a dissimilarity matrix and a exchange matrix

    :param d_ext_mat: the (n_token x  n_token) dissimilarity matrix between tokens
    :type d_ext_mat: numpy.ndarray
    :param exch_mat: the (n_token x n_token) exchange matrix between tokens
    :type exch_mat: numpy.ndarray
    :param w_mat: a (n_token x n_token) transition matrix.
    :type w_mat: numpy.ndarray
    :return: the autocorrelation index, the theoretical mean and the theoretical variance
    :rtype: (float, float, float)
    """
    # Get the weights of tokens
    f_vec = np.sum(exch_mat, 0)
    # Get the number of token
    n_token = len(f_vec)

    # Compute of the global inertia
    global_inertia = 0.5 * np.sum(np.outer(f_vec, f_vec) * d_ext_mat)
    # Compute the local inertia
    local_inertia = 0.5 * np.sum(exch_mat * d_ext_mat)
    # Compute the autocorrelation index
    autocor_index = (global_inertia - local_inertia) / global_inertia

    # Compute the theoretical expected value
    trace_w_mat = np.trace(w_mat)
    theoretical_mean = (trace_w_mat - 1) / (n_token - 1)
    # Compute the theoretical
    theoretical_var = 2 * (np.trace(w_mat @ w_mat) - 1 - (trace_w_mat - 1) ** 2 / (n_token - 1)) \
                      / (n_token ** 2 - 1)

    # Return autocorrelation index, theoretical mean and theoretical variance
    return autocor_index, theoretical_mean, theoretical_var


def lisa_computation(d_ext_mat, exch_mat, w_mat):
    """
    From an (n_token x n_token) extended dissimilarity matrix, an exchange matrix and a transition matrix,
    compute the length (n_token) lisa vector where local autocorrelation for each token is stored.

    :param d_ext_mat: an (n_token x n_token) extended dissimilarity matrix.
    :type d_ext_mat: numpy.ndarray
    :param exch_mat: an (n_token x n_token) exchange matrix.
    :type exch_mat: numpy.ndarray
    :param w_mat: a (n_token x n_token) transition matrix.
    :type w_mat: numpy.ndarray
    :return: the (n_token) lisa vector containing local autocorrelation for each token.
    :rtype: numpy.ndarray
    """

    # Get the number of tokens
    n_token, _ = d_ext_mat.shape
    # Get the weights of tokens
    f_vec = np.sum(exch_mat, 0)

    # Compute the centring matrix
    h_mat = np.identity(n_token) - np.outer(np.ones(n_token), f_vec)
    # Compute the scalar produced matrix
    b_mat = - 0.5 * h_mat.dot(d_ext_mat.dot(h_mat.T))
    # Compute of the global inertia
    global_inertia = 0.5 * np.sum(np.outer(f_vec, f_vec) * d_ext_mat)
    # Compute lisa vector
    lisa_vec = np.diag(w_mat.dot(b_mat)) / global_inertia

    # Return the result
    return lisa_vec
