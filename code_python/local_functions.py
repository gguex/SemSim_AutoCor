import os
import nltk
import colorsys
from scipy.linalg import expm
import warnings
import numpy as np


def get_all_paths(input_file, sim_tag, working_path=os.getcwd()):
    """
    A function specific to the Semsim_Autocor project that returns all the useful paths
    given a input file name and a similarity tag. If any file is missing, raise a warning and return an empty string
    (the missing of ground truth file don't raise a warning, as it is not always needed)

    :param input_file: the name of the text file
    :type input_file: str
    :param sim_tag: the similarity tag
    :type sim_tag: str
    :param working_path: the initial path of script, must anywhere in the SemSim_AutoCor project
    :type working_path: str
    :return: paths of : the corpus file, the typefreq file, the similarity file and the ground truth file.
    :rtype: (str, str, str, str)
    """

    # Getting the SemSim_AutoCor folder, if above
    base_path = str.split(working_path, "SemSim_AutoCor")[0] + "SemSim_AutoCor"

    # Path of the text file
    text_file_path = f"{base_path}/corpora/{input_file}"
    # Path of the types and frequencies file
    typefreq_file_path = f"{base_path}/similarities_frequencies/{input_file[:-4]}_{sim_tag}_typefreq.txt"
    # Path of the similarity matrix
    sim_file_path = f"{base_path}/similarities_frequencies/{input_file[:-4]}_{sim_tag}_similarities.txt"
    # Path of the ground true file
    ground_truth_path = f"{base_path}/corpora/mixgroup_{input_file[4:]}"

    if not os.path.exists(text_file_path):
        warnings.warn(f"The corpus file '{text_file_path}' is missing")
        text_file_path = ""
    if not os.path.exists(typefreq_file_path):
        warnings.warn(f"The typefreq file '{typefreq_file_path}' is missing")
        typefreq_file_path = ""
    if not os.path.exists(sim_file_path):
        warnings.warn(f"The similarity file '{sim_file_path}' is missing")
        sim_file_path = ""
    if not os.path.exists(ground_truth_path):
        ground_truth_path = ""

    # Return paths
    return text_file_path, typefreq_file_path, sim_file_path, ground_truth_path


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
    :return: the (n_token x n_token) extended matrix between text tokens and the (n_token) list of tokens used.
    :rtype: (numpy.ndarray, list[str])
    """

    # Import the text file and remove non-existing token
    with open(text_file_path, "r") as text_file:
        text_string = text_file.read()
    raw_token_list = nltk.word_tokenize(text_string)

    # Keep only the tokens which are present in the type_list
    token_list = [token for token in raw_token_list if token in type_list]
    n_token = len(token_list)

    # Compute the n_type x n_token presence matrix
    pres_mat = np.empty([0, n_token])
    for type_i in type_list:
        pres_mat = np.append(pres_mat, [[token == type_i for token in token_list]], axis=0)

    # Compute the extended distance matrix
    token_mat = pres_mat.T.dot(type_mat.dot(pres_mat))

    # Return the distance_matrix
    return token_mat, token_list


def similarity_to_dissimilarity(sim_mat, dist_option="minus_log"):
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
    elif dist_option == "max_minus":
        d_mat = np.max(sim_mat) - sim_mat

    #  Return the dissimilarity matrix
    return d_mat


def exchange_and_transition_matrices(n_token, exch_mat_opt, exch_range):
    """
    Compute the exchange matrix and the Markov chain transition matrix from given number of tokens

    :param n_token: the number of tokens
    :type n_token: int
    :param exch_mat_opt: option for the exchange matrix, "s" = standard, "u" = uniform, "d" = diffusive
    :type exch_mat_opt: str
    :param exch_range: range of the exchange matrix
    :type exch_range: int
    :return: the n_token x n_token exchange matrix and the n_token x n_token markov transition matrix
    :rtype: (numpy.ndarray, numpy.ndarray)
    """

    # To manage other options
    if exch_mat_opt not in ["s", "u", "d"]:
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
    else:
        f_vec = np.ones(n_token) / n_token
        adj_mat = np.abs(np.add.outer(np.arange(n_token), -np.arange(n_token))) <= 1
        np.fill_diagonal(adj_mat, 0)
        l_adj_mat = np.diag(np.sum(adj_mat, axis=1)) - adj_mat
        pi_outer_mat = np.outer(np.sqrt(f_vec), np.sqrt(f_vec))
        phi_mat = (l_adj_mat / pi_outer_mat) / np.trace(l_adj_mat)
        exch_mat = expm(- exch_range * phi_mat) * pi_outer_mat

    w_mat = (exch_mat / np.sum(exch_mat, axis=1)).T

    # Return the transition matrix
    return exch_mat, w_mat


def autocorrelation_index(d_ext_mat, exch_mat):
    """
    Compute the autocorrelation index regarding a dissimilarity matrix and a exchange matrix

    :param d_ext_mat: the (n_token x  n_token) dissimilarity matrix between tokens
    :type d_ext_mat: numpy.ndarray
    :param exch_mat: the (n_token x n_token) exchange matrix between tokens
    :type exch_mat: numpy.ndarray
    :return: the autocorrelation index
    :rtype: float
    """
    # Get the weights of tokens
    f_vec = np.sum(exch_mat, 0)

    # Compute the local inertia
    local_inertia = 0.5 * np.sum(exch_mat * d_ext_mat)
    # Compute of the global inertia
    global_inertia = 0.5 * np.sum(np.outer(f_vec, f_vec) * d_ext_mat)
    # Compute the autocorrelation index
    autocorrelation_index = (global_inertia - local_inertia) / global_inertia

    # Return autocorrelation index
    return autocorrelation_index


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


def discontinuity_segmentation(d_ext_mat, exch_mat, w_mat, n_groups, alpha, beta, kappa,
                               conv_threshold=1e-5, max_it=100, init_labels=None):
    """
    Cluster tokens with discontinuity segmentation from a dissimilarity matrix, exchange matrix and transition matrix.
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
    :param conv_threshold: convergence threshold (default = 1e-5)
    :type conv_threshold: float
    :param max_it: maximum iterations (default = 100)
    :type max_it: int
    :param init_labels: a vector containing initial labels. 0 = unknown class. (default = None)
    :type init_labels: numpy.ndarray
    :return: the n_tokens x n_groups membership matrix for each token
    :rtype: numpy.ndarray
    """

    # Get the number of tokens
    n_token, _ = d_ext_mat.shape

    # Get the weights of tokens
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
        print(f"Iteration {it}: {diff_pre_new}")

        # Verification of convergence
        if diff_pre_new < conv_threshold:
            converge = True
        if it > max_it:
            converge = True

        # Saving the new z_mat
        z_mat = z_new_mat

    # Return the result
    return z_mat


def cut_segmentation(d_ext_mat, exch_mat, w_mat, n_groups, gamma, beta, kappa,
                     conv_threshold=1e-5, max_it=100, init_labels=None):
    """
    Cluster tokens with cut segmentation from a dissimilarity matrix, exchange matrix and transition matrix.
    Semi-supervised option available if init_labels is given.

    :param d_ext_mat: the n_token x n_token distance matrix
    :type d_ext_mat: numpy.ndarray
    :param exch_mat: the n_token x n_token exchange matrix
    :type exch_mat: numpy.ndarray
    :param w_mat: the n_token x n_token Markov chain transition matrix
    :type w_mat: numpy.ndarray
    :param n_groups: the number of groups
    :type n_groups: int
    :param gamma: gamma parameter
    :type gamma: float
    :param beta: beta parameter
    :type beta: float
    :param kappa: kappa parameter
    :type kappa: float
    :param conv_threshold: convergence threshold (default = 1e-5)
    :type conv_threshold: float
    :param max_it: maximum iterations (default = 100)
    :type max_it: int
    :param init_labels: a vector containing initial labels. 0 = unknown class. (default = None)
    :type init_labels: numpy.ndarray
    :return: the n_tokens x n_groups membership matrix for each token
    :rtype: numpy.ndarray
    """

    # Get the number of tokens
    n_token, _ = d_ext_mat.shape

    # Get the weights of tokens
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

        # Computation of the e_gg vector
        e_gg = np.diag(z_mat.T.dot(exch_mat.dot(z_mat)))

        # Computation of H_ig
        hig_mat = beta * dig_mat + gamma * (rho_vec ** -kappa) * (rho_vec - w_mat.dot(z_mat)) \
            - (0.5 * gamma * kappa * (rho_vec ** (-kappa - 1)) * (rho_vec ** 2 - e_gg))

        # Computation of the new z_mat
        if np.sum(-hig_mat > 690) > 0:
            warnings.warn("Overflow of exp(-hig_mat)")
            hig_mat[-hig_mat > 690] = -690
        z_new_mat = rho_vec * np.exp(-hig_mat)
        z_new_mat = (z_new_mat.T / np.sum(z_new_mat, axis=1)).T

        # Print diff and it
        diff_pre_new = np.linalg.norm(z_mat - z_new_mat)
        it += 1
        print(f"Iteration {it}: {diff_pre_new}")

        # Verification of convergence
        if diff_pre_new < conv_threshold:
            converge = True
        if it > max_it:
            converge = True

        # Saving the new z_mat
        z_mat = z_new_mat

    # Return the result
    return z_mat


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
                text_file.write(f",{z_mat[i, j]}")
            text_file.write("\n")

    # Return 0 is all went well
    return 0
