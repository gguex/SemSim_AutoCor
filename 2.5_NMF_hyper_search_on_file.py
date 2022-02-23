from local_functions import seg_eval
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
import os
from tqdm import tqdm
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------------------
# --- Parameters
# -------------------------------------

base_path = os.getcwd()

# ------------ Options

input_text_file = "corpora/elements_pp/e10_pp_wostw.txt"
input_group_file = "corpora/elements_pp/e10_pp_wostw_groups.txt"

results_file_name = "results/2_hyperparam_search/NMF_Gn_e10_pp_wostw.csv"

# ---

# Number groups (if None, extracted from data)
n_groups = None

# Search : either chunk_size_vec directly (put None to use "chunk_vec_sep")
chunk_size_vec = None
# either split length(token) into approx equal parts
chunk_vec_sep = 20

# -------------------------------------
# --- Computations
# -------------------------------------

# Make results file
with open(results_file_name, "w") as output_file:
    output_file.write("input_file,n_groups,chunk_size,"
                      "mean_nmi,mean_pk,mean_rdm_pk,mean_wd,mean_rdm_wd\n")

# Get real groups
with open(input_group_file) as ground_truth:
    real_group_vec = ground_truth.read()
    real_group_vec = np.array([int(element) for element in real_group_vec.split(",")])
if n_groups is None:
    n_groups = len(set(real_group_vec))

# Get tokens
with open(input_text_file) as text_file:
    text = text_file.read()
    token_list = text.split()

# Define chunk exploration space
if chunk_size_vec is None:
    step_chunk = int(np.ceil(len(token_list) / chunk_vec_sep))
    chunk_size_vec = list(range(step_chunk, step_chunk*chunk_vec_sep + 1, step_chunk))

# -- Loop on chunk_size
for chunk_size in tqdm(chunk_size_vec):

    # Divide the corpus by chunk
    n_chunk = int(np.ceil(len(token_list) / chunk_size))
    text_list, token_list_list = [], []
    for i in range(n_chunk):
        token_chunk_list = token_list[i * chunk_size:(i + 1) * chunk_size]
        token_list_list.append(token_chunk_list)
        text_list.append(" ".join(token_chunk_list))
    if n_chunk == 1:
        chunk_size = len(token_list)

    # Build Tf-idf matrix
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text_list)

    # Build NMF model
    model = NMF(n_components=n_groups)
    model.fit(X)

    # Get model voc
    model_voc = vectorizer.get_feature_names()

    # documents x groups array
    document_array = model.transform(X)
    norm_document_array = (document_array.T / document_array.sum(axis=1)).T

    # words x groups array
    word_array = model.components_
    norm_word_array = (word_array.T / word_array.sum(axis=1)).T

    # Loop on chunk
    algo_group_vec = []
    for chunk_id, token_chunk_list in enumerate(token_list_list):

        # words x documents probabilities
        word_likelihood = (norm_word_array * np.outer(norm_document_array[chunk_id, :], np.ones(norm_word_array.shape[1]))).T
        word_groups = np.argmax(word_likelihood, 1) + 1

        # Contruct the algo_group_vec
        algo_chunk_group_vec = []
        actual_g = 1
        for w in token_chunk_list:
            if len(np.where(np.array(model_voc) == w)[0]) > 0:
                actual_g = word_groups[np.where(np.array(model_voc) == w)[0][0]]
            algo_chunk_group_vec.append(actual_g)

        algo_group_vec.extend(algo_chunk_group_vec)

    # NMI
    nmi = normalized_mutual_info_score(real_group_vec, algo_group_vec)
    # Segmentation evaluation
    pk, wd, pk_rdm, wd_rdm = seg_eval(algo_group_vec, real_group_vec)

    # Writing results
    with open(results_file_name, "a") as output_file:
        output_file.write(f"{input_text_file},{n_groups},{chunk_size},{nmi},"
                          f"{pk},{pk_rdm},{wd},{wd_rdm}\n")
