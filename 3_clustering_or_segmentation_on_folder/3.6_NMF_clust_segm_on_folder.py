import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from tqdm import tqdm
import os
from local_functions import seg_eval
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------------------
# --- Parameters
# -------------------------------------

# Input folder
input_text_folder = "corpora/elements_pp"
# Take stopwords
stop_words = False
# Output file name
output_file_name = "../results/3.1_clust_results/NMF_clust_elements.csv"

# Fixed number of groups (if none, extracted from data)
fixed_n_groups = None

# Hyperparameters
chunk_size = 156

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

# Make results file
with open(output_file_name, "w") as output_file:
    output_file.write("input_file,n_groups,chunk_size,"
                      "nmi,pk,pk_rdm,wd,wd_rdm\n")

for index_file in tqdm(range(len(input_text_file_list))):

    # Get text file associated files
    input_text_file = input_text_file_list[index_file]
    input_group_file = input_group_file_list[index_file]

    # Get real groups
    with open(input_group_file) as ground_truth:
        real_group_vec = ground_truth.read()
        real_group_vec = np.array([int(element) for element in real_group_vec.split(",")])
    if fixed_n_groups is None:
        n_groups = len(set(real_group_vec))
    else:
        n_groups = fixed_n_groups

    # Get tokens
    with open(input_text_file) as text_file:
        text = text_file.read()
        token_list = text.split()

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
        word_likelihood = (norm_word_array * np.outer(norm_document_array[chunk_id, :],
                                                      np.ones(norm_word_array.shape[1]))).T
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
    with open(output_file_name, "a") as output_file:
        output_file.write(f"{input_text_file},{n_groups},{chunk_size},{nmi},{pk},{pk_rdm},{wd},{wd_rdm}\n")
