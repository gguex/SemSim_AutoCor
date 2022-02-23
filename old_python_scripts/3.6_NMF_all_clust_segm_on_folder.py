import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from local_functions import seg_eval
from tqdm import tqdm

# -------------------------------------
# --- Parameters
# -------------------------------------

# Input folder
input_text_folder = "corpora/manifesto_pp"
# Take stopwords
stop_words = False
# Output file name
output_file_name = "results/NMF_clust_all_manifesto.csv"

# Fixed number of groups (if none, extracted from data)
fixed_n_groups = None

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

# Get all document text
text_list, token_list_list, real_group_vec_list, all_group_vec = [], [], [], []
for input_text_file, input_group_file in zip(input_text_file_list, input_group_file_list):
    # Get real groups
    with open(input_group_file) as ground_truth:
        real_group_vec = ground_truth.read()
        real_group_vec = np.array([int(element) for element in real_group_vec.split(",")])
    # Save it
    real_group_vec_list.append(real_group_vec)
    all_group_vec.extend(real_group_vec)

    # Get text
    with open(input_text_file) as text_file:
        text = text_file.read()
        token_list = text.split()
    text_list.append(text)
    token_list_list.append(token_list)

# Get the total number of groups
if fixed_n_groups is None:
    n_groups = len(set(all_group_vec))
else:
    n_groups = fixed_n_groups

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

# Make results file
with open(output_file_name, "w") as output_file:
    output_file.write("input_file,n_groups,nmi,pk,pk_rdm,wd,wd_rdm\n")

# Loop on doc
for doc_id, token_list in tqdm(enumerate(token_list_list)):

    # words x documents probabilities
    word_likelihood = (norm_word_array * np.outer(norm_document_array[doc_id, :], np.ones(norm_word_array.shape[1]))).T
    word_groups = np.argmax(word_likelihood, 1) + 1

    # Contruct the algo_group_vec
    algo_group_vec = []
    actual_g = 1
    for w in token_list:
        if len(np.where(np.array(model_voc) == w)[0]) > 0:
            actual_g = word_groups[np.where(np.array(model_voc) == w)[0][0]]
        algo_group_vec.append(actual_g)

    # Get real group vec
    real_group_vec = real_group_vec_list[doc_id]

    # NMI
    nmi = normalized_mutual_info_score(real_group_vec, algo_group_vec)
    # Segmentation evaluation
    pk, wd, pk_rdm, wd_rdm = seg_eval(algo_group_vec, real_group_vec)

    # Writing results
    with open(output_file_name, "a") as output_file:
        output_file.write(f"{input_text_file_list[doc_id]},{len(set(real_group_vec))},{nmi},"
                          f"{pk},{pk_rdm},{wd},{wd_rdm}\n")

