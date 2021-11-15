import os
from gensim.models import KeyedVectors
from local_functions import build_wv_similarity_matrix
from os.path import expanduser

# -------------------------------------
# --- Parameters
# -------------------------------------

# Input folder
input_text_folder = "corpora/manifesto_pp"

# Make with or without stopwords
stop_words = False

# List of tags to enumerate similarity to compute
sim_tag_list = ["w2v", "glv", "ftx"]

# -------------------------------------
# --- Computation
# -------------------------------------

# List files in the corpus folder
file_list = os.listdir(input_text_folder)

# Sort the list
file_list.sort()

# Restrict them to those with or without stopwords
input_file_list = [file for file in file_list if (("wostw" in file) ^ stop_words) & ("groups" not in file)]

# Loading wordvector models
home = expanduser("~")
w2v_model = KeyedVectors.load(f"{home}/Documents/data/pretrained_word_vectors/enwiki.model")
glv_model = KeyedVectors.load(f"{home}/Documents/data/pretrained_word_vectors/glove42B300d.model")
ftx_model = KeyedVectors.load(f"{home}/Documents/data/pretrained_word_vectors/en_fasttext.model")

# Loop on files and tags
for input_file in input_file_list:
    for sim_tag in sim_tag_list:
        if sim_tag == "w2v":
            build_wv_similarity_matrix(f"{input_text_folder}/{input_file}",
                                       f"similarity_matrices/{input_file[:-4]}_{sim_tag}.csv", w2v_model)
        elif sim_tag == "glv":
            build_wv_similarity_matrix(f"{input_text_folder}/{input_file}",
                                       f"similarity_matrices/{input_file[:-4]}_{sim_tag}.csv", glv_model)
        elif sim_tag == "ftx":
            build_wv_similarity_matrix(f"{input_text_folder}/{input_file}",
                                       f"similarity_matrices/{input_file[:-4]}_{sim_tag}.csv", ftx_model)
