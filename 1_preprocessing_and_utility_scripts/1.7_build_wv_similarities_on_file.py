from gensim.models import KeyedVectors
from local_functions import build_wv_similarity_matrix
from os.path import expanduser

# -------------------------------------
# --- Parameters
# -------------------------------------

# Input folder
input_file = "corpora/cities_pp/3_pp_.txt"

# List of tags to enumerate similarity to compute
sim_tag_list = ["w2v", "glv", "ftx"]

# -------------------------------------
# --- Computation
# -------------------------------------

# Short name
short_file_name = input_file.split("/")[-1:][0][:-4]

# Loading wordvector models
home = expanduser("~")
w2v_model = KeyedVectors.load(f"{home}/Documents/data/pretrained_word_vectors/enwiki.model")
glv_model = KeyedVectors.load(f"{home}/Documents/data/pretrained_word_vectors/glove42B300d.model")
ftx_model = KeyedVectors.load(f"{home}/Documents/data/pretrained_word_vectors/en_fasttext.model")

# Loop on sim tag
for sim_tag in sim_tag_list:
    if sim_tag == "w2v":
        build_wv_similarity_matrix(input_file,
                                   f"similarity_matrices/{short_file_name}_{sim_tag}.csv", w2v_model)
    elif sim_tag == "glv":
        build_wv_similarity_matrix(input_file,
                                   f"similarity_matrices/{short_file_name}_{sim_tag}.csv", glv_model)
    elif sim_tag == "ftx":
        build_wv_similarity_matrix(input_file,
                                   f"similarity_matrices/{short_file_name}_{sim_tag}.csv", ftx_model)
