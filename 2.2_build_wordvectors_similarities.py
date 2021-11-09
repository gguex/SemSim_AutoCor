from gensim.models import KeyedVectors
from local_functions import build_wv_similarity_matrix
from os.path import expanduser

# -------------------------------------
# --- Parameters
# -------------------------------------

# List of paths for text files to compute similarity
input_file_list = ["61320_199211_pp.txt",
                   "61320_200411_pp.txt",
                   "61320_201211_pp.txt",
                   "61320_201611_pp.txt",
                   "61320_202011_pp.txt",
                   "61620_200411_pp.txt",
                   "61620_200811_pp.txt",
                   "61620_201211_pp.txt",
                   "61620_201611_pp.txt",
                   "61620_202011_pp.txt",
                   "61320_199211_pp_wostw.txt",
                   "61320_200411_pp_wostw.txt",
                   "61320_201211_pp_wostw.txt",
                   "61320_201611_pp_wostw.txt",
                   "61320_202011_pp_wostw.txt",
                   "61620_200411_pp_wostw.txt",
                   "61620_200811_pp_wostw.txt",
                   "61620_201211_pp_wostw.txt",
                   "61620_201611_pp_wostw.txt",
                   "61620_202011_pp_wostw.txt"]
# List of tags to enumerate similarity to compute
sim_tag_list = ["w2v", "glv", "ftx"]

# -------------------------------------
# --- Computations
# -------------------------------------

# Loading wordvector models
home = expanduser("~")
w2v_model = KeyedVectors.load(f"{home}/Documents/data/pretrained_word_vectors/enwiki.model")
glv_model = KeyedVectors.load(f"{home}/Documents/data/pretrained_word_vectors/glove42B300d.model")
ftx_model = KeyedVectors.load(f"{home}/Documents/data/pretrained_word_vectors/en_fasttext.model")

# Loop on files and tags
for input_file in input_file_list:
    for sim_tag in sim_tag_list:
        if sim_tag == "w2v":
            build_wv_similarity_matrix(f"corpora/{input_file}", f"similarity_matrices/{input_file[:-4]}_{sim_tag}.csv",
                                       w2v_model)
        elif sim_tag == "glv":
            build_wv_similarity_matrix(f"corpora/{input_file}", f"similarity_matrices/{input_file[:-4]}_{sim_tag}.csv",
                                       glv_model)
        elif sim_tag == "ftx":
            build_wv_similarity_matrix(f"corpora/{input_file}", f"similarity_matrices/{input_file[:-4]}_{sim_tag}.csv",
                                       ftx_model)
