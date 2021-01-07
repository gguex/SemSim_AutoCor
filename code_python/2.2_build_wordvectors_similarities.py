import nltk
from gensim.models import KeyedVectors
from tqdm import tqdm
from code_python.local_functions import get_all_paths

# -------------------------------------
# --- Parameters
# -------------------------------------

# Path of the text file with only nouns, verbs, adjectives or adverbs
input_file = "The_WW_of_Oz_pp.txt"

# Name of the outputted tag for the similarity ("w2v" or "glv")
sim_tag = "glv"

# -------------------------------------
# --- Computations
# -------------------------------------

# Path of the Word vector model
if sim_tag == "w2v":
    wv_model_path = "/home/gguex/Documents/data/pretrained_word_vectors/enwiki.model"
else:
    wv_model_path = "/home/gguex/Documents/data/pretrained_word_vectors/glove42B300d.model"


# --- Defining paths --- #

# Getting all paths
file_path, type_freq_file_path, sim_matrix_file_path, _ = get_all_paths(input_file, sim_tag, warn=False)

# --- Get token, type and freq --- #

# Opening the file
with open(file_path, "r") as text_file:
    text_string = text_file.read()

# Split by tokens
token_list = nltk.word_tokenize(text_string)

# Get type list and frequencies
type_freq_dict = nltk.FreqDist(token_list)
vocab_text = set(type_freq_dict.keys())

# --- Load the gensim model, check common vocabulary and write files ---#

# load gensim model
wv_model = KeyedVectors.load(wv_model_path)

# build vocabulary
vocab_wv = set(wv_model.vocab.keys())

# Find the common vocabulary
vocab_common = list(vocab_wv & vocab_text)

# Write the two files
with open(type_freq_file_path, "w") as type_freq_file, open(sim_matrix_file_path, "w") as sim_matrix_file:
    for type_1 in tqdm(vocab_common):
        type_freq_file.write(type_1 + ";" + str(type_freq_dict[type_1]) + "\n")
        for type_2 in vocab_common:
            sim_matrix_file.write(str(wv_model.similarity(type_1, type_2)))
            if type_2 != vocab_common[len(vocab_common) - 1]:
                sim_matrix_file.write(";")
            else:
                sim_matrix_file.write("\n")
