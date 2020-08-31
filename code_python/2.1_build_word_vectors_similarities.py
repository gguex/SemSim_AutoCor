import nltk
from gensim.models import KeyedVectors
import os
from tqdm import tqdm

# --- Parameters --- #

# Path of the text file with only nouns, verbs, adjectives or adverbs
input_file = "Animal_farm_nouns.txt"

# Path of the Word vector model (absolute path, not in the project directory)
wv_model_path = "/home/gguex/Documents/data/pretrained_word_vectors/enwiki.model"

# Name of the outputted tag for the similarity
sim_tag = "wesim"

# --- Defining paths --- #

# Getting the base path (must run the script from a folder inside the "SemSim_Autocor" folder)
working_path = os.getcwd()
base_path = str.split(working_path, "SemSim_AutoCor")[0] + "SemSim_AutoCor/"

# Path of the inputted file
file_path = base_path + "corpora/" + input_file
# Path of the outputted present types and frequencies
type_freq_file_path = base_path + "similarities_frequencies/" + input_file[:-4] + "_" + sim_tag + "_typefreq.txt"
# Path of the outputted similarity matrix
sim_matrix_file_path = base_path + "similarities_frequencies/" + input_file[:-4] + \
                       "_" + sim_tag + "_similarities.txt"


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
wv_wiki = KeyedVectors.load(wv_model_path)

# build vocabulary
vocab_wiki = set(wv_wiki.vocab.keys())

# Find the common vocabulary
vocab_common = list(vocab_wiki & vocab_text)

# Write the two files
with open(type_freq_file_path, "w") as type_freq_file, open(sim_matrix_file_path, "w") as sim_matrix_file:
    for type_1 in tqdm(vocab_common):
        type_freq_file.write(type_1 + ";" + str(type_freq_dict[type_1]) + "\n")
        for type_2 in vocab_common:
            sim_matrix_file.write(str(wv_wiki.similarity(type_1, type_2)))
            if type_2 != vocab_common[len(vocab_common) - 1]:
                sim_matrix_file.write(";")
            else:
                sim_matrix_file.write("\n")
