import nltk
from gensim.models import KeyedVectors

# --- Parameters --- #

# Path of the text file with only nouns
noun_only_file_path = "/home/gguex/Documents/data/corpora/The_Wonderful_Wizard_of_Oz_noun_only.txt"
# Path of the outputted present types and frequencies
type_freq_file_path = \
    "/home/gguex/Documents/recherche/SemSim_AutoCor-master/similarities_frequencies/Wizard_of_Oz_typefreq.txt"
# Path of the outputted similarity matrix
sim_matrix_file_path = \
    "/home/gguex/Documents/recherche/SemSim_AutoCor-master/similarities_frequencies/Wizard_of_Oz_similarities.txt"

# --- Get token, type and freq --- #

# Opening the file
with open(noun_only_file_path, "r") as text_file:
    text_string = text_file.read()

# Split by tokens
token_list = nltk.word_tokenize(text_string)

# Get type list and frequencies
type_freq_dict = nltk.FreqDist(token_list)
vocab_text = set(type_freq_dict.keys())

# --- Load the gensim model, check common vocabulary and write files ---#

# load gensim model
wv_wiki = KeyedVectors.load("/home/gguex/Documents/data/pretrained_word_vectors/enwiki.model")

# build vocabulary
vocab_wiki = set(wv_wiki.vocab.keys())

# Find the common vocabulary
vocab_common = list(vocab_wiki & vocab_text)

# Write the two files
with open(type_freq_file_path, "w") as type_freq_file, open(sim_matrix_file_path, "w") as sim_matrix_file:
    for type_1 in vocab_common:
        type_freq_file.write(type_1 + ";" + str(type_freq_dict[type_1]) + "\n")
        for type_2 in vocab_common:
            sim_matrix_file.write(str(wv_wiki.similarity(type_1, type_2)))
            if type_2 != vocab_common[len(vocab_common) - 1]:
                sim_matrix_file.write(";")
            else:
                sim_matrix_file.write("\n")
