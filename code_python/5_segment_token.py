import numpy as np
import nltk
import pandas as pd
import os

# --- Parameters --- #

# Path of the text file with only nouns, verbs, adjectives or adverbs to compute autocorrelation
input_file = "The_WW_of_Oz_nouns.txt"

# Name of the tag for the similarity
sim_tag = "wesim"

# Autocorrelation windows size
segm_range = 100

# Number of groups
n_groups = 5

# Alpha parameter
alpha = 0.1

# Beta parameter
beta = 0.1

# --- Defining paths --- #

# Getting the base path (must run the script from a folder inside the "SemSim_Autocor" folder)
working_path = os.getcwd()
base_path = str.split(working_path, "SemSim_AutoCor")[0] + "SemSim_AutoCor/"

# Path of the text file
file_path = base_path + "corpora/" + input_file
# Path of the types and frequencies file
typefreq_file_path = base_path + "similarities_frequencies/" + input_file[:-4] + "_" + sim_tag + "_typefreq.txt"
# Path of the similarity matrix
similarities_file_path = base_path + "similarities_frequencies/" + input_file[:-4] + \
                         "_" + sim_tag + "_similarities.txt"
# Results path file
results_file_path = base_path + "results/" + input_file[:-4] + "_" + sim_tag + "_segm" + str(segm_range) + ".png"

# --- Load the data --- #

# Import the type freq file
type_freq_df = pd.read_csv(typefreq_file_path, sep=";", header=None)
type_list = list(type_freq_df[0])
freq_vec = np.array(type_freq_df[1])
freq_vec = freq_vec / sum(freq_vec)
n_type = len(type_list)  # The number of types

# Import the text file and remove non-existing token
with open(file_path, "r") as text_file:
    text_string = text_file.read()
raw_token_list = nltk.word_tokenize(text_string)
token_list = [token for token in raw_token_list if token in type_list]
n_token = len(token_list)  # The number of tokens

# Import the similarity matrix
sim_mat = np.loadtxt(similarities_file_path, delimiter=";")

# --- Computation ---#

# Compute the exchange matrix and the markov chain transition matrix
exch_m = np.abs(np.add.outer(np.arange(n_token), -np.arange(n_token))) <= segm_range
np.fill_diagonal(exch_m, 0)
exch_m = exch_m / np.sum(exch_m)
f_vec = np.sum(exch_m, axis=0)
w_mat = (exch_m / f_vec).T

# Easy computation of dissimilarity matrix
d_mat = 1 - sim_mat

# Compute the n_type x n_token presence matrix
pres_mat = np.empty([0, n_token])
for type_i in type_list:
    pres_mat = np.append(pres_mat, [[token == type_i for token in token_list]], axis=0)

# Compute the extended distance matrix
d_ext_mat = pres_mat.T.dot(d_mat.dot(pres_mat))

