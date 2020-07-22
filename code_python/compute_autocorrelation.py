import numpy as np
import nltk
import pandas as pd
import matplotlib.pyplot as plt

# --- Parameters --- #

# Path of the text file with only nouns
noun_only_file_path = "/home/gguex/Documents/data/corpora/The_Wonderful_Wizard_of_Oz_noun_only.txt"
# Path of the present types and frequencies
type_freq_file_path = \
    "/home/gguex/Documents/recherche/SemSim_AutoCor-master/similarities_frequencies/Wizard_of_Oz_typefreq.txt"
# Path of the similarity matrix
sim_matrix_file_path = \
    "/home/gguex/Documents/recherche/SemSim_AutoCor-master/similarities_frequencies/Wizard_of_Oz_similarities.txt"

# --- Load the data --- #

# Import the type freq file
type_freq_df = pd.read_csv(type_freq_file_path, sep=";", header=None)
type_list = list(type_freq_df[0])
freq_vec = np.array(type_freq_df[1])
freq_vec = freq_vec / sum(freq_vec)
n_type = len(type_list)  # The number of types

# Import the text file and remove non-existing token
with open(noun_only_file_path, "r") as text_file:
    text_string = text_file.read()
raw_token_list = nltk.word_tokenize(text_string)
token_list = [token for token in raw_token_list if token in type_list]
n_token = len(token_list)  # The number of tokens

# Import the similarity matrix
sim_mat = np.loadtxt(sim_matrix_file_path, delimiter=";")

# --- Compute the dissimilarity matrix and the exchange matrix function ---#

# Easy computation of dissimilarity matrix (BUT CERTAINLY BETTER CHOICES ?)
d_mat = 1 - sim_mat

# Function for the exchange matrix
def exch_mat(r):
    exch_m = np.abs(np.add.outer(np.arange(n_token), -np.arange(n_token))) <= r
    np.fill_diagonal(exch_m, 0)
    exch_m = exch_m / np.sum(exch_m)
    return exch_m


# --- Computation of the presence matrix ---#

# Compute the n_type x n_token presence matrix
pres_mat = np.empty([0, n_token])
for type_i in type_list:
    pres_mat = np.append(pres_mat, [[token == type_i for token in token_list]], axis=0)

# --- Computation of the autocorrelation --- #

# Computation of the global inertia
global_inertia = np.sum(np.outer(freq_vec, freq_vec) * d_mat) / 2

# r_range
r_range = list(range(1, 51))

# autocor_vec
autocor_vec = []
for r in r_range:
    # epsilon matrix (page 5 article)
    epsilon_mat = pres_mat.dot(exch_mat(r).dot(pres_mat.T))
    # local inertial
    local_inertia = np.sum(epsilon_mat * d_mat) / 2
    # autocorrelation index
    autocor_index = (global_inertia - local_inertia) / global_inertia
    # append
    autocor_vec.append(autocor_index)

# --- Plot --- #

plt.figure("autocorrelation")
plt.scatter(r_range, autocor_vec)
plt.plot(r_range, autocor_vec)
plt.xlabel("Neighbourhood size r")
plt.ylabel("Autocorrelation index")
plt.show()
