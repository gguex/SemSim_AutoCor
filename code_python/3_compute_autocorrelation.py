import os

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Parameters --- #

# Path of the text file with only nouns, verbs, adjectives or adverbs to compute autocorrelation
input_files = ("The_WW_of_Oz_nouns.txt", "The_WW_of_Oz_verbs.txt", "Animal_farm_nouns.txt", "Animal_farm_verbs.txt")
sim_tags = ("resnik", "wu-palmer", "leacock-chodorow", "wesim")

# Autocorrelation range limit
max_range = 50

# --- Defining paths --- #

# Getting the base path (must run the script from a folder inside the "SemSim_Autocor" folder)
working_path = os.getcwd()

def compute_autocorrelation(working_path, input_file, sim_tag):
    base_path = str.split(working_path, "SemSim_AutoCor")[0] + "SemSim_AutoCor/"

    # Path of the text file
    file_path = base_path + "corpora/" + input_file
    # Path of the types and frequencies file
    typefreq_file_path = base_path + "similarities_frequencies/" + input_file[:-4] + "_" + sim_tag + "_typefreq.txt"

    if not (os.path.exists(file_path) and os.path.exists(typefreq_file_path)):
        return

    # Path of the similarity matrix
    similarities_file_path = base_path + "similarities_frequencies/" + input_file[:-4] + \
                             "_" + sim_tag + "_similarities.txt"
    # Results path file
    results_file_path = base_path + "results/" + input_file[:-4] + "_" + sim_tag + "_autocor.png"

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

    # --- Compute the dissimilarity matrix and the exchange matrix function ---#

    # Easy computation of dissimilarity matrix (BUT CERTAINLY BETTER CHOICES ?)
    d_mat = np.max(sim_mat) - sim_mat

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
    r_range = list(range(1, max_range + 1))

    # autocor_vec
    autocor_vec = []
    for r in tqdm(r_range):
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
    plt.title(input_file + " | Sim: " + sim_tag + " | n tokens: " + str(n_token))
    plt.xlabel("Neighbourhood size r")
    plt.ylabel("Autocorrelation index")
    plt.savefig(results_file_path)
    plt.close()

for input_file in input_files:
    for sim_tag in sim_tags:
        compute_autocorrelation(working_path, input_file, sim_tag)
