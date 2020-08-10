import numpy as np
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.linalg import expm

# --- Parameters --- #

# Path of the text file with only nouns, verbs, adjectives or adverbs to compute autocorrelation
input_file = "The_WW_of_Oz_nouns.txt"

# Name of the tag for the similarity
sim_tag = "wesim"

# Exchange matrix option ("u" = uniform, "d" = diffusive)
exch_mat_opt = "d"
# Exchange matrix range (for uniform) OR time step (for diffusive)
exch_range = 1000

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
results_file_path = base_path + "results/" + input_file[:-4] + "_" + sim_tag + "_lisa" + str(exch_range) + ".png"

reshtml_file_path = base_path + "results/" + input_file[:-4] + "_" + sim_tag + "_lisa" + str(exch_range) + ".html"

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

# --- Computation --- #

# Compute the exchange matrix and the markov chain transition matrix
f_vec = np.ones(n_token) / n_token

if exch_mat_opt not in ["u", "d"]:
    print("EXCHANGE MATRIX OPTION NOT RECOGNIZED, SETTING IT TO UNIFORM")
    exch_mat_opt = "u"

if exch_mat_opt == "u":
    adj_mat = np.abs(np.add.outer(np.arange(n_token), -np.arange(n_token))) <= exch_range
    np.fill_diagonal(adj_mat, 0)
    g_vec = np.sum(adj_mat, axis=1) / np.sum(adj_mat)
    k_vec = f_vec / g_vec
    b_mat = np.array([[min(v1, v2) for v2 in g_vec] for v1 in g_vec]) * adj_mat / np.sum(adj_mat)
    exch_mat = np.diag(f_vec) - np.diag(np.sum(b_mat, axis=1)) + b_mat
else:
    adj_mat = np.abs(np.add.outer(np.arange(n_token), -np.arange(n_token))) <= 1
    np.fill_diagonal(adj_mat, 0)
    l_adj_mat = np.diag(np.sum(adj_mat, axis=1)) - adj_mat
    pi_outer_mat = np.outer(np.sqrt(f_vec), np.sqrt(f_vec))
    phi_mat = (l_adj_mat / pi_outer_mat) / np.trace(l_adj_mat)
    exch_mat = expm(- exch_range * phi_mat) * pi_outer_mat

# Compute the exchange matrix and the markov chain transition matrix (OLD)
# exch_m = np.abs(np.add.outer(np.arange(n_token), -np.arange(n_token))) <= lisa_range
# np.fill_diagonal(exch_m, 0)
# exch_m = exch_m / np.sum(exch_m)
# f_vec = np.sum(exch_m, axis=0)

w_mat = (exch_mat / np.sum(exch_mat, axis=1)).T

# Easy computation of dissimilarity matrix
d_mat = 1 - sim_mat

# Compute the n_type x n_token presence matrix
pres_mat = np.empty([0, n_token])
for type_i in type_list:
    pres_mat = np.append(pres_mat, [[token == type_i for token in token_list]], axis=0)

# Compute the extended distance matrix
d_ext_mat = pres_mat.T.dot(d_mat.dot(pres_mat))

# Compute the centration matrix
h_mat = np.identity(n_token) - np.outer(np.ones(n_token), f_vec)

# Compute the scalar produced matrix
b_mat = - 0.5 * h_mat.dot(d_ext_mat.dot(h_mat.T))

# Compute of the global inertia
global_inertia = np.sum(np.outer(freq_vec, freq_vec) * d_mat) / 2

# Compute lisa
lisa_vec = np.diag(w_mat.dot(b_mat)) / global_inertia

# --- Plot --- #

plt.figure("Lisa index")
plt.plot(list(range(1, n_token + 1)), lisa_vec, linewidth=0.1)
plt.title("{} | Sim: {} | Ntoken: {} | Exch: {} | Exch_range: {}".format(input_file, sim_tag, n_token, exch_mat_opt,
                                                                         exch_range))
plt.xlabel("text token")
plt.ylabel("Lisa index")
plt.savefig(results_file_path)

# --- HTML output --- #

# vector of color Lisa
color_lisa = np.copy(lisa_vec)
color_lisa[color_lisa > 0] = color_lisa[color_lisa > 0] / np.max(lisa_vec) * 255
color_lisa[color_lisa < 0] = color_lisa[color_lisa < 0] / np.abs(np.min(lisa_vec)) * 255
color_lisa = np.intc(color_lisa)

with open(reshtml_file_path, 'w') as html_file:
    html_file.write("<html>\n<head></head>\n")
    html_file.write("<body><p>Input file: {} | Similarity tag: {} | Exchange matrix option: {} | "
                    "Exchange matrix range: {}</p> <p>".format(input_file,
                                                               sim_tag,
                                                               exch_mat_opt,
                                                               exch_range))
    for i in range(len(token_list)):
        if color_lisa[i] >= 0:
            html_file.write("<span style=\"background-color: rgb({},255,{})\">".format(
                255 - color_lisa[i], 255 - color_lisa[i]))
        else:
            html_file.write("<span style=\"background-color: rgb(255,{},{})\">".format(
                255 + color_lisa[i], 255 + color_lisa[i]))
        html_file.write(token_list[i] + " </span>")
    html_file.write("</p></body>\n</html>")
