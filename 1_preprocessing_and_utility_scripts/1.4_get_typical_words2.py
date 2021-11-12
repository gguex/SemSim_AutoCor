import os
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import heapq
import pandas as pd

# -------------------------------------
# --- Parameters
# -------------------------------------

input_text_folder = "corpora/manifesto_pp"

stop_words = False

output_word_file = "results/manifesto_typical_words.csv"
output_stat_file = "results/manifesto_typical_stat.csv"

top_n = 20

min_freq_in_txt = 1

# -------------------------------------
# --- Computations
# -------------------------------------

# List files in the corpus folder
file_list = os.listdir(input_text_folder)

# Restrict them to those with or without stopwords
file_list = [file for file in file_list if ("wostw" in file) ^ stop_words]

# Sort the list
file_list.sort()

# Split groups and non-groups file
input_text_file_list = [f"{input_text_folder}/{file}" for file in file_list if "groups" not in file]
input_group_file_list = [f"{input_text_folder}/{file}" for file in file_list if "groups" in file]

# -------------------------------------
# --- Computations
# -------------------------------------

full_token_list = []
full_group_list = []
corpus_by_text = []
for i, input_text_file in enumerate(input_text_file_list):

    # Open text file
    with open(input_text_file, "r") as text_file:
        text_string = text_file.read()

    # Put the token in the full list
    text_token_list = nltk.word_tokenize(text_string)
    full_token_list.extend(text_token_list)

    # Store the corpus
    corpus_by_text.append(" ".join(text_token_list))

    # Open group file
    with open(input_group_file_list[i], "r") as group_file:
        group_file_cont = group_file.read()

    # Put the groups in the full list
    full_group_list.extend([int(element) for element in group_file_cont.split(",")])


# id of groups and number
id_group_list = list(set(full_group_list))
n_groups = len(id_group_list)

# Make the corpus by text
corpus_by_group = []
for id_group in id_group_list:
    group_token = list(np.array(full_token_list)[np.array(full_group_list) == id_group])
    corpus_by_group.append(" ".join(group_token))

# Vectorizer
vectorizer_txt = CountVectorizer()
text_token = vectorizer_txt.fit_transform(corpus_by_text)

# Keep word with enough occurences
index_ok = (np.min(text_token, axis=0) > min_freq_in_txt).todense().A1
# Token list
type_vec = np.array(vectorizer_txt.get_feature_names())[index_ok]

# Vectorizer
vectorizer_grp = CountVectorizer(vocabulary=type_vec)
CT = vectorizer_grp.fit_transform(corpus_by_group)
X = (CT.T / np.sum(CT.T, axis=1)).T

# Odd ratio
new_X = np.copy(X)
for i, id_group in enumerate(id_group_list):
    other_group_id = list(set(range(n_groups)) - set([i]))
    new_X[i, :] = X[i, :] / (np.sum(X[other_group_id, :], axis=0) + 1e-40)
X = new_X

# Token list
type_list = vectorizer_grp.get_feature_names()

# Get the highest values
highest_type_id = []
for i, _ in enumerate(id_group_list):
    highest_id = heapq.nlargest(top_n, range(len(type_list)), X[i, ].take)
    highest_type_id.extend(highest_id)

highest_type_id = np.sort(list(set(highest_type_id)))
highest_value = X[:, highest_type_id]
highest_type_list = np.array(type_list)[highest_type_id]

df_results = pd.DataFrame(highest_value.T, index=highest_type_list)

df_results.to_csv(output_word_file, header=False)