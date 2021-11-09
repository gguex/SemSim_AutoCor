import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import heapq
import pandas as pd

# -------------------------------------
# --- Parameters
# -------------------------------------

output_file_name = "results/corpus_study.csv"

input_text_file_list = ["corpora/61320_199211_pp_wostw.txt",
                        "corpora/61320_200411_pp_wostw.txt",
                        "corpora/61320_201211_pp_wostw.txt",
                        "corpora/61320_201611_pp_wostw.txt",
                        "corpora/61320_202011_pp_wostw.txt",
                        "corpora/61620_200411_pp_wostw.txt",
                        "corpora/61620_200811_pp_wostw.txt",
                        "corpora/61620_201211_pp_wostw.txt",
                        "corpora/61620_201611_pp_wostw.txt",
                        "corpora/61620_202011_pp_wostw.txt"]

input_group_file_list = ["corpora/61320_199211_pp_wostw_groups.txt",
                         "corpora/61320_200411_pp_wostw_groups.txt",
                         "corpora/61320_201211_pp_wostw_groups.txt",
                         "corpora/61320_201611_pp_wostw_groups.txt",
                         "corpora/61320_202011_pp_wostw_groups.txt",
                         "corpora/61620_200411_pp_wostw_groups.txt",
                         "corpora/61620_200811_pp_wostw_groups.txt",
                         "corpora/61620_201211_pp_wostw_groups.txt",
                         "corpora/61620_201611_pp_wostw_groups.txt",
                         "corpora/61620_202011_pp_wostw_groups.txt"]

top_n = 40

# -------------------------------------
# --- Computations
# -------------------------------------

full_token_list = []
full_group_list = []

for i, input_text_file in enumerate(input_text_file_list):

    # Open text file
    with open(input_text_file, "r") as text_file:
        text_string = text_file.read()

    # Put the token in the full list
    text_token_list = nltk.word_tokenize(text_string)
    full_token_list.extend(text_token_list)

    # Open group file
    with open(input_group_file_list[i], "r") as group_file:
        group_file_cont = group_file.read()

    # Put the groups in the full list
    full_group_list.extend([int(element) for element in group_file_cont.split(",")])

# id of groups and number
id_group_list = list(set(full_group_list))
n_groups = len(id_group_list)

# Make the corpus by groups
corpus_by_group = []
for id_group in id_group_list:
    group_token = list(np.array(full_token_list)[np.array(full_group_list) == id_group])
    corpus_by_group.append(" ".join(group_token))

# Tfidf computation
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus_by_group)

# Token list
type_list = vectorizer.get_feature_names()

# Get the highest values
highest_type_id = []
for i, _ in enumerate(id_group_list):
    highest_id = heapq.nlargest(top_n, range(X[i, ].shape[1]), X[i, ].todense().take)
    highest_type_id.extend(highest_id)

highest_type_id = np.sort(list(set(highest_type_id)))
highest_value = X[:,highest_type_id].todense()
highest_type_list = np.array(type_list)[highest_type_id]

df_results = pd.DataFrame(highest_value.T, index=highest_type_list)

# Construct the index
for i, id_group in enumerate(id_group_list):
    other_group_id = list(set(range(n_groups)) - set([i]))
    df_results[f"I{id_group}"] = df_results.iloc[:, i] / df_results.iloc[:, other_group_id].max(axis=1)

df_results.to_csv(output_file_name, header=False)