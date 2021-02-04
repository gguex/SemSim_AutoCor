# from gensim.models.ldamodel import LdaModel
# from gensim.corpora.dictionary import Dictionary
# from code_python.local_functions import get_all_paths, type_to_token_matrix_expansion
# import numpy as np
# import csv
# from sklearn.metrics import normalized_mutual_info_score

# -------------------------------------
# --- Parameters
# -------------------------------------

# File name to explore
input_file = "mix_word1.txt"
# Similarity tag
sim_tag = "w2v"
# Number of groups
n_groups = 4
# Number of test
n_tests = 50

# -------------------------------------
# --- Loading and preprocessing
# -------------------------------------

# Get the file paths
text_file_path, typefreq_file_path, sim_file_path, ground_truth_path = get_all_paths(input_file, sim_tag)

# Loading the similarity matrix
sim_mat = np.loadtxt(sim_file_path, delimiter=";")
# And the corresponding list of types
with open(typefreq_file_path, 'r') as typefreq_file:
    csv_reader = csv.reader(typefreq_file, delimiter=";")
    type_list = [row[0] for row in csv_reader]
# Compute the extended version of the similarity matrix
sim_ext_mat, token_list = type_to_token_matrix_expansion(text_file_path, sim_mat, type_list)

# Loading ground truth
with open(ground_truth_path) as ground_truth:
    real_group_vec = ground_truth.read()
    real_group_vec = np.array([int(element) for element in real_group_vec.split(",")])

# Create the new list of types
new_type_list = list(set(token_list))
new_type_list.sort()

# Preparing token_list for LDA
dict_token = Dictionary([token_list])
lda_corpus = [dict_token.doc2bow(token_list)]

nmi_vec = []
for i in range(n_tests):
    # Computing the LDA
    lda = LdaModel(lda_corpus,
                   num_topics=n_groups,
                   alpha="auto",
                   eta="auto")

    # Get groups for type
    type_group_value = np.argmax(lda.get_topics().T, 1) + 1
    # Get it for token
    algo_group_value = [type_group_value[new_type_list.index(token)] for token in token_list]

    # Compute nmi score
    nmi = normalized_mutual_info_score(real_group_vec, algo_group_value)
    nmi_vec.append(nmi)

print(np.array(nmi_vec).mean())
print(np.array(nmi_vec).std() / np.sqrt(n_tests) * 2)