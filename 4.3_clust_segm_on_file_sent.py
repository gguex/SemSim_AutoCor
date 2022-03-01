from sentence_transformers import SentenceTransformer, util
from local_functions import *
from sklearn.metrics import normalized_mutual_info_score
import pandas as pd

# -------------------------------------
# --- Parameters
# -------------------------------------

# Input text file
input_text_file = "corpora/manifesto_pp/61320_201211_pp_wostw.txt"
# Input group file
input_group_file = "corpora/manifesto_pp/61320_201211_pp_wostw_groups.txt"
# Root name for output files
output_names_root = "results/sent_clust_61320_200411"

# ---

# Number of groups (if none, extracted from data)
n_groups = None

# Algo hyperparameters
dist_option = "max_minus"
exch_mat_opt = "u"
exch_range = 15
alpha = 5
beta = 50
kappa = 0.5
known_label_ratio = 0  # if > 0, semi-supervised model

# -------------------------------------
# --- Computations
# -------------------------------------

# Load corpus
with open(input_text_file, "r") as text_file:
    sent_list = text_file.readlines()

# Load ground truth
with open(input_group_file, "r") as group_file:
    group_list = group_file.read().split(",")

# Transform the vector to get 1 group by sentence
ind_1 = 0
real_group_vec, sent_weights, sent_token_list = [], [], []
for sent in sent_list:
    sent_token = nltk.word_tokenize(sent)
    sent_token_list.append(sent_token)
    token_group = group_list[ind_1:(ind_1 + len(sent_token))]
    real_group_vec.append(int(max(set(token_group), key=token_group.count)))
    ind_1 = ind_1 + len(sent_token)
    sent_weights.append(len(sent_token))
sent_weights = np.array(sent_weights) / sum(sent_weights)
real_group_vec = np.array(real_group_vec)
n_groups = len(set(real_group_vec))

# Load sentence model
sbert_model = SentenceTransformer("all-mpnet-base-v2")
# Make the sentence vectors
sentence_embeddings = sbert_model.encode(sent_list)
# Make sim matrix
sim_mat = np.array(util.pytorch_cos_sim(sentence_embeddings, sentence_embeddings))

# Compute the dissimilarity matrix
d_ext_mat = similarity_to_dissimilarity(sim_mat, dist_option=dist_option)

# Compute the exchange and transition matrices
exch_mat, w_mat = exchange_and_transition_matrices(len(sent_list), exch_mat_opt=exch_mat_opt, exch_range=exch_range)

# Compute the membership matrix
result_matrix = spatial_clustering(d_ext_mat=d_ext_mat, exch_mat=exch_mat, w_mat=w_mat, n_groups=n_groups, alpha=alpha,
                                   beta=beta, kappa=kappa, verbose=True)

# Compute the groups
algo_group_vec = np.argmax(result_matrix, 1) + 1
# Compute the groups on token
ext_real_group_vec = []
ext_algo_group_vec = []
for i, sent in enumerate(sent_list):
    ext_real_group_vec.extend([real_group_vec[i]] * len(sent))
    ext_algo_group_vec.extend([algo_group_vec[i]] * len(sent))

# Compute nmi, ext_nmi, p_k and win_diff score, then print them
nmi = normalized_mutual_info_score(real_group_vec, algo_group_vec)
ext_nmi = normalized_mutual_info_score(ext_real_group_vec, ext_algo_group_vec)
pk_res, win_diff, pk_rdm, win_diff_rdm = seg_eval(algo_group_vec, real_group_vec)

# Compute the membership of tokens
token_results_matrix = np.empty((0, n_groups))
token_list = []
for id_sent, sent_token in enumerate(sent_token_list):
    token_list.extend(sent_token)
    for token in sent_token:
        token_results_matrix = np.row_stack((token_results_matrix, result_matrix[id_sent, :]))

# Compute the P(g|w)
df_results = pd.DataFrame(token_results_matrix)
df_results["Token"] = token_list
type_results = df_results.groupby("Token").mean()
type_list = list(type_results.index)
type_values = type_results.to_numpy()
# Compute P(w|g)
ntype_results = df_results.groupby("Token").sum()
ntype_list = list(ntype_results.index)
ntype_values = ntype_results.to_numpy()
ntype_values = ntype_values / ntype_values.sum(axis=0)

# Compute the real membership matrix
z_real_mat = np.zeros((len(token_list), n_groups))
for i, label in enumerate(group_list):
    label = int(label)
    if label != 0:
        z_real_mat[i, :] = 0
        z_real_mat[i, label - 1] = 1

# -------------------------------------
# --- Writing
# -------------------------------------

# Write html results
write_groups_in_html_file(output_names_root + "_aglo.html", token_list, token_results_matrix,
                          comment_line=f"nmi = {nmi}, pk={pk_res}, win_diff={win_diff}")
# Write real html results
write_groups_in_html_file(output_names_root + "_real.html", token_list, z_real_mat, comment_line="Real results")
# Write csv results
write_membership_mat_in_csv_file(output_names_root + "_token.csv", token_list, token_results_matrix)
# Write csv type result
write_membership_mat_in_csv_file(output_names_root + "_P_topic_type.csv", type_list, type_values)
# Write csv type result
write_membership_mat_in_csv_file(output_names_root + "_P_type_topic.csv", ntype_list, ntype_values)
# Print stat
print(f"nmi = {nmi}, ext_nmi = {ext_nmi}, pk={pk_res} (rdm={pk_rdm}), win_diff={win_diff} (rdm={win_diff_rdm})")
