from local_functions import *
from os.path import expanduser
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

# -------------------------------------
# --- Parameters
# -------------------------------------

input_text_file = "corpora/manifesto_pp/61320_199211_pp_wostw.txt"
input_group_file = "corpora/manifesto_pp/61320_199211_pp_wostw_groups.txt"
home = expanduser("~")
vector_model_path = f"{home}/Documents/data/pretrained_word_vectors/enwiki.model"

n_groups = 7
dist_option = "max_minus"
exch_mat_opt = "u"
exch_range = 15
alpha = 5
beta = 50
kappa = 0.5
known_label_ratio = 0  # if > 0, semi-supervised model

# Block size
block_size = 2000

# -------------------------------------
# --- Computations
# -------------------------------------


z_res, existing_token_list, existing_pos_list = \
    token_clustering_on_file(input_text_file, vector_model_path, dist_option,
                             exch_mat_opt, exch_range, n_groups, alpha, beta,
                             kappa, block_size=block_size, verbose=True, strong_pass=True)

write_groups_in_html_file("test.html", existing_token_list, z_res)

# Compute the aggregate labels
df_results = pd.DataFrame(z_res)
df_results["Token"] = existing_token_list
type_results = df_results.groupby("Token").mean()
type_list = list(type_results.index)
type_values = type_results.to_numpy()

write_membership_mat_in_csv_file("test.csv", type_list, type_values)

# Compute the groups
algo_group_vec = np.argmax(z_res, 1) + 1

# Loading ground truth
with open(input_group_file) as ground_truth:
    real_group_vec = ground_truth.read()
    real_group_vec = np.array([int(element) for element in real_group_vec.split(",")])
real_group_vec = real_group_vec[existing_pos_list]

# Compute nmi score
nmi = normalized_mutual_info_score(real_group_vec, algo_group_vec)

print(nmi)
