from local_functions import *
from os.path import expanduser
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

input_file = "61320_199211_ppl.txt"

sim_tag = "w2v"
dist_option = "max_minus"
exch_mat_opt = "u"
exch_range = 15
alpha = 5
beta = 50
kappa = 0.5

block_size = 7000
n_groups = 7

# Getting the file path
working_path = os.getcwd()
base_path = str.split(working_path, "SemSim_AutoCor")[0] + "SemSim_AutoCor"
file_path = f"{base_path}/corpora/{input_file}"

# Getting the wv path
home = expanduser("~")
if sim_tag == "w2v":
    word_vector_path = f"{home}/Documents/data/pretrained_word_vectors/enwiki.model"
elif sim_tag == "glv":
    word_vector_path = f"{home}/Documents/data/pretrained_word_vectors/glove42B300d.model"
else:
    word_vector_path = f"{home}/Documents/data/pretrained_word_vectors/en_fasttext.model"

z_res, existing_token_list, existing_pos_list = cut_clustering_from_raw(file_path, word_vector_path, dist_option,
                                                                        exch_mat_opt, exch_range, n_groups, alpha, beta,
                                                                        kappa, block_size=block_size, verbose=True)

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
_, _, _, ground_truth_path = get_all_paths(input_file, sim_tag)
# Loading ground truth
with open(ground_truth_path) as ground_truth:
    real_group_vec = ground_truth.read()
    real_group_vec = np.array([int(element) for element in real_group_vec.split(",")])
real_group_vec = real_group_vec[existing_pos_list]
# Compute nmi score
nmi = normalized_mutual_info_score(real_group_vec, algo_group_vec)

print(nmi)


