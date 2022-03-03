from local_functions import *
from sklearn.metrics import normalized_mutual_info_score
from tqdm import tqdm
import timeit

# -------------------------------------
# --- Parameters
# -------------------------------------

# Input text file
input_text_file = "corpora/manifesto_pp/61320_201211_pp_wostw.txt"
# Input group file
input_group_file = "corpora/manifesto_pp/61320_201211_pp_wostw_groups.txt"
# Output file
output_file = "results/3.6_block_results/block_clust_weak_61320_201211.csv"

#---

# Number of groups (if none, extracted from data)
fixed_n_groups = None

# Block size
block_size_list = [1000, 2000, 3000, 4000, 5000]

# Strong pass
strong_pass = False

# Algo hyperparameters
sim_tag = "w2v"
dist_option = "max_minus"
exch_mat_opt = "u"
exch_range = 15
alpha = 5
beta = 100
kappa = 0.5

# -------------------------------------
# --- Computations
# -------------------------------------

# Vector models
home = os.path.expanduser("~")
if sim_tag == "w2v":
    vector_model_path = f"{home}/Documents/data/pretrained_word_vectors/enwiki.model"
elif sim_tag == "glv":
    vector_model_path = f"{home}/Documents/data/pretrained_word_vectors/glove42B300d.model"
else:
    vector_model_path = f"{home}/Documents/data/pretrained_word_vectors/en_fasttext.model"

# Loading ground truth
with open(input_group_file) as ground_truth:
    real_group_nr_vec = ground_truth.read()
    real_group_nr_vec = np.array([int(element) for element in real_group_nr_vec.split(",")])
if fixed_n_groups is None:
    n_groups = len(set(real_group_nr_vec))
else:
    n_groups = fixed_n_groups

# Create output file
with open(output_file, "w") as res_file:
    res_file.write(f"file,block_size,elapse_time,nmi,pk_res,pk_rdm,win_diff,win_diff_rdm\n")

for block_size in tqdm(block_size_list):

    # Starting time
    start_time = timeit.default_timer()

    # Run the algorithm
    result_matrix, existing_token_list, existing_pos_list = \
        spatial_clustering_on_file(input_text_file, vector_model_path, dist_option,
                                   exch_mat_opt, exch_range, n_groups, alpha, beta,
                                   kappa, block_size=block_size, verbose=True, strong_pass=strong_pass)


    # Elapse time
    elapse_time = timeit.default_timer() - start_time

    # Restrain real group
    real_group_vec = real_group_nr_vec[existing_pos_list]

    # Compute the groups
    algo_group_vec = np.argmax(result_matrix, 1) + 1

    # Compute nmi score
    nmi = normalized_mutual_info_score(real_group_vec, algo_group_vec)

    # Segmentation evaluation
    pk_res, win_diff, pk_rdm, win_diff_rdm = seg_eval(algo_group_vec, real_group_vec)

    with open(output_file, "a") as res_file:
        res_file.write(f"{input_text_file},{block_size},{elapse_time},{nmi},"
                       f"{pk_res},{pk_rdm},{win_diff},{win_diff_rdm}\n")

