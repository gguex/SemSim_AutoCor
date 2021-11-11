from sentence_transformers import SentenceTransformer, util
from local_functions import *
from sklearn.metrics import normalized_mutual_info_score

# -------------------------------------
# --- Parameters
# -------------------------------------

input_text_file = "corpora/wiki50_pp/28187_pp.txt"
input_group_file = "corpora/wiki50_pp/28187_pp_groups.txt"

output_names_root = "results/61320_199211_sbert"

dist_option = "max_minus"
exch_mat_opt = "u"
exch_range = 15
alpha = 5
beta = 50
kappa = 0.5
known_label_ratio = 0 # if > 0, semi-supervised model

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
real_group_vec = []
for sent in sent_list:
    sent_token = nltk.word_tokenize(sent)
    token_group = group_list[ind_1:(ind_1 + len(sent_token))]
    real_group_vec.append(int(max(set(token_group), key=token_group.count)))
    ind_1 = ind_1 + len(sent_token)
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
exch_mat, w_mat = exchange_and_transition_matrices(len(sent_list),
                                                   exch_mat_opt=exch_mat_opt,
                                                   exch_range=exch_range)

# Compute the membership matrix
result_matrix = token_clustering(d_ext_mat=d_ext_mat, exch_mat=exch_mat, w_mat=w_mat, n_groups=n_groups, alpha=alpha,
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

print(f"nmi = {nmi}, ext_nmi = {ext_nmi}, pk={pk_res} (rdm={pk_rdm}), win_diff={win_diff} (rdm={win_diff_rdm})")