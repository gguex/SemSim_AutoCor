from code_python.functions import *

### Parameters

input_file = "mix_sent1_min5.txt"
sim_tag = "wesim"
dist_option = "minus_log"
exch_mat_opt = "u"
exch_range = 3
n_groups = 4
alpha = 1
beta = 10
kappa = 1

### Paths

# Working path
working_path = os.getcwd()
# Getting the SemSim_AutoCor folder, if above
base_path = str.split(working_path, "SemSim_AutoCor")[0] + "SemSim_AutoCor/"
# Path of the types and frequencies file
typefreq_file_path = base_path + "similarities_frequencies/" + input_file[:-4] + "_" + sim_tag + "_typefreq.txt"
# Path of the text file
file_path = base_path + "corpora/" + input_file

### Computations

# Compute the dissimilartiy_matrix
d_ext_mat = sim_to_dissim_matrix(input_file=input_file,
                                 sim_tag=sim_tag,
                                 dist_option=dist_option)

# Compute the exchange and transition matrices
exch_mat, w_mat = exchange_and_transition_mat(input_file=input_file,
                                              sim_tag=sim_tag,
                                              exch_mat_opt=exch_mat_opt,
                                              exch_range=exch_range)

# Compute the membership matrix
result_matrix = discontinuity_segmentation(d_ext_mat=d_ext_mat,
                                           exch_mat=exch_mat,
                                           w_mat=w_mat,
                                           n_groups=n_groups,
                                           alpha=alpha,
                                           beta=beta,
                                           kappa=kappa)

# Import the text file and tokenize
with open(file_path, "r") as text_file:
    text_string = text_file.read()
raw_token_list = nltk.word_tokenize(text_string)

# Import the type freq file for token list
type_freq_df = pd.read_csv(typefreq_file_path, sep=";", header=None)
type_list = type_freq_df[0].to_list()

# Remove non-existing tokens
token_list = [token for token in raw_token_list if token in type_list]

# Write html results
display_groups_in_html_file("test.html", token_list, result_matrix)

# Real results
with open("/home/gguex/PycharmProjects/SemSim_AutoCor/corpora/mixgroup_sent1_min5.txt") as group_file:
    real_group_vec = group_file.read()
    real_group_vec = np.array([int(element) for element in real_group_vec.split(",")])
z_real_mat = np.zeros((len(token_list), n_groups))
for i, label in enumerate(real_group_vec):
    if label != 0:
        z_real_mat[i, :] = 0
        z_real_mat[i, label - 1] = 1

# Write html results
display_groups_in_html_file("test_real.html", token_list, z_real_mat)
