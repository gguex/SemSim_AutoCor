import os
import nltk
import numpy as np
import segeval

# -------------------------------------
# --- Parameters
# -------------------------------------

input_text_folder = "corpora/elements_pp"
stop_words = False

output_file = "results/stats_elements_wostw.csv"

# -------------------------------------
# --- Computation
# -------------------------------------

# List files in the corpus folder
file_list = os.listdir(input_text_folder)

# Restrict them to those with or without stopwords
file_list = [file for file in file_list if ("wostw" in file) ^ stop_words]

# Sort the list
file_list.sort()

# Split groups and non-groups file
text_file_list = [file for file in file_list if "groups" not in file]
group_file_list = [file for file in file_list if "groups" in file]

# Create results file
with open(output_file, "w") as output:
    output.write("file,n_token,n_type,n_sent,n_groups,mean_seq_len,mean_seq_sent_len\n")

# Lists for mean values
n_token_list = []
n_type_list = []
n_sent_list = []
n_groups_list = []
mean_seq_len_list = []
mean_seq_sent_len_list = []
# Loop on files
for i, text_file in enumerate(text_file_list):

    # Get the corpus
    with open(f"{input_text_folder}/{text_file}", "r") as txt_f:
        sent_list = txt_f.readlines()
    # Make the whole text
    text_string = " ".join(sent_list)
    # Split by tokens
    token_list = nltk.word_tokenize(text_string)
    # Vocabulary of text
    vocab_text = set(token_list)

    # Get the groups
    with open(f"{input_text_folder}/{group_file_list[i]}", "r") as grp_f:
        token_group_vec = grp_f.read()
        token_group_vec = np.array([int(element) for element in token_group_vec.split(",")])

    n_groups = len(set(token_group_vec))
    token_segm_vec = segeval.convert_positions_to_masses(token_group_vec)

    # Make groups by sentences
    sent_group_vec = []
    ind_1 = 0
    for sent in sent_list:
        sent_token = nltk.word_tokenize(sent)
        token_group = list(token_group_vec[ind_1:(ind_1 + len(sent_token))])
        sent_group_vec.append(int(max(set(token_group), key=token_group.count)))
        ind_1 = ind_1 + len(sent_token)
    sent_group_vec = np.array(sent_group_vec)
    sent_segm_vec = segeval.convert_positions_to_masses(sent_group_vec)

    # Write results
    with open(output_file, "a") as output:
        output.write(f"{text_file},{len(token_list)},{len(vocab_text)},{len(sent_list)},{n_groups},"
                     f"{np.mean(token_segm_vec)},{np.mean(sent_segm_vec)}\n")

    # Save them
    n_token_list.append(len(token_list))
    n_type_list.append(len(vocab_text))
    n_sent_list.append(len(sent_list))
    n_groups_list.append(n_groups)
    mean_seq_len_list.append(np.mean(token_segm_vec))
    mean_seq_sent_len_list.append(np.mean(sent_segm_vec))

# Write mean values
with open(output_file, "a") as output:
    output.write(f"Mean_values,{np.median(n_token_list)},{np.median(n_type_list)},{np.median(n_sent_list)},"
                 f"{np.median(n_groups_list)},{np.median(mean_seq_len_list)},{np.median(mean_seq_sent_len_list)}\n")
