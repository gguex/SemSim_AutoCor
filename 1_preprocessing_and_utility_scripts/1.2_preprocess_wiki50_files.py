import os
import string
import re
import nltk
from nltk.corpus import stopwords

# Set root path
root_path = os.getcwd().split("SemSim_AutoCor")[0] + "SemSim_AutoCor"

# Setting the raw files path
raw_files_path = f"{root_path}/corpora/wiki50"

# Setting the output files path
output_files_path = f"{root_path}/corpora/wiki50_pp"

# Getting files
input_file_list = os.listdir(raw_files_path)

# Removing stopwords option
remove_stopwords = False

# Getting stopwords
stopwords = stopwords.words('english')

# Improved punctuation
improved_punctuation = string.punctuation + "”’—“–"

# Loop on files
for input_file in input_file_list:

    # Open file and get every sentence (line)
    with open(f"{raw_files_path}/{input_file}", 'r') as text_file:
        sent_list = text_file.readlines()
        sent_list = [sent for sent in sent_list if "*LIST*" not in sent]

    # Get the index of separating sentences
    sep_index_list = [i for i, sent in enumerate(sent_list) if re.match("=====", sent)]

    # Preprocessing each sentence
    if remove_stopwords:
        output_name = f"{input_file}_pp_wostw"
    else:
        output_name = f"{input_file}_pp"
    with open(f"{output_files_path}/{output_name}.txt", "w") as text_file, \
            open(f"{output_files_path}/{output_name}_groups.txt", "w") as groups_file:

        # Loop on sentences
        group_value = 0
        for i, sentence in enumerate(sent_list):
            if i in sep_index_list:
                if not ((i > 0) & ((i-1) in sep_index_list)):
                    group_value += 1
            else:
                # Lower cases
                sentence_pp = sentence.lower()
                # Remove numbers
                sentence_pp = re.sub(r"[0-9]", " ", sentence_pp)
                # Remove punctuation
                sentence_pp = sentence_pp.translate(
                    str.maketrans(improved_punctuation, " " * len(improved_punctuation)))
                # Split by token
                token_list = nltk.word_tokenize(sentence_pp)

                # Removing stopwords if option is on
                if remove_stopwords:
                    token_list = [token for token in token_list if token not in stopwords]

                if len(token_list) > 0:
                    # Write the sentence
                    text_file.write(" ".join(token_list))
                    # Write the groups
                    groups_file.write(",".join([str(group_value)] * len(token_list)))
                    # If not last sentence, write a connector
                    if i < (len(sent_list) - 1):
                        text_file.write(" \n")
                        groups_file.write(",")
