import os
import csv
import string
import re
import nltk
from nltk.corpus import stopwords

# Getting the os path
working_path = os.getcwd()

# Getting the SemSim_AutoCor folder, if above
base_path = str.split(working_path, "SemSim_AutoCor")[0] + "SemSim_AutoCor"

# Name of the file
file_name_list = ["61320_199211.csv",
                  "61320_200411.csv",
                  "61320_201211.csv",
                  "61320_201611.csv",
                  "61320_202011.csv",
                  "61620_200411.csv",
                  "61620_200811.csv",
                  "61620_201211.csv",
                  "61620_201611.csv",
                  "61620_202011.csv"]

# Removing stopwords option
remove_stopwords = False

# Getting stopwords
stopwords = stopwords.words('english')

# Loop on files
for file_name in file_name_list:
    # To store the final number of tokens
    n_token = 0
    # Loading the file
    with open(f"{base_path}/corpora/manifesto_csv_file/{file_name}", 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        sent_list = []
        group_list = []
        for row in csv_reader:
            try:
                class_id = float(row[1])
                class_id = int(class_id // 100)
                if class_id > 0:
                    sent_list.append(row[0])
                    group_list.append(class_id)
            except:
                pass

    # Preprocessing each sentence
    if remove_stopwords:
        output_name = f"{file_name[:-4]}_pp_wostw"
    else:
        output_name = f"{file_name[:-4]}_pp"
    with open(f"{base_path}/corpora/{output_name}.txt", "w") as text_file, \
            open(f"{base_path}/corpora/{output_name}_groups.txt", "w") as groups_file:

        for i, sentence in enumerate(sent_list):

            # Lower cases
            sentence_pp = sentence.lower()
            # Remove numbers
            sentence_pp = re.sub(r"[0-9]", " ", sentence_pp)
            # Remove punctuation
            sentence_pp = sentence_pp.translate(str.maketrans(" ", " ", string.punctuation + "”’—“"))
            # Split by token
            token_list = nltk.word_tokenize(sentence_pp)

            # Removing stopwords if option is on
            if remove_stopwords:
                token_list = [token for token in token_list if token not in stopwords]

            # Write the sentence
            text_file.write(" ".join(token_list))
            # Write the groups
            groups_file.write(",".join([str(group_list[i])] * len(token_list)))
            if i < len(sent_list) - 1:
                text_file.write(" \n")
                groups_file.write(",")
