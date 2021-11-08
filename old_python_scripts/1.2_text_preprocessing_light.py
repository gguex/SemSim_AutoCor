import os
import string
import nltk
import re
import contractions
from tqdm import tqdm

# -------------------------------------
# --- Parameters
# -------------------------------------

# Corpus name list
corpus_name_list = ["Civil_Disobedience.txt",
                    "Flowers_of_the_Farm.txt",
                    "Sidelights_on_relativity.txt",
                    "Prehistoric_Textile.txt"]

# -------------------------------------
# --- Computations
# -------------------------------------

# Getting the base path (must run the script from a folder inside the "SemSim_Autocor" folder)
working_path = os.getcwd()
base_path = str.split(working_path, "SemSim_AutoCor")[0] + "/SemSim_AutoCor/"

# Loop on coprora
for corpus_name in tqdm(corpus_name_list):

    # Path of the raw text file
    text_file_path = f"{base_path}corpora/{corpus_name}"
    # Path of the outputted preprocessed text file
    pp_output_path = f"{base_path}corpora/{corpus_name[:-4]}_ppl.txt"

    # --- Loading and Preprocessing --- #

    # Opening the file
    with open(text_file_path, "r") as text_file:
        text_string = text_file.read()

    # To lower case
    text_string_pp = text_string.lower()

    # Remove contractions
    text_string_pp = contractions.fix(text_string_pp)

    # Remove number
    text_string_pp = re.sub(r"[0-9]", " ", text_string_pp)

    # Split by sentence
    sentence_list = nltk.sent_tokenize(text_string_pp)

    # Loop on sentences
    with open(pp_output_path, "w") as output_file:
        for sentence in sentence_list:

            # Remove punctuation
            sentence_pp = sentence.translate(str.maketrans(" ", " ", string.punctuation))

            # Split by token and remove punctuation
            tokenizer = nltk.RegexpTokenizer(r"\w+")
            token_list = tokenizer.tokenize(sentence_pp)

            # Write the sentence
            output_file.write(" ".join(token_list) + "\n")