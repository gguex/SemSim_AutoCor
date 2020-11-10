import nltk
from flair.models import SequenceTagger
from flair.data import Sentence
import re
from tqdm import tqdm
import os

# --- Parameters --- #

# Corpus names
corpus_name_list = ["Lectures_on_Landscape_all.txt",
                    "Metamorphosis_all.txt",
                    "On_the_Duty_of_Civil_Disobedience_all.txt",
                    "Sidelights_on_relativity_all.txt"]

# Sentence mixing, or words
unit_of_mix = "sent"
#unit_of_mix = "word"
if unit_of_mix not in ["sent", "word"]:
    unit_of_mix = "sent"

# Number of units for each bin
nb_of_units = 1

# Minimum of words in sentence to mix
min_nb_of_words_in_sent = 5

# --- Defining paths --- #

# Getting the base path (must run the script from a folder inside the "SemSim_Autocor" folder)
working_path = os.getcwd()
base_path = str.split(working_path, "SemSim_AutoCor")[0] + "/SemSim_AutoCor/"

# Defining paths of the text files
text_path_list = [base_path + "corpora/mixed_corpora/" + corpus_name for corpus_name in corpus_name_list]

# Defining paths of the outputs
if unit_of_mix == "sent":
    output_file = f"{base_path}corpora/mixed_corpora/mix_sent{nb_of_units}_min{min_nb_of_words_in_sent}.txt"
    word_group_file = f"{base_path}corpora/mixed_corpora/wrdgrp_sent{nb_of_units}_min{min_nb_of_words_in_sent}.txt"
else:
    output_file = f"{base_path}corpora/mixed_corpora/mix_word{nb_of_units}.txt"
    word_group_file = f"{base_path}corpora/mixed_corpora/wrdgrp_word{nb_of_units}.txt"

# --- Creating mixing file --- #

# Loop for opening file
text_string_list = []
for text_path in text_path_list:
    with open(text_path, "r") as text_file:
        if unit_of_mix == "sent":
            text_string_list.append(text_file.read().splitlines())
        else:
            # A FAIRE text_string_list.append(text_file.read().rstrip("\n"))