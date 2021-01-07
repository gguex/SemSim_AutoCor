import re
import numpy as np
import os
import random

# -------------------------------------
# --- Parameters
# -------------------------------------

# Corpus names
corpus_names_list = ["Lectures_on_Landscape_all.txt",
                    "Metamorphosis_all.txt",
                    "Civil_Disobedience_all.txt",
                    "Sidelights_on_relativity_all.txt"]

# Sentence mixing, or words
unit_of_mix = "sent"
#unit_of_mix = "word"
if unit_of_mix not in ["sent", "word"]:
    unit_of_mix = "sent"

# Number of units for each bin
nb_of_units = 10

# Minimum of words in sentence to mix
min_nb_of_words_in_sent = 5

# Div factor (corpus size is divided by that to fit github size limit)
div_factor = 2

# -------------------------------------
# --- Computations
# -------------------------------------

# --- Defining paths --- #

# Getting the base path (must run the script from a folder inside the "SemSim_Autocor" folder)
working_path = os.getcwd()
base_path = str.split(working_path, "SemSim_AutoCor")[0] + "/SemSim_AutoCor/"

# Defining paths of the text files
text_path_list = [base_path + "corpora/" + corpus_name for corpus_name in corpus_names_list]

# Defining paths of the outputs
if unit_of_mix == "sent":
    output_file = f"{base_path}corpora/mix_sent{nb_of_units}_min{min_nb_of_words_in_sent}.txt"
    word_group_file = f"{base_path}corpora/mixgroup_sent{nb_of_units}_min{min_nb_of_words_in_sent}.txt"
else:
    output_file = f"{base_path}corpora/mix_word{nb_of_units}.txt"
    word_group_file = f"{base_path}corpora/mixgroup_word{nb_of_units}.txt"

# --- Creating mixing file --- #

# Number of corpora
n_corpora = len(corpus_names_list)

# Loop for opening file
text_string_list = []
indices_list = []
for i, text_path in enumerate(text_path_list):
    with open(text_path, "r") as text_file:
        if unit_of_mix == "sent":
            # If unit_of_mix = "sent", keep only large enough sentences
            sentences_list = text_file.read().splitlines()
            sentences_to_keep_list = []
            indices_to_keep_list = []
            for sentence in sentences_list:
                sentence = re.sub(" +", " ", sentence).strip()
                sentence_length = len(sentence.split(" "))
                if sentence_length >= min_nb_of_words_in_sent:
                    sentences_to_keep_list.append(sentence)
                    indices_to_keep_list.append([i+1] * sentence_length)

            text_string_list.append(sentences_to_keep_list)
            indices_list.append(indices_to_keep_list)
        else:
            words_list = text_file.read().replace("\n", "").strip().split(" ")
            text_string_list.append(words_list)
            indices_list.append([i+1] * len(words_list))

# Size of the smallest text
min_size = int(min([len(text_string) for text_string in text_string_list]) / div_factor)
# Be sure we can have the desirable number of units
min_size = min_size // nb_of_units * nb_of_units
# Reduce to have list of same size
text_string_list = [text_string[0:min_size] for text_string in text_string_list]
indices_list = [indices[0:min_size] for indices in indices_list]

# Draw indices of corpora to mix
rdm_corpora_indices_list = list(np.repeat(list(range(n_corpora)), min_size // nb_of_units))
random.shuffle(rdm_corpora_indices_list)
rdm_corpora_indices_list = list(np.repeat(rdm_corpora_indices_list, nb_of_units))

# Making final text
final_text = ""
final_indices_list = []
for corpora_index in rdm_corpora_indices_list:
    final_text += text_string_list[corpora_index].pop(0) + " "
    if unit_of_mix == "sent":
        final_indices_list.extend(indices_list[corpora_index].pop(0))
    else:
        final_indices_list.append(indices_list[corpora_index].pop(0))

final_text = final_text.strip()

# Writing results
with open(output_file, "w") as text_file:
    text_file.write(final_text)
with open(word_group_file, "w") as text_file:
    text_file.write(str(final_indices_list)[1:-1])
