import nltk
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from code_python.local_functions import get_all_paths
from itertools import product
import multiprocessing as mp
from miniutils import parallel_progbar

# -------------------------------------
# --- Parameters
# -------------------------------------

# List of paths for text files to compute similarity
# input_file_list = ["mix_word1.txt",
#                    "mix_word5.txt",
#                    "mix_sent1.txt",
#                    "mix_sent5.txt"]
input_file_list = ["mix_sent1.txt"]

# List of tags to enumerate similarities to compute
# sim_tag_list = ["lch", "path", "wup"]
sim_tag_list = ["lch"]


# Number of cpus to use
n_cpu = mp.cpu_count() - 1

# Threshold for minimum similarity value
# (a type is dropped if its maximum similarity with other types doesn't reach this threshold)
sim_threshold = 1e-3

# -------------------------------------
# --- Computations
# -------------------------------------

# Loading brown corpus information
brown_ic = wordnet_ic.ic('ic-brown.dat')

# Loop on files and tags
for input_file in input_file_list:
    for sim_tag in sim_tag_list:

        # Print
        print(f"Computing similarity {sim_tag} for corpus {input_file}")

        # Defining similarity
        if sim_tag == "wup":
            def wn_similarity(synset_1, synset_2):
                return wn.wup_similarity(synset_1, synset_2)
        elif sim_tag == "lch":
            def wn_similarity(synset_1, synset_2):
                return wn.lch_similarity(synset_1, synset_2)
        elif sim_tag == "path":
            def wn_similarity(synset_1, synset_2):
                return wn.path_similarity(synset_1, synset_2)
        elif sim_tag == "resb":
            def wn_similarity(synset_1, synset_2):
                if synset_1.pos() not in ["a", "s", "r"] and synset_2.pos() not in ["a", "s", "r"]:
                    return wn.res_similarity(synset_1, synset_2, brown_ic)
                else:
                    return None
        else:
            def wn_similarity(synset_1, synset_2):
                if synset_1.pos() not in ["a", "s", "r"] and synset_2.pos() not in ["a", "s", "r"]:
                    return wn.jcn_similarity(synset_1, synset_2, brown_ic)
                else:
                    return None

        # Getting all paths
        file_path, type_freq_file_path, sim_matrix_file_path, _ = get_all_paths(input_file, sim_tag, warn=False)

        # Opening the file
        with open(file_path, "r") as text_file:
            text_string = text_file.read()

        # Split by tokens
        token_list = nltk.word_tokenize(text_string)

        # Get type list and frequencies
        type_freq_dict = nltk.FreqDist(token_list)
        vocab_text = set(type_freq_dict.keys())

        # build vocabulary
        vocab_in_wordnet = [word for word in vocab_text if len(wn.synsets(word)) > 0]

        # Build autosim to check if synset is valid
        auto_sim_list = []
        checked_vocab = []
        for word in vocab_in_wordnet:
            type_synset_list = wn.synsets(word)
            sim_list = [wn_similarity(type_synset, type_synset) for type_synset in type_synset_list
                        if wn_similarity(type_synset, type_synset) is not None]
            if len(sim_list) > 0:
                auto_sim_list.append(max(sim_list))
                checked_vocab.append(word)
        n_type = len(checked_vocab)

        # Computing the similarity matrix
        sim_mat = np.zeros((n_type, n_type))
        # Vector for index of type with existing similarities
        ok_sim_index_list = []

        # Define compute col function
        def compute_col(i):
            type_1_synset_list = wn.synsets(checked_vocab[i])
            sim_col = np.repeat(0, i)
            for j in range(i + 1, n_type):
                type_2_synset_list = wn.synsets(checked_vocab[j])
                sim_list = []
                for cross_item in product(type_1_synset_list, type_2_synset_list):
                    try:
                        sim_1 = wn_similarity(cross_item[0], cross_item[1])
                    except:
                        sim_1 = 0
                    try:
                        sim_2 = wn_similarity(cross_item[1], cross_item[0])
                    except:
                        sim_2 = 0
                    if sim_1 is None:
                        sim_1 = 0
                    if sim_2 is None:
                        sim_2 = 0
                    sim_list.append(max(sim_1, sim_2))
                if len(sim_list) > 0:
                    sim = max(sim_list)
                else:
                    sim = 0
                sim_col = np.append(sim_col, sim)
            return sim_col

        # Multiprocess
        sim_mat = np.array(parallel_progbar(compute_col, range(n_type), nprocs=n_cpu))

        # Compute the final matrix
        np.fill_diagonal(sim_mat, auto_sim_list)

        # Write the similarity
        with open(sim_matrix_file_path, "w") as sim_matrix_file:
            np.savetxt(sim_matrix_file, sim_mat, delimiter=";", fmt="%.8f")
        # Write the words
        with open(type_freq_file_path, "w") as type_freq_file:
            for type in checked_vocab:
                type_freq_file.write(f"{type};{type_freq_dict[type]}\n")
