import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from code_python.local_functions import get_all_paths
from tqdm import tqdm
import itertools

# -------------------------------------
# --- Parameters
# -------------------------------------

# List of paths for text files to compute similarity
input_file_list = ["Metamorphosis_pp.txt",
                   "Civil_Disobedience_pp.txt",
                   "Sidelights_on_relativity_pp.txt"]

# List of tags to enumerate similarities to compute
sim_tag_list = ["wup", "path", "resb"]

# -------------------------------------
# --- Computations
# -------------------------------------

# Loading brown corpus information
brown_ic = wordnet_ic.ic('ic-brown.dat')

# Loop on files and tags
for input_file in input_file_list:
    for sim_tag in sim_tag_list:

        # Print which is computed
        print(f"Computing similarity {sim_tag} for corpus {input_file}")

        # Defining similarity
        if sim_tag == "wup":
            def wn_similarity(synset_1, synset_2):
                return wn.wup_similarity(synset_1, synset_2)
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

        # Build autosim to check if any synset is connected
        auto_sim_list = []
        checked_vocab_in_wordnet = []
        for word in vocab_in_wordnet:
            type_synsets_list = wn.synsets(word)
            sim_list = [wn_similarity(type_synsets, type_synsets) for type_synsets in type_synsets_list
                        if wn_similarity(type_synsets, type_synsets) is not None]
            if len(sim_list) > 0:
                auto_sim_list.append(max(sim_list))
                checked_vocab_in_wordnet.append(word)

        # Write the two files
        with open(type_freq_file_path, "w") as type_freq_file, open(sim_matrix_file_path, "w") as sim_matrix_file:
            for type_1 in tqdm(checked_vocab_in_wordnet):

                type_1_synsets_list = wn.synsets(type_1)
                type_freq_file.write(type_1 + ";" + str(type_freq_dict[type_1]) + "\n")

                for i, type_2 in enumerate(checked_vocab_in_wordnet):

                    if type_2 != type_1:
                        # Loop on synsets
                        type_2_synsets_list = wn.synsets(type_2)
                        sim_list = [wn_similarity(*cross_item)
                                    for cross_item in itertools.product(type_1_synsets_list, type_2_synsets_list)
                                    if cross_item[0].pos() == cross_item[1].pos() and
                                    wn_similarity(*cross_item) is not None]
                        if len(sim_list) > 0:
                            sim = max(sim_list)
                        else:
                            sim = 0
                    else:
                        sim = auto_sim_list[i]

                    # Write the similarity
                    sim_matrix_file.write(str(sim))
                    if type_2 != checked_vocab_in_wordnet[len(checked_vocab_in_wordnet) - 1]:
                        sim_matrix_file.write(";")
                    else:
                        sim_matrix_file.write("\n")
