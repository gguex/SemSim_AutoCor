"""
make shure to use:
import nltk
nltk.download() # at lease wordnet, and wornet_ic
"""

import re
import os
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetError
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus import wordnet_ic
from collections import defaultdict
# Corpus name
corpus_names =  ("Animal_farm.txt", "The_WW_of_Oz.txt")
verbose = False
working_path = os.getcwd()
# --- Defining paths --- #



def compute_wn_similarities(working_path, corpus_name, verbose):
    base_path = str.split(working_path, "SemSim_AutoCor")[0] + "/SemSim_AutoCor/"
    # Path of the raw text file
    text_file_path = base_path + "corpora/" + corpus_name
    # Path of the inputted text file with two columns : token, POS tag
    tagged_input_path = base_path + "corpora/" + corpus_name[:-4] + "_tagged.txt"
    # Path of the inputted text file with only nouns
    noun_only_input_path = base_path + "corpora/" + corpus_name[:-4] + "_nouns.txt"
    verb_only_input_path = base_path + "corpora/" + corpus_name[:-4] + "_verbs.txt"
    adjective_only_input_path = base_path + "corpora/" + corpus_name[:-4] + "_adjectives.txt"
    adverb_only_input_path = base_path + "corpora/" + corpus_name[:-4] + "_adverbs.txt"


    corp = PlaintextCorpusReader(base_path + "corpora/", corpus_name)
    # ic = wn.ic(corp, False, 1.0) # does not work because the text may not contain ic for least common subsumer -> negative similarities
    brown_ic = wordnet_ic.ic('ic-brown.dat')
    # semcor_ic = wordnet_ic.ic('ic-semcor.dat')
    ic = brown_ic

    pos_to_path = {
        wn.NOUN: noun_only_input_path,
        wn.VERB: verb_only_input_path,
        # wn.ADJ: adjective_only_input_path,
        # wn.ADV: adverb_only_input_path
    }

    synsets = dict()
    frequencies = dict()
    total = 0.0
    for pos, path in pos_to_path.items():
        synsets[pos] = list()
        with open(path, "r") as token_file:
            tokens = token_file.read()
            tokens = re.sub("^\s+", "", tokens)
            tokens = re.sub("\s+$", "", tokens)
            tokens = sorted(set(re.sub(r"\s+", " ", tokens).split(" ")))
            frequencies[pos] = defaultdict(float)
            for t in tqdm(tokens):
                try:
                    freq = tokens.count(t)
                    morpheme = wn.morphy(t, pos)
                    if t:
                        synset = wn.synsets(morpheme, pos=pos)
                        i = 0
                        while synset[i].pos() != pos and i < len(synset):
                            i += 1
                        synset = synset[i]
                        synsets[pos].append(synset)
                        offset = synset.offset()
                        frequencies[pos][offset] += freq
                        total += freq
                except (WordNetError, IndexError, AttributeError) as e:
                    print('wn.synsets("', t, '", pos=\'', pos, "') : ", e, sep="")



    for pos, path in pos_to_path.items():
        synsets[pos] = sorted(set(synsets[pos]))
        combos = list(itertools.combinations_with_replacement(synsets[pos], 2))
        dists = list()
        for s1, s2 in tqdm(combos):
            try:
                dist = s1.res_similarity(s2, ic, verbose=verbose)
                if dist:
                    dists.append((s1.name(), s2.name(), dist))
            except WordNetError as e:
                print(s1, s2)
                print(e)
        if dists:
            data = pd.DataFrame().from_dict(dists)
            upper = data.pivot(index=0, columns=1, values=2)
            lower = data.pivot(index=1, columns=0, values=2)
            np.fill_diagonal(lower.values, 0.0)
            data = upper.fillna(0) + lower.fillna(0.0)
            data.to_csv(base_path + "similarities_frequencies/" + corpus_name[:-4] + "_" + pos_to_path[pos].split(corpus_name[:-4])[1][1:-4] + "_resnik_similarities.txt",
                        header=False, index=False, sep=";")


    for pos, path in pos_to_path.items():
        synsets[pos] = sorted(set(synsets[pos]))
        combos = list(itertools.combinations_with_replacement(synsets[pos], 2))
        dists = list()
        for s1, s2 in tqdm(combos):
            try:
                dist = s1.wup_similarity(s2, verbose=verbose)
                if dist:
                    dists.append((s1.name(), s2.name(), dist))
            except WordNetError as e:
                print(s1, s2)
                print(e)
        if dists:
            data = pd.DataFrame().from_dict(dists)
            upper = data.pivot(index=0, columns=1, values=2)
            lower = data.pivot(index=1, columns=0, values=2)
            np.fill_diagonal(lower.values, 0.0)
            data = upper.fillna(0) + lower.fillna(0.0)
            data.to_csv(base_path + "similarities_frequencies/" + corpus_name[:-4] + "_" + pos_to_path[pos].split(corpus_name[:-4])[1][1:-4] + "_wu-palmer_similarities.txt",
                        header=False, index=False, sep=";")


    for pos, path in pos_to_path.items():
        synsets[pos] = sorted(set(synsets[pos]))
        combos = list(itertools.combinations_with_replacement(synsets[pos], 2))
        dists = list()
        for s1, s2 in tqdm(combos):
            try:
                dist = s1.lch_similarity(s2, verbose=verbose)
                if dist:
                    dists.append((s1.name(), s2.name(), dist))
            except WordNetError as e:
                print(s1, s2)
                print(e)
        if dists:
            data = pd.DataFrame().from_dict(dists)
            upper = data.pivot(index=0, columns=1, values=2)
            lower = data.pivot(index=1, columns=0, values=2)
            np.fill_diagonal(lower.values, 0.0)
            data = upper.fillna(0) + lower.fillna(0.0)
            data.to_csv(base_path + "similarities_frequencies/" + corpus_name[:-4] + "_" + pos_to_path[pos].split(corpus_name[:-4])[1][1:-4] + "_leacock-chodorow_similarities.txt",
                        header=False, index=False, sep=";")


    for pos, path in pos_to_path.items():
        total = sum(frequencies[pos].values())
        freq = [(wn.synset_from_pos_and_offset(pos,k).lemma_names()[0], v/total) for k, v in frequencies[pos].items()]
        data = pd.DataFrame().from_dict(freq)
        data.to_csv(base_path + "similarities_frequencies/" + corpus_name[:-4] + "_" + pos_to_path[pos].split(corpus_name[:-4])[1][1:-4] + "_resnik_typefreq.txt",
                    header=False, index=False, sep=";")
        data.to_csv(base_path + "similarities_frequencies/" + corpus_name[:-4] + "_" + pos_to_path[pos].split(corpus_name[:-4])[1][1:-4] + "_wu-palmer_typefreq.txt",
                    header=False, index=False, sep=";")
        data.to_csv(base_path + "similarities_frequencies/" + corpus_name[:-4] + "_" + pos_to_path[pos].split(corpus_name[:-4])[1][1:-4] + "_leacock-chodorow_typefreq.txt",
                    header=False, index=False, sep=";")

    print(corpus_name)

# Getting the base path (must run the script from a folder inside the "SemSim_Autocor" folder)
for corpus_name in corpus_names:
    compute_wn_similarities(working_path, corpus_name, verbose)
