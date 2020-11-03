import re
import os
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetError
from nltk.corpus.reader.plaintext import PlaintextCorpusReader

from collections import defaultdict
# Corpus name
corpus_name = "The_WW_of_Oz.txt"

# --- Defining paths --- #

# Getting the base path (must run the script from a folder inside the "SemSim_Autocor" folder)
working_path = os.getcwd()
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


corp = PlaintextCorpusReader(base_path + "corpora/", 'The_WW_of_Oz.txt')
ic = wn.ic(corp, False, 0.0)

pos_to_path = {
    wn.NOUN: noun_only_input_path,
    wn.VERB: verb_only_input_path,
    wn.ADJ: adjective_only_input_path,
    wn.ADV: adverb_only_input_path
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
                mopheme = wn.morphy(t, pos)
                if t:
                    synset = wn.synsets(mopheme, pos=pos)
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
            dist = s1.res_similarity(s2, ic)
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
        data.to_csv(base_path + "similarities/res_similarity" + re.sub("^.*corpora", "", path[:-4]) + ".csv",
                    header=False, index=False)


for pos, path in pos_to_path.items():
    synsets[pos] = sorted(set(synsets[pos]))
    combos = list(itertools.combinations_with_replacement(synsets[pos], 2))
    dists = list()
    for s1, s2 in tqdm(combos):
        try:
            dist = s1.res_similarity(s2, ic)
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
        data.to_csv(base_path + "similarities/res_similarity" + re.sub("^.*corpora", "", path[:-4]) + ".csv",
                    header=False, index=False)


for pos, path in pos_to_path.items():
    synsets[pos] = sorted(set(synsets[pos]))
    combos = list(itertools.combinations_with_replacement(synsets[pos], 2))
    dists = list()
    for s1, s2 in tqdm(combos):
        try:
            dist = s1.wup_similarity(s2)
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
        data.to_csv(base_path + "similarities/wup_similarity" + re.sub("^.*corpora", "", path[:-4]) + ".csv",
                    header=False, index=False)


for pos, path in pos_to_path.items():
    synsets[pos] = sorted(set(synsets[pos]))
    combos = list(itertools.combinations_with_replacement(synsets[pos], 2))
    dists = list()
    for s1, s2 in tqdm(combos):
        try:
            dist = s1.lch_similarity(s2)
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
        data.to_csv(base_path + "similarities/lch_similarity" + re.sub("^.*corpora", "", path[:-4]) + ".csv",
                    header=False, index=False)


for pos, path in pos_to_path.items():
    total = sum(frequencies[pos].values())
    freq = [(wn.synset_from_pos_and_offset(pos,k).name(), v/total) for k, v in frequencies[pos].items()]
    data = pd.DataFrame().from_dict(freq)
    data.to_csv(base_path + "similarities/typefreq_" + re.sub("^.*corpora[/]", "", path[:-4]) + ".csv",
                header=False, index=False)
