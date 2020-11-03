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
corpus_name = "The_WW_of_Oz_disambiguated.txt"

# --- Defining paths --- #

# Getting the base path (must run the script from a folder inside the "SemSim_Autocor" folder)
working_path = os.getcwd()
base_path = str.split(working_path, "SemSim_AutoCor")[0] + "/SemSim_AutoCor/"

# Path of the raw text file
text_file_path = base_path + "corpora/" + corpus_name


corp = PlaintextCorpusReader(base_path + "corpora/", 'The_WW_of_Oz.txt')
ic = wn.ic(corp, False, 0.0)

#get  sense keys like : |live%2:42:08::
sense_key_regex = re.compile(r"(?:\|)(\w+%\d:\d{2}:\d{2}::)")

with open(text_file_path, "r") as text:
    txt = text.read()
    print(txt)
    sense_keys = sense_key_regex.findall(txt)

synsets = list()
errors = list()
for sk in sense_keys:
    try:
        synset = wn.synset_from_sense_key(sk)
        if synset:
            synsets.append(synset)
    except WordNetError as e:
        errors.append((sk, e))


total = len(synsets)
freq = [(synset, synsets.count(s) / total) for s in synsets]
data = pd.DataFrame().from_dict(freq)
data.to_csv(base_path + "similarities/typefreq_" + re.sub("^.*corpora[/]", "", path[:-4]) + ".csv",
            header=False, index=False)

pos_to_synsets = {
    'n': list(),
    'v': list(),
    'a': list(),
    'r': list(),
    's': list()
}
for synset in set(synsets):
    pos_to_synsets[synset.pos()].append(synset)

for pos, synsets in pos_to_synsets:
    combos = list(itertools.combinations_with_replacement(synsets, 2))
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

for pos, synsets in pos_to_synsets:
    combos = list(itertools.combinations_with_replacement(synsets, 2))
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


for pos, synsets in pos_to_synsets:
    combos = list(itertools.combinations_with_replacement(synsets, 2))
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


for pos, synsets in pos_to_synsets:
    combos = list(itertools.combinations_with_replacement(synsets, 2))
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



