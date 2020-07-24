import nltk
from flair.models import SequenceTagger
from flair.data import Sentence
import re
from tqdm import tqdm
import os

# --- Parameters --- #

# Corpus name
corpus_name = "The_WW_of_Oz.txt"

# --- Defining paths --- #

# Getting the base path (must run the script from a folder inside the "SemSim_Autocor" folder)
working_path = os.getcwd()
base_path = str.split(working_path, "SemSim_AutoCor")[0] + "SemSim_AutoCor/"

# Path of the raw text file
text_file_path = base_path + "corpora/" + corpus_name
# Path of the outputted text file with two columns : token, POS tag
tagged_output_path = base_path + "corpora/" + corpus_name[:-4] + "_tagged.txt"
# Path of the outputted text file with only nouns
noun_only_output_path = base_path + "corpora/" + corpus_name[:-4] + "_nouns.txt"
verb_only_output_path = base_path + "corpora/" + corpus_name[:-4] + "_verbs.txt"
adjective_only_output_path = base_path + "corpora/" + corpus_name[:-4] + "_adjectives.txt"
adverb_only_output_path = base_path + "corpora/" + corpus_name[:-4] + "_adverbs.txt"

# --- POS tagging of the file --- #

# Opening the file
with open(text_file_path, "r") as text_file:
    text_string = text_file.read()

# Split by sentences
sentence_list = nltk.sent_tokenize(text_string)

# Using flair POS tagging on each sentence
tagger = SequenceTagger.load("pos")
tagged_sentence_list = []
for sentence in tqdm(sentence_list):
    sent = Sentence(sentence.replace("\n", " "))
    tagger.predict(sent)
    tagged_sentence_list.append(sent)

# --- SAVING FILES --- #

# Saving the file with all tokens and tags
with open(tagged_output_path, "w") as output_tagged_file:
    output_tagged_file.write("token\ttag\n")
    for tagged_sentence in tqdm(tagged_sentence_list):
        for token in tagged_sentence:
            output_tagged_file.write(token.text + "\t" + token.get_tag("pos").value + "\n")

# Saving the file with one type of part of speach
#
# NN <- nouns
# VB <- verbs
# JJ <- adjectives
# RB <- adverbs

lemmatizer = nltk.stem.WordNetLemmatizer()
with open(noun_only_output_path, "w") as noun_only_file, \
        open(verb_only_output_path, "w") as verb_only_file, \
        open(adjective_only_output_path, "w") as adjective_only_file, \
        open(adverb_only_output_path, "w") as adverb_only_file:
    for tagged_sentence in tqdm(tagged_sentence_list):
        for token in tagged_sentence:
            pos = token.get_tag("pos").value[:2]
            if pos not in ("NN", "VB", "JJ", "RB"):
                continue;
            processed_token = token.text.lower()
            processed_token = re.sub(r'[^\w\s\-]', "", processed_token)
            processed_token = re.sub(r'\-', " ", processed_token)
            processed_token = lemmatizer.lemmatize(processed_token)
            if pos == "NN":
                noun_only_file.write(processed_token + " ")
            elif pos == "VB":
                verb_only_file.write(processed_token + " ")
            elif pos == "JJ":
                adjective_only_file.write(processed_token + " ")
            else:
                adverb_only_file.write(processed_token + " ")
        noun_only_file.write("\n")
        verb_only_file.write("\n")
        adjective_only_file.write("\n")
        adverb_only_file.write("\n")
