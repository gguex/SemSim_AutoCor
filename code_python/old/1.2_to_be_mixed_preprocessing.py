import nltk
from flair.models import SequenceTagger
from flair.data import Sentence
import re
from tqdm import tqdm
import os
from gensim.models import KeyedVectors

# -------------------------------------
# --- Parameters
# -------------------------------------

# Corpus names
corpus_name_list = ["Lectures_on_Landscape.txt",
                    "Metamorphosis.txt",
                    "Civil_Disobedience.txt",
                    "Sidelights_on_relativity.txt"]

# -------------------------------------
# --- Computations
# -------------------------------------

# --- Defining paths --- #

# Getting the base path (must run the script from a folder inside the "SemSim_Autocor" folder)
working_path = os.getcwd()
base_path = str.split(working_path, "SemSim_AutoCor")[0] + "/SemSim_AutoCor/"

# Defining paths of the raw text files and outputted files
text_path_list = [base_path + "corpora/" + corpus_name for corpus_name in corpus_name_list]
all_output_path_list = [base_path + "corpora/" + corpus_name[:-4] + "_all.txt"
                        for corpus_name in corpus_name_list]

# Path of the Word vector model (absolute path, not in the project directory)
wv_model_path = "/home/gguex/Documents/data/pretrained_word_vectors/enwiki.model"

# Loading word vector vocab

# load gensim model
wv_wiki = KeyedVectors.load(wv_model_path)
# build vocabulary
vocab_wiki = set(wv_wiki.vocab.keys())

# --- POS tagging of the file --- #

# Loop on files
for i in range(len(corpus_name_list)):
    # Opening the file
    with open(text_path_list[i], "r") as text_file:
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

    # Saving the file with only NN, VB, JJ, and RB tag
    # One sentence by line
    lemmatizer = nltk.stem.WordNetLemmatizer()
    with open(all_output_path_list[i], "w") as all_file:
        for tagged_sentence in tqdm(tagged_sentence_list):
            nb_written_token = 0
            for token in tagged_sentence:
                pos = token.get_tag("pos").value[:2]
                if pos not in ("NN", "VB", "JJ", "RB"):
                    continue;
                if token.text.lower() not in vocab_wiki:
                    continue;
                processed_token = token.text.lower()
                processed_token = re.sub(r'[^\w\s\-]', "", processed_token)
                processed_token = re.sub(r'\-', " ", processed_token)
                processed_token = lemmatizer.lemmatize(processed_token)
                all_file.write(processed_token + " ")
                nb_written_token += 1
            if nb_written_token > 0:
                all_file.write("\n")
