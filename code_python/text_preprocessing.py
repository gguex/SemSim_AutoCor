import nltk
from flair.models import SequenceTagger
from flair.data import Sentence
import re

# --- Parameters --- #

# Path of the raw text file
text_file_path = "/home/gguex/Documents/data/corpora/pg43936-The_Wonderful_Wizard_of_Oz.txt"
# Path of the outputted text file with two columns : token, POS tag
tagged_output_path = "/home/gguex/Documents/data/corpora/The_Wonderful_Wizard_of_Oz_tagged.txt"
# Path of the outputted text file with only nouns
noun_only_output_path = "/home/gguex/Documents/data/corpora/The_Wonderful_Wizard_of_Oz_noun_only.txt"

# --- POS tagging of the file --- #

# Opening the file
with open(text_file_path, "r") as text_file:
    text_string = text_file.read()

# Split by sentences
sentence_list = nltk.sent_tokenize(text_string)

# Using flair POS tagging on each sentence
tagger = SequenceTagger.load("pos")
tagged_sentence_list = []
for sentence in sentence_list:
    sent = Sentence(sentence.replace("\n", " "))
    tagger.predict(sent)
    tagged_sentence_list.append(sent)

# --- SAVING FILES --- #

# Saving the file with all tokens and tags
with open(tagged_output_path, "w") as output_tagged_file:
    output_tagged_file.write("token\ttag\n")
    for tagged_sentence in tagged_sentence_list:
        for token in tagged_sentence:
            output_tagged_file.write(token.text + "\t" + token.get_tag("pos").value + "\n")

# Saving the file with only nouns
allowed_tag = ["NN", "NNS", "NNP", "NNPS"]
lemmatizer = nltk.stem.WordNetLemmatizer()
with open(noun_only_output_path, "w") as noun_only_file:
    for tagged_sentence in tagged_sentence_list:
        for token in tagged_sentence:
            if token.get_tag("pos").value in allowed_tag:
                processed_token = token.text.lower()
                processed_token = re.sub(r'[^\w\s\-]', "", processed_token)
                processed_token = re.sub(r'\-', " ", processed_token)
                processed_token = lemmatizer.lemmatize(processed_token)
                noun_only_file.write(processed_token + " ")
        noun_only_file.write("\n")
