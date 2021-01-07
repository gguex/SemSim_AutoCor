import os
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet

# -------------------------------------
# --- Parameters
# -------------------------------------

# Corpus name
corpus_name = "Sidelights_on_relativity.txt"

# -------------------------------------
# --- Computations
# -------------------------------------

# --- Defining paths --- #

# Getting the base path (must run the script from a folder inside the "SemSim_Autocor" folder)
working_path = os.getcwd()
base_path = str.split(working_path, "SemSim_AutoCor")[0] + "/SemSim_AutoCor/"

# Path of the raw text file
text_file_path = base_path + "corpora/" + corpus_name
# Path of the outputted preprocessed text file
pp_output_path = base_path + "corpora/" + corpus_name[:-4] + "_pp.txt"

# --- Loading and Preprocessing --- #

# Opening the file
with open(text_file_path, "r") as text_file:
    text_string = text_file.read()

# To lower case
text_string_pp = text_string.lower()

# Remove punctuation
text_string_pp = "".join([char for char in text_string_pp if char not in string.punctuation])

# Split by token
token_list = nltk.word_tokenize(text_string_pp)

# Remove stopwords
stop_words = stopwords.words('english')
token_list = [token for token in token_list if token not in stop_words]

# POS tag
tagged_token_list = nltk.pos_tag(token_list)

# Changing POS tag to wordnet tag
wn_tagged_token_list = []
for tagged_token in tagged_token_list:
    if tagged_token[1].startswith('J'):
        wn_tagged_token_list.append((tagged_token[0], wordnet.ADJ))
    elif tagged_token[1].startswith('V'):
        wn_tagged_token_list.append((tagged_token[0], wordnet.VERB))
    elif tagged_token[1].startswith('N'):
        wn_tagged_token_list.append((tagged_token[0], wordnet.NOUN))
    elif tagged_token[1].startswith('R'):
        wn_tagged_token_list.append((tagged_token[0], wordnet.ADV))
    else:
        wn_tagged_token_list.append([tagged_token[0]])

# Lemmatization
lemmatizer = WordNetLemmatizer()
token_list_pp = [lemmatizer.lemmatize(*tagged_token) for tagged_token in wn_tagged_token_list]

# --- SAVING FILE --- #

# Saving the file with all tokens and tags
with open(pp_output_path, "w") as output_file:
    for token in token_list_pp:
        output_file.write(f"{token} ")