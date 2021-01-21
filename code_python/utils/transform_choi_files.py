import os
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import contractions

# Getting the os path
working_path=os.getcwd()

# Getting the SemSim_AutoCor folder, if above
base_path = str.split(working_path, "SemSim_AutoCor")[0] + "SemSim_AutoCor"

# Name of the file
file_name = "1/9-11/0.ref"

# Loading the file
with open(f"{base_path}/corpora/choi/{file_name}", 'r') as ref_file:
    text_string = ref_file.read()
    text_list = text_string.split("====\n")


# Getting stopwords
stop_words = stopwords.words('english')
# Lemmatizer
lemmatizer = WordNetLemmatizer()

# Preprocessing each sentence
with open(f"{base_path}/corpora/choi_pp.txt", "w") as text_file, \
        open(f"{base_path}/corpora/choi_pp_groups.txt", "w") as groups_file:
    group_id = 1
    for i, text_string in enumerate(text_list):

        # To lower case
        text_string_pp = text_string.lower()

        # Remove contractions
        text_string_pp = contractions.fix(text_string_pp)

        # Split by sentence
        sentence_list = nltk.sent_tokenize(text_string_pp)

        # Loop on sentences
        something_written = False
        for sentence in sentence_list:

            # Remove punctuation
            sentence_pp = "".join([char for char in sentence if char in string.ascii_letters + " "])

            # Split by token
            token_list = nltk.word_tokenize(sentence_pp)

            # Remove stopwords
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
            token_list_pp = [lemmatizer.lemmatize(*tagged_token) for tagged_token in wn_tagged_token_list]

            # If non_empty, saving to files
            if len(token_list_pp) > 0:
                for j, token in enumerate(token_list_pp):
                    text_file.write(f"{token} ")
                    groups_file.write(f"{group_id}, ")
                    something_written = True
        if something_written:
            group_id += 1

    groups_file.seek(groups_file.tell() - 2)
    groups_file.truncate()


