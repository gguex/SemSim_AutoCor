import os
import csv
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
file_name_list = ["61320_199211.csv",
                  "61320_200411.csv",
                  "61320_201211.csv",
                  "61320_201611.csv",
                  "61620_200411.csv",
                  "61620_200811.csv",
                  "61620_201211.csv",
                  "61620_201611.csv"]

# Loop on files
for file_name in file_name_list:
        # To store the final number of tokens
        n_token = 0
        # Loading the file
        with open(f"{base_path}/corpora/manifesto_csv_file/{file_name}", 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            sent_list = []
            group_list = []
            for row in csv_reader:
                try:
                    class_id = float(row[1])
                    class_id = class_id // 100
                    sent_list.append(row[0])
                    group_list.append(class_id)
                except:
                    pass


        # Setting dictionary for group labels
        translation_dic = {key: (i+1) for i, key in enumerate(set(group_list))}

        # Getting stopwords
        stop_words = stopwords.words('english')
        # Lemmatizer
        lemmatizer = WordNetLemmatizer()
        # Preprocessing each sentence
        with open(f"{base_path}/corpora/{file_name[:-4]}_pp.txt", "w") as text_file, \
            open(f"{base_path}/corpora/{file_name[:-4]}_pp_groups.txt", "w") as groups_file:

            for i, sent in enumerate(sent_list):

                # To lower case
                sent_pp = sent.lower()
                # Remove contractions
                sent_pp = contractions.fix(sent_pp)
                # Remove punctuation
                sent_pp = "".join([char for char in sent_pp if char in string.ascii_letters + " "])
                # Split by token
                token_list = nltk.word_tokenize(sent_pp)
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
                    n_token += len(token_list_pp)
                    for j, token in enumerate(token_list_pp):
                        text_file.write(f"{token} ")
                        groups_file.write(f"{translation_dic[group_list[i]]}, ")
                # Endline
                text_file.write("\n")

            groups_file.seek(groups_file.tell() - 2)
            groups_file.truncate()
            print(n_token)
