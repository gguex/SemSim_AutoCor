# Notes on python scripts

# Prepare:

python -m nltk.downloader all

## Main scripts

The pipeline for obtaining results take different steps :

1. The text pre-processing (1.1) which lowers cases, removes contractions/punctuation/stopwords and lemmatizes tokens. 
   The text mixing step (1.2) creates artificial datasets by mixing different pre-processed corpora, and a creates a 
   group file with true groups.
2. The computation of the similarity matrices between types from a pre-processed file, 
   along with the frequencies of each type (which is also useful for referencing the row and column of the 
   similarity matrix). (2.1) computes WordNet similarities, (2.2) Word Vectors similarities. The (2.2) script requires a copy of word vectors downloaded from GloVe or Wikipedia2Vec websites.
3. The computation of the global autocorrelation index from multiple text files and a similarity. Results are saved
   into a csv file and can be plotted with "utils/graph_from_data.py".
4. The computation of the local autocorrelation index (LISA) for every token from a text file and 
   a similarity matrix. Results are given with a plot of this index along the text, and with a html file coloring 
   tokens.
5. Different scripts using the fuzzy topic clustering algorithm to compute results. (5.1) makes k-fold 
   cross-validations on multiple file with a grid search on hyperparameters. (5.2) computes a unique result and compares
   it to a group file (ground truth). (5.3) computes a simple result on any file.

Along this pipeline, different functions are used, which are contained in the "local_functions.py" file.

The pipeline works with the help of tags to identify similarity. Moreover, the "corpora", "similarities_frequencies",
and "results" directories are hard-coded into some functions to ease the workflow.

## Utils

Contains utility 

**count_mean_sequence_length.py**: return the number of groups and the mean sequence length from group files.

**count_token_type.py**: return the number of tokens and types from a preprocessed file.

**graph_from_data.py**: plot the graph from the results of script "3_compute_autocor.py".

**transform_choi_files.py**: preprocesses files from the choi dataset in order to perform experiments.

**transform_maifesto_files.py**: preprocesses files from the manifesto dataset in order to perform experiments.

***transform_text_into_gensim_model.py**: This short script take a text word vector model
(from e.g. https://wikipedia2vec.github.io/wikipedia2vec/pretrained/) and transform it into a gensim model
(it's faster to load in a gensim format)

## Warnings

The similarities and typefreq files for the manifesto files are missing from the github page, because 
the similarities files were too big to be stored on github (>100Mo). They can be quickly obtained with (2.2) script.
