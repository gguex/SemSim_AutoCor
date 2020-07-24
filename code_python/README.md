# Notes on python scripts

## Main scripts

The pipeline for the computation of autocorrelation take 3 steps: 

1. The text processing to extract nouns, verbs, adjectives, and adverbs from a raw text file.
2. The computation of the similarity matrix between terms from a POS-selection file (containing only nouns, verbs, 
adjectives or adverbs) along with the frequencies of each terms.
3. The computation of the autocorrelation from a POS-selection file and a similarity matrix

The pipeline work with the help of extensions added to the end of the raw text file to ease the workflow.

**1_text_preprocessing.py** : This script take a raw text file and create five text files: One file containing every 
token with POS tag, and 4 POS-selection files.

- *INPUTS*: 

    - the *corpus_name*, as found in the "corpora" folder.
    
- *OUPUTS*:

    - The token-POStag file *<corpus_name>_tagged.txt* containing two columns: the first one with every token from the text file 
  and the second the corresponding POS tag obtained by Flair library (https://github.com/flairNLP/flair).
  
    - 4 POS-selection files, respectively, *<corpus_name>_nouns.txt*, *<corpus_name>_verbs.txt*, 
    *<corpus_name>_adjectives.txt*, and *<corpus_name>_adverbs.txt*, containing respectively only nouns, verbs, 
    adjectives, and adverbs from the corpus (tokens with tag beginning respectively by "NN", "VB", "JJ" and "RB" in 
    Flair). Each token is lemmatized, lower-cased and non-alpha characters are removed (compound words are split). 
    Tokens are separated by a whitespace and sentences by "\n" (end of line).

**2.1_build_word_vectors_similarities.py** : This script use the one of the POS-selection files from 
"1_text_preprocessing.py" and build the similarities and frequencies between terms coming from word embeddings.

- *INPUT*:

    - The *input_file*, i.e. the name of the POS-selection file to use, as found in the "corpora" folder.
    
    - The *wv_model_path*, i.e. the absolute path to a word vector model (see utils/transform_text_into_gensim.py), 
    outside the project director (the word vector file is too big to be stored here).
    
    - A *sim_tag*, i.e. a tag name for the similarity, currently set to "wesim" for this script.
    
- *OUTPUT*: (these files are stored in the "similarities_frequencies" folder)
    
    - The *<input_file>_<sim_tag>_typefreq.txt* containing two columns: the first one with every type contained in the 
    text file and which were also present in the word vectors, the second one the frequency of each type in the text 
    file. The order of the types in this file correspond to the order of row and column in the similarity matrix. 
    (separator = ";"). This file also help to know which terms appear both in the POS-selection file and in 
    the word vector model, i.e. which term possesses a similarity with the others.
  
    - The *<input_file>_<sim_tag>_similarities.txt* containing the n_type x n_type similarity matrix between each type 
    obtained from the word embedding. (separator = ";")
  
**3_compute_autocorrelation.py** : This script take a POS-selection file, the type-frequency file, and the similarity 
file to plot the autocorrelation index for a maximum range.

- *INPUTS*: 

    - The *input_file*, i.e. the name of the POS-selection file to use, as found in the "corpora" folder.
    
    - The *sim_tag*, i.e. the name of the tag for a similarity given during the creation of the similarity.
    
    - The *max_range* parameter, defining the maximum autocorrelation range to compute.

- *OUPUTS*:

    - The file <input_file>_<sim_tag>_autocor.png in the "results" folder, storing the graphic of the 
    autocorrelation index.

## Utils

Contains utility scripts

**transform_text_into_gensim_model.py** : This short script take a text word vector model 
(from e.g. https://wikipedia2vec.github.io/wikipedia2vec/pretrained/) and transform it into a gensim model 
(it's faster to load in a gensim format)

