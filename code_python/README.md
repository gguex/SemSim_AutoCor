# Notes on python scripts

**build_word_vectors_similarities.py** : This script use the "noun only" file and build two outputted files :
  - The first file (ending with \_typefreq.txt) contains two columns, the first one with every type contained in the text file and which were also present in the word vectors, the second one the frequency of each type in the text file. The order of the types in this file correspond to the order of row and column in the similarity matrix. (separator = ";")
  - The second file (ending with \_similarities.txt) contains the n_type x n_type similarity matrix between each type obtained from the word embedding. (separator = ";")
  
**compute_autocorrelation.py** : This script take the "noun only" file, the type-frequency file, and the similarity file to compute the autocorrelation index for a certain range. (MIGHT STILL BE FIXED)

**text_preprocessing.py** : This script take a raw text file and create five text files.

  - The first file (ending with \_tagged.txt) contains two columns, the first one with every token from the text file and the second the corresponding POS tag obtained by Flair library (https://github.com/flairNLP/flair).
  
  - The 4 other files (ending with \_nouns.txt, \_verbes.txt, \_adjectives.txt, \_adverbs.txt) contains respectively only nouns, only adjectives, only adverbs and only verbs from the raw text file (tokens with tag beginning respectively by "NN", "VB", "JJ" and "RB" by Flair). Each token is lemmatized, lower-cased and non-alpha characters are removed (compound words are splitted). Tokens are separated by a whitespace and sentences by "\n" (end of line).

**transform_text_into_gensim_model.py** : This short script take a text word vector model (from e.g. https://wikipedia2vec.github.io/wikipedia2vec/pretrained/) and transform it into a gensim model (it's faster to load in a gensim format)

