# Notes on python scripts

**text_processing.py** : This script take a raw text file and create two text files.

  - The first file (ending with \_tagged.txt) contains two columns, the first one with every token from the text file and the second the corresponding POS tag obtained by Flair library (https://github.com/flairNLP/flair).
  
  - The second file (ending with \_noun_only.txt) contains only nouns from the raw text file (tokens tagged with "NN", "NNS", "NNP" and "NNPS" by Flair). Each token is lemmatized, lower-cased and non-alpha characters are removed (compound words are splitted). Tokens are separated by a whitespace and sentences by "\n" (end of line).

