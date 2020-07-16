# Notes on python scripts

**text_processing.py** : This script take a raw text file and create two text files.
  - The first file (\_tagged.txt) contains two columns, the first one with every token from the text file and the second the POS tagging obtained by Flair library.
  - The second file (\_noun_only.txt) contains only the noun from the text file (tokens tagged with "NN", "NNS", "NNP" and "NNPS"). Each token is lemmatized, lower cases, punctuation is removed (composed word separaed by "-" are splitted) and are separated by a whitespace. Each sentence is separated by endofline.

