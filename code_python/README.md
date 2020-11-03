# Notes on python scripts

# Prepare:

python -m nltk.downloader all

## Main scripts

The pipeline for obtaining results take 2 preliminary scripts (1, 2) and 3 computation scripts (3, 4, 5):

1. The text processing to extract nouns, verbs, adjectives, and adverbs from a raw text file.
2. The computation of the similarity matrix between terms from a POS-selection file (containing only nouns, verbs,
adjectives or adverbs) along with the frequencies of each terms.
3. The computation of the autocorrelation from a POS-selection file and a similarity matrix.
4. The computation of the Local Indicator of Spatial Autocorrelation (LISA) on every token from a POS-selection file
and a similarity matrix.
5. The computation of a (n_token x n_group) membership matrix Z from a POS-selection file and a similarity matrix.

The pipeline works with the help of suffixes to ease the workflow.

**1_text_preprocessing.py** : This script take a raw text file and create five text files: One file containing every
token with POS tag, and 4 POS-selection files, and only token only file.

- *INPUTS*:

    - the *corpus_name*, as found in the "corpora" folder.

- *OUTPUTS*:

    - The token-POStag file *<corpus_name>_tagged.txt* containing two columns: the first one with every token from the text file
  and the second the corresponding POS tag obtained by Flair library (https://github.com/flairNLP/flair).
    - 4 POS-selection files, respectively, *<corpus_name>_nouns.txt*, *<corpus_name>_verbs.txt*,
    *<corpus_name>_adjectives.txt*, and *<corpus_name>_adverbs.txt*, containing respectively only nouns, verbs,
    adjectives, and adverbs from the corpus (tokens with tag beginning respectively by "NN", "VB", "JJ" and "RB" in
    Flair). Each token is lemmatized, lower-cased and non-alpha characters are removed (compound words are split).
    Tokens are separated by a whitespace and sentences by "\n" (end of line).
    - The file with all tokens *<corpus_name>_all.txt*, with the same structure found in the POS-selection files.

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

- *OUTPUTS*:

    - The file <input_file>_<sim_tag>_autocor.png in the "results" folder, storing the graphic of the
    autocorrelation index.

**4_compute_lisa.py** : This script take a POS-selection file and a similarity (by tag), and compute the Local
Indicator of Spatial Autocorrelation (LISA) on all token.

- *INPUTS*:

    - The *input_file*, i.e. the name of the POS-selection file to use, as found in a the "corpora" folder.
    - The *sim_tag*, i.e. the name of the tag for a similarity given during the creation of the similarity.
    - The *exch_mat_opt* option, i.e. either "u" or "d", for respectively an uniform or diffusive exchange matrix.
    - The *exch_range* parameter, i.e. the range of the exchange matrix (for the uniform exchange matrix) or
    the time parameter (for the diffusive exchange matrix).

- *OUTPUTS*:

    - The file <input_file>_<sim_tag>_lisa<lisa_range>.png, showing the LISA index for every token.
    - The file <input_file>_<sim_tag>_lisa<lisa_range>.html, coloring every token depending its LISA index
    (green = positive, red = negative).

**5.1_discontinuity_segment_token.py** : This script take a POS-selection file and a similarity (by tag),
and compute a (n_token x n_group) membership matrix Z, which softy assign each token to a group (discontinuity method).

- *INPUTS*:

    - The *input_file*, i.e. the name of the POS-selection file to use, as found in a the "corpora" folder.
    - The *sim_tag*, i.e. the name of the tag for a similarity given during the creation of the similarity.
    - The *exch_mat_opt* option, i.e. either "u" or "d", for respectively an uniform or diffusive exchange matrix.
    - The *exch_range* parameter, i.e. the range of the exchange matrix (for the uniform exchange matrix) or
    the time parameter (for the diffusive exchange matrix).
    - The *n_groups*, i.e. the number of groups.
    - The *alpha*, *beta* and *kappa* hyper-parameters, which are defined in article.
    - The *conv_threshold* parameter, defining when iterations have to stop.
    - The *max_it* parameter, defining the maximum iterations possible.

- *OUTPUTS*:

    - The file <input_file>_<sim_tag>_discsegm.csv, containing every token ordered by decreasing membership value for
    each groups.
    - The file <input_file>_<sim_tag>_discsegm.html, coloring every token according to its membership values.

**5.2_cut_segment_token.py** : This script take a POS-selection file and a similarity (by tag),
and compute a (n_token x n_group) membership matrix Z, which softy assign each token to a group (cut method).

- *INPUTS*:

    - The *input_file*, i.e. the name of the POS-selection file to use, as found in a the "corpora" folder.
    - The *sim_tag*, i.e. the name of the tag for a similarity given during the creation of the similarity.
    - The *exch_mat_opt* option, i.e. either "u" or "d", for respectively an uniform or diffusive exchange matrix.
    - The *exch_range* parameter, i.e. the range of the exchange matrix (for the uniform exchange matrix) or
    the time parameter (for the diffusive exchange matrix).
    - The *n_groups*, i.e. the number of groups.
    - The *gamma*, *beta* and *kappa* hyper-parameters, which are defined in article.
    - The *conv_threshold* parameter, defining when iterations have to stop.
    - The *max_it* parameter, defining the maximum iterations possible.

- *OUTPUTS*:

    - The file <input_file>_<sim_tag>_cutsegm.csv, containing every token ordered by decreasing membership value for
    each groups.
    - The file <input_file>_<sim_tag>_cutsegm.html, coloring every token according to its membership values.

## Utils

Contains utility scripts

**transform_text_into_gensim_model.py** : This short script take a text word vector model
(from e.g. https://wikipedia2vec.github.io/wikipedia2vec/pretrained/) and transform it into a gensim model
(it's faster to load in a gensim format)

