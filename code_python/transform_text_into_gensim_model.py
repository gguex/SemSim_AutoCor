from gensim.models import KeyedVectors
from gensim.test.utils import datapath

# --- Parameters --- #

# Path of the input text file (from e.g. https://wikipedia2vec.github.io/wikipedia2vec/pretrained/)
input_text_path = "/home/gguex/Documents/data/pretrained_word_vectors/enwiki_20180420_300d.txt"
# Path of outputted Gensim model
output_model_path = "/home/gguex/Documents/data/pretrained_word_vectors/enwiki.model"

# --- Model creation --- #

wv_from_text = KeyedVectors.load_word2vec_format(datapath(input_text_path), binary=False)
wv_from_text.save(output_model_path)
