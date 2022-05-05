from sentence_transformers import SentenceTransformer, util
from local_functions import *
import pandas as pd

# Loading the corpus
democrat_1992 = pd.read_csv("Democrat_1992.csv")
# Saving the sentences
sentences = list(democrat_1992["text"])

# Load sentence model
sbert_model = SentenceTransformer("all-mpnet-base-v2")
# Make the sentence vectors
sentence_embeddings = sbert_model.encode(sentences)
# Make similarity matrix
sim_mat = np.array(util.pytorch_cos_sim(sentence_embeddings, sentence_embeddings))

# Compute the dissimilarity matrix
d_mat = similarity_to_dissimilarity(sim_mat)
# Compute the exchange and transition matrices
exch_mat, w_mat = exchange_and_transition_matrices(len(sentences), exch_mat_opt="u", exch_range=5)
# Compute the membership matrix
membership_matrix = spatial_clustering(d_ext_mat=d_mat, exch_mat=exch_mat, w_mat=w_mat, n_groups=6,
                                       alpha=10, beta=100, kappa=0.75, verbose=True)

# Write html results
write_groups_in_html_file("Democrat_1992_results.html", sentences, membership_matrix)