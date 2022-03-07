import pandas as pd
import matplotlib.pyplot as plt

type = "strong"


# Data loading
clust_res = pd.read_csv(f"results/3.6_block_results/block_clust_{type}_61320_202011.csv")
clust_max = pd.read_csv(f"results/3.6_block_results/block_clust_none_61320_202011.csv")
segm_res = pd.read_csv(f"results/3.6_block_results/block_segm_{type}_61320_202011.csv")
segm_max = pd.read_csv(f"results/3.6_block_results/block_segm_none_61320_202011.csv")

# Correction of None into numbre of token
n_max_token = 25870
clust_max.iloc[0, 1] = n_max_token
segm_max.iloc[0, 1] = n_max_token

# Merging datasets
clust_df = pd.concat((clust_res, clust_max))
segm_df = pd.concat((segm_res, segm_max))

plt.figure()
plt.grid()
plt.plot(clust_res["block_size"], clust_res["elapse_time"], '--bo')
plt.xlabel("Block size")
plt.ylabel("Computing time (sec)")
plt.axhline(y=float(clust_max["elapse_time"]), color="red")
plt.show()

plt.figure()
plt.grid()
plt.plot(clust_res["block_size"], clust_res["nmi"], '--bo')
plt.xlabel("Block size")
plt.ylabel("NMI")
plt.axhline(y=float(clust_max["nmi"]), color="red")
plt.show()

plt.figure()
plt.grid()
plt.plot(segm_res["block_size"], segm_res["pk"], '--bo')
plt.xlabel("Block size")
plt.ylabel("Pk")
plt.axhline(y=float(segm_max["pk"]), color="red")
plt.show()