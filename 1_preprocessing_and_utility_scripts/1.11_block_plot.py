import pandas as pd
import matplotlib.pyplot as plt

# Data loading
clust_s_res = pd.read_csv(f"results/3.6_block_results/block_clust_strong_61320_202011.csv")
clust_w_res = pd.read_csv(f"results/3.6_block_results/block_clust_weak_61320_202011.csv")
clust_w2_res = pd.read_csv(f"results/3.6_block_results/block_clust_weak2_61320_202011.csv")
clust_max = pd.read_csv(f"results/3.6_block_results/block_clust_none_61320_202011.csv")
clust_max2 = pd.read_csv(f"results/3.6_block_results/block_clust_none2_61320_202011.csv")
segm_s_res = pd.read_csv(f"results/3.6_block_results/block_segm_strong_61320_202011.csv")
segm_w_res = pd.read_csv(f"results/3.6_block_results/block_segm_weak_61320_202011.csv")
segm_w2_res = pd.read_csv(f"results/3.6_block_results/block_segm_weak2_61320_202011.csv")
segm_max = pd.read_csv(f"results/3.6_block_results/block_segm_none_61320_202011.csv")
segm_max2 = pd.read_csv(f"results/3.6_block_results/block_segm_none2_61320_202011.csv")

# Correction of None into number of tokens
n_max_token = 25870
clust_max.iloc[0, 1] = n_max_token
segm_max.iloc[0, 1] = n_max_token
clust_max2.iloc[0, 1] = n_max_token
segm_max2.iloc[0, 1] = n_max_token

# Time results
time_df = pd.concat([clust_s_res["elapse_time"], clust_w_res["elapse_time"], clust_w2_res["elapse_time"],
           segm_s_res["elapse_time"], segm_w_res["elapse_time"], segm_w2_res["elapse_time"]], axis=1)
time_mean = time_df.mean(axis=1)
time_sd = time_df.std(axis=1)

max_time_df = pd.concat([clust_max["elapse_time"], clust_max2["elapse_time"],
                         segm_max["elapse_time"], segm_max2["elapse_time"]], axis=1)
max_time_mean = max_time_df.mean(axis=1)
max_time_sd = max_time_df.std(axis=1)

plt.figure()
plt.grid()
plt.plot(clust_s_res["block_size"], time_mean, '-bo', color="black")
plt.plot(clust_s_res["block_size"], time_mean - 2.57*time_sd/(5**(1/2)), '--', color="black")
plt.plot(clust_s_res["block_size"], time_mean + 2.57*time_sd/(5**(1/2)), '--', color="black")
plt.plot()
plt.xlabel("Block size")
plt.ylabel("Computing time (sec)")
plt.axhline(y=float(max_time_mean), color="red")
plt.axhline(y=float(max_time_mean) - float(3.18*max_time_sd/(3**(1/2))), ls='--', color="red")
plt.axhline(y=float(max_time_mean) + float(3.18*max_time_sd/(3**(1/2))), ls='--', color="red")
plt.show()

# NMI results
nmi_df = pd.concat([clust_s_res["nmi"], clust_w_res["nmi"], clust_w2_res["nmi"]], axis=1)
nmi_mean = nmi_df.mean(axis=1)
nmi_sd = nmi_df.std(axis=1)

max_nmi_df = pd.concat([clust_max["nmi"], clust_max2["nmi"], pd.Series(0.132106)], axis=1)
max_nmi_mean = max_nmi_df.mean(axis=1)
max_nmi_sd = max_nmi_df.std(axis=1)

plt.figure()
plt.grid()
plt.plot(clust_s_res["block_size"], nmi_mean, '-bo', color="black")
plt.plot(clust_s_res["block_size"], nmi_mean - 4.3*nmi_sd/(2**(1/2)), '--', color="black")
plt.plot(clust_s_res["block_size"], nmi_mean + 4.5*nmi_sd/(2**(1/2)), '--', color="black")
plt.plot()
plt.xlabel("Block size")
plt.ylabel("NMI")
plt.axhline(y=float(max_nmi_mean), color="red")
plt.axhline(y=float(max_nmi_mean) - float(4.3*max_nmi_sd/(2**(1/2))), ls='--', color="red")
plt.axhline(y=float(max_nmi_mean) + float(4.3*max_nmi_sd/(2**(1/2))), ls='--', color="red")
plt.show()

# Pk results
pk_df = pd.concat([segm_s_res["pk"], segm_w_res["pk"], segm_w2_res["pk"]], axis=1)
pk_mean = pk_df.mean(axis=1)
pk_sd = pk_df.std(axis=1)

max_pk_df = pd.concat([segm_max["pk"], segm_max2["pk"], pd.Series(0.392931)], axis=1)
max_pk_mean = max_pk_df.mean(axis=1)
max_pk_sd = max_pk_df.std(axis=1)

plt.figure()
plt.grid()
plt.plot(clust_s_res["block_size"], pk_mean, '-bo', color="black")
plt.plot(clust_s_res["block_size"], pk_mean - 4.3*pk_sd/(2**(1/2)), '--', color="black")
plt.plot(clust_s_res["block_size"], pk_mean + 4.5*pk_sd/(2**(1/2)), '--', color="black")
plt.plot()
plt.xlabel("Block size")
plt.ylabel("Pk")
plt.axhline(y=float(max_pk_mean), color="red")
plt.axhline(y=float(max_pk_mean) - float(4.3*max_pk_sd/(2**(1/2))), ls='--', color="red")
plt.axhline(y=float(max_pk_mean) + float(4.3*max_pk_sd/(2**(1/2))), ls='--', color="red")
plt.show()