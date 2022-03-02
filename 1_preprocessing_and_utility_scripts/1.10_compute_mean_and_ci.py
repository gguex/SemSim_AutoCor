import pandas as pd
from scipy.stats import sem, t

path_to_file = "results/3.6_block_results/clust_b10000_strong.csv"
col_to_compute = "nmi"

# Computations

dataframe = pd.read_csv(path_to_file)

mean = dataframe[col_to_compute].mean()
diff = sem(dataframe[col_to_compute]) * t.ppf((1 + 0.95) / 2, len(dataframe[col_to_compute]) - 1)

print(f"Mean = {mean} +/- {diff}")