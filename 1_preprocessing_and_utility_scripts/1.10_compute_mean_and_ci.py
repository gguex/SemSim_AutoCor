import pandas as pd
from scipy.stats import sem, t

path_to_file = "results/3.2_segm_results/segm_manifesto.csv"
col_to_compute = "wd"

# Computations

dataframe = pd.read_csv(path_to_file)

mean = dataframe[col_to_compute].mean()
diff = sem(dataframe[col_to_compute]) * t.ppf((1 + 0.95) / 2, len(dataframe[col_to_compute]) - 1)

print(f"Mean = {mean} +/- {diff}")