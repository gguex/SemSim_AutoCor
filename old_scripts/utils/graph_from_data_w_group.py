from itertools import cycle
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import csv
import os
import colorsys

# CSV file list
csv_file_name_list = ["3.2_pplautocor50_w2v.csv"]
# Group list
group_list = [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4
# Max range
max_range = 50

# Computations
# Working path
working_path = os.getcwd()
# Getting the SemSim_AutoCor folder, if above
base_path = str.split(working_path, "SemSim_AutoCor")[0] + "SemSim_AutoCor"

# Loop on csv file names
for csv_file_name in csv_file_name_list:

    # Sim tag name
    sim_tag = csv_file_name[-7:-4]

    # File name path
    with open(f"{base_path}/results/{csv_file_name}", 'r') as typefreq_file:
        csv_reader = csv.reader(typefreq_file)
        data_list = [row for row in csv_reader]

    input_file_list = []
    autocor_vec_list = []
    exch_range_window = [0]
    for i, data_item in enumerate(data_list):
        if i == 0:
            exch_range_window = list(map(int, data_item[1:(max_range+1)]))
        else:
            input_file_list.append(data_item[0])
            autocor_vec_list.append(list(map(float, data_item[1:(max_range+1)])))

    # line cycler
    line_cycler = cycle(["-", "--", "-.", ":"])

    # Set important p-value
    percent_list = np.array([0.5, 0.75, 0.95, 0.99, 0.999])
    quant_list = norm.ppf(percent_list)

    # Set color
    n_groups = len(set(group_list))
    # Creating group colors
    color_rgb_list = []
    for i in range(n_groups):
        color_rgb_list.append(np.array(colorsys.hsv_to_rgb(i * 1 / n_groups, 1, 1)))
    color_rgb_mat = np.array(color_rgb_list)

    # Plot the autocor vector
    plt.figure("Autocorrelation")

    for i, input_file in enumerate(input_file_list):
        plt.plot(exch_range_window, autocor_vec_list[i], next(line_cycler), label=input_file,
                 color=color_rgb_list[group_list[i]])

    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    plt.autoscale(False)

    for i, quant in enumerate(quant_list):
        if quant > ylim[0]:
            plt.plot(exch_range_window, np.repeat(quant, len(exch_range_window)), ":", color="black")
            plt.text(45, quant, str(percent_list[i] * 100) + "%")

    plt.xlabel("Neighbourhood size r")
    plt.ylabel(f"Global autocorrelation z-score for {sim_tag}")
    plt.legend(fontsize=5)

    plt.savefig(f"{base_path}/results/{csv_file_name[:-4]}.png")
    plt.close()
