import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import datetime as dt

import detectionefficiency
import inst_param as inst
import fitfunc

# Constants
# cpc = "SN210"
cpc = "Cortland"
# ini_temps = [98, 96, 91, 86, 81, 76, 71, 61, 40, 35]
# ini_temps = [98]
ini_temps = [98,91,81,71,61]
# thab_mon = 228
# thab_tri = 425
# skip_start = 10
# skip_end = 10
skip = (10, 10)  # (start_skip, end_skip)
thab = (216, 325)  # (thabMon, thabDi)
negative_ions = False

fit_skip = 0
# Loop through settings, calculate, and join detection efficiency data tables
fits = np.arange(0)
for temp in ini_temps:
    # Data title = Growth tube + Initator Temp
    data_title = cpc + "_" + str(temp)
    print(data_title)

    # Calculate detection efficency
    detect_eff, data_directory = detectionefficiency.calc_cpc_cal(
        data_title, thab, skip, negative_ions
    )

    detect_eff[detect_eff == np.inf] = 0
    detect_eff = detect_eff.fillna(0)
    detect_eff.loc[
        detect_eff["elec_concentration"] < 50, "Detection Efficiency"
    ] = 0

    # ADD SCALING HERE:
    scaling_factor = 316 / 311
    detect_eff["Detection Efficiency"] = detect_eff["Detection Efficiency"] * scaling_factor
    print(f"Applied scaling factor: {scaling_factor:.4f}")

    print(detect_eff.head())

    x = detect_eff.loc[fit_skip:, "Diameter"].values
    y = detect_eff.loc[fit_skip:, "Detection Efficiency"].values
    try:
        popt, _ = curve_fit(
            fitfunc.cpc_eta_activ_w_GK,
            x,
            y,
            bounds=inst.fit_settings["bounds"],
            maxfev=5000,
        )
    except:
        popt = np.zeros(len(inst.fit_settings["bounds"][0]))
    print(popt)
    fits = np.append(fits, popt)

    # Merge dataframes
    detect_eff = detect_eff.add_suffix("_" + data_title)
    try:
        combined_detect_eff = combined_detect_eff.join(detect_eff, how="left")
    except:
        combined_detect_eff = detect_eff

fits = fits.reshape(len(ini_temps), len(popt))

# Save combined dataframe with the date
file_date = data_directory[1][0:8]
output_filename = file_date + "_detect_eff_" + cpc + ".csv"
output_path = os.path.join(data_directory[0], output_filename)
combined_detect_eff.to_csv(output_path)

# Save fits
fits_output_filename = file_date + "_fits_" + cpc + ".csv"
fits_output_path = os.path.join(data_directory[0], fits_output_filename)
np.savetxt(fits_output_path, fits, delimiter=",")

# Save report
report_output_filename = data_directory[1][0:15] + "_report_" + cpc + ".txt"
report_output_path = os.path.join(data_directory[0], report_output_filename)


def generate_analysis_report(output_path, negative_ions, thab, skip):
    f = open(output_path, "w")

    f.writelines(
        [
            "Analysis Date: "
            + dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            + "\n",
            "THAB Monomer Voltage: " + str(thab[0]) + "\n",
            "THAB Trimer Voltage: " + str(thab[1]) + "\n",
            "Negative Ions? " + str(negative_ions),
            "Start Skip: " + str(skip[0]) + "\n",
            "End Skip: " + str(skip[1]) + "\n",
        ]
    )
    f.close()


generate_analysis_report(report_output_path, negative_ions, thab, skip)

# Plot constants
graph_mode = "Diameter"
graph_title = file_date + "_" + cpc + "_Combined"
x = np.linspace(1, 10, 100)
plot_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Define different marker shapes for each temperature
marker_shapes = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', 'h', '+']

# FIXED: Create figure ONCE, then plot all curves on the same axes
fig = None
ax = None
plot_legend = []

for i, temp in enumerate(ini_temps):
    data_title = cpc + "_" + str(temp)

    # Get color and marker for this temperature
    color = plot_colors[i % len(plot_colors)]
    marker = marker_shapes[i % len(marker_shapes)]

    # Only create figure for the first temperature
    if i == 0:
        fig, ax = detectionefficiency.plot_detect_eff(
            graph_mode,
            graph_title,
            data_directory[0],
            combined_detect_eff,
            1,
            "_" + data_title,
        )
        # Clear the default plot from the function to start fresh
        ax.clear()
        ax.set_xlabel('Diameter (nm)')
        ax.set_ylabel('Detection Efficiency')
        ax.set_title(graph_title)

    # Extract data for this temperature
    diameter_col = f"Diameter_{data_title}"
    efficiency_col = f"Detection Efficiency_{data_title}"

    if diameter_col in combined_detect_eff.columns and efficiency_col in combined_detect_eff.columns:
        # Plot the experimental data points with unique shape and color
        valid_data = combined_detect_eff[[diameter_col, efficiency_col]].dropna()
        ax.scatter(valid_data[diameter_col], valid_data[efficiency_col],
                   color=color, marker=marker, s=50, alpha=0.7, label=f'1/{temp}°C')

    # Plot the fitted curve with matching color (no label for legend)
    ax.plot(x, fitfunc.cpc_eta_activ_w_GK(x, *fits[i, :]),
            color=color, linestyle='-', linewidth=1)

# Set axis limits
ax.set_ylim([0, 1])
ax.set_xlim([0, 10])

# Configure x-axis ticks - show every 1
ax.set_xticks(np.arange(1, 10, 1))  # Major ticks every 1 unit from 1 to 10

# Configure y-axis ticks - show every 0.2 for readability
ax.set_yticks(np.arange(0, 1, 0.1))  # Major ticks every 0.2 from 0 to 1.2

# Add grid
ax.grid(True, which='major', alpha=0.7, linestyle='-', linewidth=0.8)  # Major grid lines
ax.grid(True, which='minor', alpha=0.3, linestyle='--', linewidth=0.5)  # Minor grid lines

# Set minor ticks for finer grid
ax.set_xticks(np.arange(1, 10, 0.1), minor=True)  # Minor ticks every 0.1 on x-axis
ax.set_yticks(np.arange(0, 1, 0.1), minor=True)  # Minor ticks every 0.1 on y-axis

# Legend - only for the scatter points (shapes+colors), not the fitted lines
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
fig.subplots_adjust(right=0.75)

# Save plot
fig.savefig(
    os.path.join(
        data_directory[0],
        "Graphs",
        file_date + "_" + cpc + "_Combined_detect_eff_dia",
    ),
    dpi=300,
)

plt.show()