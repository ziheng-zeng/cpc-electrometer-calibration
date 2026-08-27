import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np


def calc_mobility_conv(thab):
    thabMon = thab[0]
    thabTri = thab[1]
    thabMonMobilDia = 1.47
    thabTriMobilDia = 1.78
    mobilityConvSlope = 1 / (
        (thabTri - thabMon) / (thabTriMobilDia - thabMonMobilDia)
    )
    mobilityConvOffset = thabMonMobilDia - thabMon * mobilityConvSlope
    return mobilityConvSlope, mobilityConvOffset


def calc_detect_eff(
    joined_data,
    mobilityConvSlope,
    mobilityConvOffset,
    skip,
    negative_ions=False,
):
    # Create new df for detection efficiency related measurements
    detect_eff = joined_data.loc[
        :,
        [
            "elec_dma_voltage",
            "cpc_concentration",
            "elec_concentration",
            "elec_dma_set_voltage",
        ],
    ]
    detect_eff["elec_dma_set_voltage"] = abs(detect_eff["elec_dma_set_voltage"])

    # Calculate diameter
    detect_eff["Diameter"] = (
        abs(detect_eff["elec_dma_voltage"]) * mobilityConvSlope
        + mobilityConvOffset
    )
    start_skip = skip[0]
    end_skip = skip[1]

    # Preprocess end_skip to prevent errors
    if end_skip == 0:
        end_skip = None
    elif end_skip > 0:
        end_skip = -end_skip

    # Group then skip rows
    detect_eff_avg = detect_eff.groupby(
        "elec_dma_set_voltage", as_index=False
    ).apply(lambda x: x.iloc[start_skip:end_skip])

    # Average detection effiency measurements
    detect_eff_avg = detect_eff_avg.groupby(
        "elec_dma_set_voltage", as_index=False
    ).mean()

    # Correct Electrometer Measurements
    detect_eff_avg["elec_concentration"] = detect_eff_avg[
        "elec_concentration"
    ] * (-1 + 2 * negative_ions)
    detect_eff_avg["elec_concentration"] = (
        detect_eff_avg["elec_concentration"]
        - detect_eff_avg.reset_index().at[0, "elec_concentration"]
    )

    # Calculate Detection Efficiency
    detect_eff_avg["Detection Efficiency"] = (
        detect_eff_avg["cpc_concentration"]
        / detect_eff_avg["elec_concentration"]
    )

    return detect_eff_avg


def plot_detect_eff(
    x_param, data_title, data_dir, detect_eff_avg, fig_num=None, suffix=""
):
    if x_param == "Voltage":
        graph_param = [
            "elec_dma_voltage" + suffix,
            "DMA Voltage (V)",
            "_detect_eff_vlt.png",
        ]
    elif x_param == "Diameter":
        graph_param = [
            "Diameter" + suffix,
            "Mobility Diameter",
            "_detect_eff_dia.png",
        ]
    # plt.figure(fig_num)
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(
        detect_eff_avg[graph_param[0]],
        detect_eff_avg["Detection Efficiency" + suffix],
        "x",
    )
    ax.set_title(data_title + " CPC Detection Efficiency")
    ax.set_xlabel(graph_param[1])
    ax.set_ylabel("Detection Efficiency")
    ax.set_ylim([-0.2, 1.2])
    # plt.savefig(
    #     os.path.join(data_dir, "Graphs", data_title + graph_param[2]),
    #     dpi=300,
    # )
    return fig, ax


def plot_conc(x_param, data_directory, data_title, detect_eff_avg):
    if x_param == "Voltage":
        graph_param = ["elec_dma_voltage", "DMA Voltage (V)", "_conc_vlt.png"]
    elif x_param == "Diameter":
        graph_param = ["Diameter", "Mobility Diameter", "_conc_dia.png"]
    fig, (ax1, ax2) = plt.subplots(2, constrained_layout=True, sharex=True)

    # Electrometer Concentration Plot
    ax1.plot(
        detect_eff_avg[graph_param[0]],
        detect_eff_avg["elec_concentration"],
    )
    ax1.set_ylabel("Elec. Concentration (#/cc)")

    # CPC Concentration Plot
    ax2.plot(
        detect_eff_avg[graph_param[0]], detect_eff_avg["cpc_concentration"]
    )
    ax2.set_ylabel("CPC Concentration (#/cc)")

    # Plot Labels
    for ax in (ax1, ax2):
        ax.label_outer()
    fig.suptitle("Size Distribution & CPC Concentration")
    fig.supxlabel(graph_param["x_label"])

    return fig
    # fig.savefig(
    #     os.path.join(data_directory, "Graphs", data_title + graph_param[2]),
    #     dpi=300,
    # )


# Constants
def calc_cpc_cal(data_title, thab, skip=(0, 0), negative_ions=False):
    # Read in data
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    PathNameJoinedData = filedialog.askopenfilenames(
        title="Import Joined Data File",
        filetypes=(("CSV Files", "joined*.csv"),),
    )
    joined_data = pd.read_csv(
        PathNameJoinedData[0],
        # engine="pyarrow",
        header=0,
        index_col=0,
    )

    # Save Directory
    data_directory = os.path.split(PathNameJoinedData[0])
    os.makedirs(os.path.join(data_directory[0], "Graphs"), exist_ok=True)

    # Calculate voltage to mobility conversion
    mobilityConvSlope, mobilityConvOffset = calc_mobility_conv(thab)

    # Calculate Detection Effiency
    detect_eff_avg = calc_detect_eff(
        joined_data,
        mobilityConvSlope,
        mobilityConvOffset,
        skip,
        negative_ions,
    )

    output_filename = (
        data_directory[1][0:15] + "_detect_eff_" + data_title + ".csv"
    )
    output_path = os.path.join(data_directory[0], output_filename)
    detect_eff_avg.to_csv(output_path)

    return detect_eff_avg, data_directory


def plot_cpc_cal(data_title, detect_eff_avg, data_directory):
    # Plot detection efficiencies
    plot_detect_eff("Voltage", data_title, data_directory[0], detect_eff_avg)
    plot_detect_eff("Diameter", data_title, data_directory[0], detect_eff_avg)

    # popt, _ = curve_fit(sigmoid, x, pulse_height_avg.iloc[i, -8192:], p0=[1, 1000, 100])

    # Plot concentration of electrometer, CPC by voltage
    plot_conc("Voltage", data_directory[0], data_title, detect_eff_avg)
    plot_conc("Diameter", data_directory[0], data_title, detect_eff_avg)


# Define sigmoid fit function
def sigmoid(x, eta, dp_50, k):
    return eta / (1 + np.exp(-k * (x - dp_50)))


def main():
    detect_eff_avg, data_directory = calc_cpc_cal(
        data_title, thab, skip, negative_ions
    )
    plot_cpc_cal(data_title, detect_eff_avg, data_directory)


if __name__ == "__main__":
    # Ask for data title
    data_title = input("Enter Data Title: ")
    negative_ions = False
    skip = (10, 10)  # (start_skip, end_skip)
    # start_skip = 10
    # end_skip = 10
    thab = (228, 425)  # (thabMon, thabTri)
    # thabMon = 228
    # thabTri = 425
    main()
