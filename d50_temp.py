"""
Standalone D50 temperature analysis tool.

This script reads detection-efficiency and fit-parameter files produced by
run_detecteff.py, then estimates D50 as the diameter where the fitted
detection-efficiency curve reaches 50% of its fitted maximum.

Example:
    python d50_temp.py --data-dir results --file-date 20250826 --cpc Earligold \
        --conditioner-temp 10 --temperatures 98 91 81 71 61 51 41
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import fitfunc

CPC_NAME = ""
CONDITIONER_TEMP = 10.0
TEMPERATURES = []
DATA_DIRECTORY = "."
FILE_DATE = ""
DETECT_EFF_FILE = ""
FITS_FILE = ""
TARGET_CUT_POINTS = [1.6, 1.9, 2.3, 2.7, 3.3]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate D50 vs. temperature from CPC detection-efficiency fits."
    )
    parser.add_argument("--data-dir", required=True, help="Directory containing detect_eff and fits CSV files.")
    parser.add_argument("--file-date", required=True, help="Date prefix used in output files, e.g. 20250826.")
    parser.add_argument("--cpc", required=True, help="CPC name used in output files, e.g. Earligold.")
    parser.add_argument(
        "--conditioner-temp",
        type=float,
        required=True,
        help="Conditioner temperature in degrees C, used to calculate Delta T.",
    )
    parser.add_argument(
        "--temperatures",
        nargs="+",
        type=float,
        required=True,
        help="Initiator temperatures in degrees C, in the same order as the fit rows.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        type=float,
        default=TARGET_CUT_POINTS,
        help="Target D50 cut points in nm.",
    )
    return parser.parse_args()


def configure(args):
    global CPC_NAME, CONDITIONER_TEMP, TEMPERATURES
    global DATA_DIRECTORY, FILE_DATE, DETECT_EFF_FILE, FITS_FILE, TARGET_CUT_POINTS

    CPC_NAME = args.cpc
    CONDITIONER_TEMP = args.conditioner_temp
    TEMPERATURES = args.temperatures
    DATA_DIRECTORY = args.data_dir
    FILE_DATE = args.file_date
    TARGET_CUT_POINTS = args.targets
    DETECT_EFF_FILE = os.path.join(DATA_DIRECTORY, f"{FILE_DATE}_detect_eff_{CPC_NAME}.csv")
    FITS_FILE = os.path.join(DATA_DIRECTORY, f"{FILE_DATE}_fits_{CPC_NAME}.csv")

def load_data():
    """Load detection efficiency and fit data"""
    print(f"Loading data from:")
    print(f"  Detection efficiency: {DETECT_EFF_FILE}")
    print(f"  Fits: {FITS_FILE}")

    # Load detection efficiency data
    if not os.path.exists(DETECT_EFF_FILE):
        print(f"ERROR: Detection efficiency file not found: {DETECT_EFF_FILE}")
        return None, None

    detect_eff_df = pd.read_csv(DETECT_EFF_FILE, index_col=0)

    # Load fits data
    if not os.path.exists(FITS_FILE):
        print(f"ERROR: Fits file not found: {FITS_FILE}")
        return detect_eff_df, None

    fits_array = np.loadtxt(FITS_FILE, delimiter=',')

    # Handle case where fits is 1D (single temperature)
    if fits_array.ndim == 1:
        fits_array = fits_array.reshape(1, -1)

    print(f"Loaded data successfully:")
    print(f"  Detection efficiency shape: {detect_eff_df.shape}")
    print(f"  Fits shape: {fits_array.shape}")
    print(f"  Number of temperatures: {len(TEMPERATURES)}")

    return detect_eff_df, fits_array

def find_d50_from_fitted_curve(fit_params, temp_c):
    """
    Find d50 using the fitted curve for both maximum efficiency and D50 calculation
    This is the most accurate method as it uses the smooth fitted curve
    D50 = diameter at 50% of the fitted curve maximum efficiency
    """
    try:
        # Create diameter array with fine resolution
        diameters = np.linspace(1, 15, 3000)  # Extended range for finding true maximum

        # Calculate efficiencies from fitted curve
        efficiencies = []
        for d in diameters:
            try:
                eff = fitfunc.cpc_eta_activ_w_GK(float(d), *fit_params)
                # Ensure we get a scalar value
                if hasattr(eff, '__len__'):
                    eff = eff.item() if eff.size == 1 else eff[0]
                if np.isfinite(eff):  # Check for valid efficiency
                    efficiencies.append(float(eff))
                else:
                    efficiencies.append(np.nan)
            except Exception as e:
                efficiencies.append(np.nan)

        efficiencies = np.array(efficiencies)

        # Debug: Check array shapes
        # print(f"  Debug: diameters shape: {diameters.shape}, efficiencies shape: {efficiencies.shape}")

        # Remove NaN values - both arrays should be 1D now
        valid_mask = ~np.isnan(efficiencies) & np.isfinite(efficiencies)
        # print(f"  Debug: valid_mask shape: {valid_mask.shape}, valid count: {np.sum(valid_mask)}")

        if not np.any(valid_mask):
            print(f"  No valid efficiencies from fitted curve for {temp_c}°C")
            return np.nan

        # Apply mask to get valid points - should work now
        valid_effs = efficiencies[valid_mask]
        valid_diams = diameters[valid_mask]

        # print(f"  Debug: valid_effs shape: {valid_effs.shape}, valid_diams shape: {valid_diams.shape}")

        # Find maximum efficiency from the fitted curve
        max_idx = np.argmax(valid_effs)
        max_efficiency = valid_effs[max_idx]
        max_diameter = valid_diams[max_idx]
        target_efficiency = 0.5 * max_efficiency  # 50% of fitted curve maximum

        print(f"  Fitted curve max: {max_efficiency:.3f} at {max_diameter:.2f} nm")
        print(f"  Target (50% of fitted max): {target_efficiency:.3f}")

        # Check if we can reach the target (look in the rising part of the curve)
        # Only consider diameters smaller than the maximum
        rising_mask = valid_diams <= max_diameter
        rising_effs = valid_effs[rising_mask]
        rising_diams = valid_diams[rising_mask]

        if len(rising_effs) == 0:
            print(f"  No rising portion found for {temp_c}°C")
            return np.nan

        if rising_effs.min() > target_efficiency:
            print(f"  Target efficiency {target_efficiency:.3f} not reachable in rising portion for {temp_c}°C")
            return np.nan

        # Find the diameter in the rising portion where efficiency equals target
        diff_from_target = np.abs(rising_effs - target_efficiency)
        closest_idx = np.argmin(diff_from_target)
        d50 = rising_diams[closest_idx]
        closest_eff = rising_effs[closest_idx]

        # Refine with linear interpolation if we have nearby points
        if closest_idx > 0 and closest_idx < len(rising_effs) - 1:
            # Get adjacent points for interpolation
            idx_before = closest_idx - 1
            idx_after = closest_idx + 1

            eff_before = rising_effs[idx_before]
            eff_current = rising_effs[closest_idx]
            eff_after = rising_effs[idx_after]

            diam_before = rising_diams[idx_before]
            diam_current = rising_diams[closest_idx]
            diam_after = rising_diams[idx_after]

            # Choose the best interpolation interval based on which side is closer to target
            if abs(eff_before - target_efficiency) < abs(eff_after - target_efficiency):
                # Interpolate between before and current
                if eff_current != eff_before:
                    t = (target_efficiency - eff_before) / (eff_current - eff_before)
                    d50 = diam_before + t * (diam_current - diam_before)
            else:
                # Interpolate between current and after
                if eff_after != eff_current:
                    t = (target_efficiency - eff_current) / (eff_after - eff_current)
                    d50 = diam_current + t * (diam_after - diam_current)

        print(f"  Found D50 = {d50:.3f} nm (eff = {target_efficiency:.3f}, 50% of fitted max) for {temp_c}°C")
        return d50

    except Exception as e:
        print(f"  Error in fitted curve d50 search for {temp_c}°C: {e}")
        import traceback
        print(f"  Traceback: {traceback.format_exc()}")
        return np.nan

def find_d50_from_data(detect_eff_df, temp):
    """
    Fallback method: Find d50 directly from the measured data
    D50 = diameter at 50% of MAXIMUM detection efficiency in the data
    """
    try:
        data_title = f"{CPC_NAME}_{temp}"
        diameter_col = f"Diameter_{data_title}"
        efficiency_col = f"Detection Efficiency_{data_title}"

        if diameter_col not in detect_eff_df.columns or efficiency_col not in detect_eff_df.columns:
            print(f"  Columns not found for {temp}°C")
            return np.nan

        # Get valid data
        valid_data = detect_eff_df[[diameter_col, efficiency_col]].dropna()
        if len(valid_data) == 0:
            print(f"  No valid data for {temp}°C")
            return np.nan

        diameters = valid_data[diameter_col].values
        efficiencies = valid_data[efficiency_col].values

        # Find maximum efficiency in the measured data
        max_efficiency = efficiencies.max()
        target_efficiency = 0.5 * max_efficiency  # 50% of maximum

        print(f"  Data max efficiency: {max_efficiency:.3f}, Target (50%): {target_efficiency:.3f}")

        # Check if we can reach the target
        if efficiencies.min() > target_efficiency:
            print(f"  Min measured efficiency {efficiencies.min():.3f} > target {target_efficiency:.3f} for {temp}°C")
            return np.nan

        # Sort by diameter for interpolation
        sorted_idx = np.argsort(diameters)
        sorted_diams = diameters[sorted_idx]
        sorted_effs = efficiencies[sorted_idx]

        # Find the crossing point
        # Look for where efficiency crosses the target value
        crossing_indices = []
        for i in range(len(sorted_effs) - 1):
            if ((sorted_effs[i] <= target_efficiency <= sorted_effs[i+1]) or
                (sorted_effs[i] >= target_efficiency >= sorted_effs[i+1])):
                crossing_indices.append(i)

        if not crossing_indices:
            # No crossing found, find closest point
            closest_idx = np.argmin(np.abs(sorted_effs - target_efficiency))
            d50 = sorted_diams[closest_idx]
            closest_eff = sorted_effs[closest_idx]
            print(f"  No crossing found, closest point: D50 = {d50:.3f} nm (eff = {closest_eff:.3f})")
            return d50

        # Use the first crossing (smallest diameter where target is reached)
        crossing_idx = crossing_indices[0]

        # Linear interpolation between the two points
        x1, y1 = sorted_diams[crossing_idx], sorted_effs[crossing_idx]
        x2, y2 = sorted_diams[crossing_idx + 1], sorted_effs[crossing_idx + 1]

        # Interpolate to find exact diameter
        if y2 != y1:  # Avoid division by zero
            d50 = x1 + (target_efficiency - y1) * (x2 - x1) / (y2 - y1)
        else:
            d50 = (x1 + x2) / 2

        print(f"  Found D50 = {d50:.3f} nm (target eff = {target_efficiency:.3f}, {target_efficiency/max_efficiency*100:.1f}% of max) for {temp}°C")
        return d50

    except Exception as e:
        print(f"  Error finding d50 from data for {temp}°C: {e}")
        return np.nan

def analyze_d50_temperature_relationship():
    """Main analysis function"""
    print("="*60)
    print("D50 Temperature Analysis")
    print("="*60)

    # Load data
    detect_eff_df, fits_array = load_data()
    if detect_eff_df is None:
        return

    print(f"\nAnalyzing D50 for temperatures: {TEMPERATURES}")
    print("-" * 50)

    d50_values = []
    valid_temps = []
    valid_d50s = []

    for i, temp in enumerate(TEMPERATURES):
        print(f"\nTemperature {temp}°C:")

        d50 = np.nan

        # Method 1: Try fitted curve method first (most accurate)
        if fits_array is not None and i < len(fits_array):
            print(f"  Trying fitted curve method...")
            d50 = find_d50_from_fitted_curve(fits_array[i, :], temp)

        # Method 2: If fitted method fails, fall back to direct data interpolation
        if np.isnan(d50):
            print(f"  Fitted curve failed, trying direct data interpolation...")
            d50 = find_d50_from_data(detect_eff_df, temp)

        d50_values.append(d50)

        # Only accept reasonable D50 values
        if not np.isnan(d50) and 0.5 <= d50 <= 20:
            valid_temps.append(temp)
            valid_d50s.append(d50)
        else:
            print(f"  Rejecting unreasonable D50 value: {d50}")

    # Results summary
    print("\n" + "="*60)
    print("D50 ANALYSIS RESULTS")
    print("="*60)
    print("Temperature (°C) | D50 (nm)")
    print("-" * 30)

    for temp, d50 in zip(TEMPERATURES, d50_values):
        if not np.isnan(d50):
            print(f"{temp:8.0f}        | {d50:6.3f}")
        else:
            print(f"{temp:8.0f}        | No D50 found")

    if len(valid_temps) < 2:
        print(f"\nNot enough valid D50 values ({len(valid_temps)}) for interpolation")
        return

    # Calculate Delta T values
    delta_t_values = [temp - CONDITIONER_TEMP for temp in valid_temps]

    # DEBUG: Print the values to see what's happening
    print(f"\nDEBUG - Temperature Calculations:")
    print(f"Conditioner temperature set to: {CONDITIONER_TEMP}°C")
    print("Initiator T (°C) | Delta T (°C) | D50 (nm)")
    print("-" * 42)

    for temp, dt, d50 in zip(valid_temps, delta_t_values, valid_d50s):
        print(f"{temp:8.0f}        | {dt:8.1f}    | {d50:6.3f}")

    # Check for unreasonable Delta T values
    if any(dt < 0 or dt > 200 for dt in delta_t_values):
        print(f"\n⚠ WARNING: Unreasonable Delta T values detected!")
        print(f"This suggests the conditioner temperature ({CONDITIONER_TEMP}°C) might be wrong.")
        print(f"Typical conditioner temperatures are 5-20°C.")
        print(f"Please check and update CONDITIONER_TEMP in the configuration section.")

        # Ask user what conditioner temp to use
        print(f"\nCurrent settings:")
        print(f"- Initiator temperatures: {valid_temps}")
        print(f"- Conditioner temperature: {CONDITIONER_TEMP}°C")
        print(f"- Resulting Delta T: {delta_t_values}")

        # Suggest a reasonable conditioner temperature
        min_reasonable_delta_t = 30  # Typical minimum
        max_reasonable_delta_t = 120  # Typical maximum

        # Work backwards from reasonable Delta T
        suggested_conditioner = min(valid_temps) - max_reasonable_delta_t
        print(f"\nSuggested conditioner temperature: {suggested_conditioner:.0f}°C")
        print(f"This would give Delta T range: {min(valid_temps) - suggested_conditioner:.0f} to {max(valid_temps) - suggested_conditioner:.0f}°C")

        return  # Exit early to avoid bad plots

    # Create interpolation functions
    # Sort by Delta T for proper interpolation
    sorted_indices = np.argsort(delta_t_values)
    sorted_delta_t = np.array(delta_t_values)[sorted_indices]
    sorted_d50s = np.array(valid_d50s)[sorted_indices]
    sorted_temps = np.array(valid_temps)[sorted_indices]

    # Interpolation functions
    d50_vs_delta_t = interp1d(sorted_delta_t, sorted_d50s,
                             kind='linear',
                             bounds_error=False,
                             fill_value='extrapolate')

    delta_t_vs_d50 = interp1d(sorted_d50s, sorted_delta_t,
                             kind='linear',
                             bounds_error=False,
                             fill_value='extrapolate')

    temp_vs_d50 = interp1d(sorted_d50s, sorted_temps,
                          kind='linear',
                          bounds_error=False,
                          fill_value='extrapolate')

    target_cut_points = TARGET_CUT_POINTS

    print(f"\nYOUR TARGET CUT POINT SETTINGS")
    print("="*70)
    print("Target D50 (nm) | Required ΔT (°C) | Required Init T (°C) | Status")
    print("-" * 70)

    achievable_targets = []
    extrapolation_targets = []

    for target_d50 in target_cut_points:
        if min(sorted_d50s) <= target_d50 <= max(sorted_d50s):
            # Within measured range - reliable interpolation
            required_delta_t = float(delta_t_vs_d50(target_d50))
            required_init_temp = float(temp_vs_d50(target_d50))
            status = "✓ Achievable"
            achievable_targets.append((target_d50, required_delta_t, required_init_temp))
            print(f"{target_d50:8.1f}        | {required_delta_t:8.1f}        | {required_init_temp:8.1f}        | {status}")
        else:
            # Outside measured range - requires extrapolation
            required_delta_t = float(delta_t_vs_d50(target_d50))
            required_init_temp = float(temp_vs_d50(target_d50))

            if target_d50 < min(sorted_d50s):
                status = "⚠ Need higher ΔT"
                extrapolation_targets.append((target_d50, required_delta_t, required_init_temp, "higher"))
            else:
                status = "⚠ Need lower ΔT"
                extrapolation_targets.append((target_d50, required_delta_t, required_init_temp, "lower"))

            print(f"{target_d50:8.1f}        | {required_delta_t:8.1f}        | {required_init_temp:8.1f}        | {status}")

    # Recommendations for achieving all targets
    print(f"\nRECOMMENDATIONS")
    print("="*50)

    if achievable_targets:
        print("✓ IMMEDIATELY ACHIEVABLE:")
        for d50, dt, temp in achievable_targets:
            print(f"  D50 {d50} nm: Set initiator to {temp:.0f}°C (ΔT = {dt:.0f}°C)")

    if extrapolation_targets:
        print("\n⚠ REQUIRE ADDITIONAL CALIBRATION:")
        for d50, dt, temp, direction in extrapolation_targets:
            if direction == "higher":
                print(f"  D50 {d50} nm: Need initiator >{max(valid_temps):.0f}°C (estimated {temp:.0f}°C)")
                print(f"    → Recommended: Test at {max(valid_temps)+10:.0f}°C, {max(valid_temps)+20:.0f}°C")
            else:
                print(f"  D50 {d50} nm: Need initiator <{min(valid_temps):.0f}°C (estimated {temp:.0f}°C)")
                print(f"    → Recommended: Test at {min(valid_temps)-10:.0f}°C, {min(valid_temps)-20:.0f}°C")

    # Current coverage
    print(f"\nCURRENT CALIBRATION COVERAGE:")
    print(f"  D50 range: {min(sorted_d50s):.1f} - {max(sorted_d50s):.1f} nm")
    print(f"  ΔT range: {min(sorted_delta_t):.0f} - {max(sorted_delta_t):.0f}°C")
    print(f"  Init T range: {min(sorted_temps):.0f} - {max(sorted_temps):.0f}°C")

    # Additional recommendations
    if any(target < min(sorted_d50s) for target in target_cut_points):
        min_missing = min([t for t in target_cut_points if t < min(sorted_d50s)])
        suggested_temp = max(valid_temps) + 15
        print(f"\nSUGGESTED NEXT CALIBRATION:")
        print(f"  To achieve D50 = {min_missing} nm, try initiator temperature ≈ {suggested_temp:.0f}°C")
        print(f"  Recommended test temperatures: {suggested_temp-5:.0f}°C, {suggested_temp:.0f}°C, {suggested_temp+5:.0f}°C")

    # Create plots
    create_analysis_plots(valid_temps, valid_d50s, delta_t_values,
                         d50_vs_delta_t, temp_vs_d50)

    # Save results
    save_analysis_results(valid_temps, valid_d50s, delta_t_values)

    # Create lookup functions
    create_lookup_functions(d50_vs_delta_t, delta_t_vs_d50, temp_vs_d50)

def create_analysis_plots(valid_temps, valid_d50s, delta_t_values,
                         d50_vs_delta_t, temp_vs_d50):
    """Create analysis plots with target cut points highlighted"""
    target_cut_points = TARGET_CUT_POINTS

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: D50 vs Initiator Temperature
    ax1.scatter(valid_temps, valid_d50s, color='red', s=80, zorder=5,
               label='Measured D50', edgecolor='black', linewidth=1)

    temp_range = np.linspace(min(valid_temps), max(valid_temps), 100)
    d50_range = [temp_vs_d50(t) for t in temp_range]
    ax1.plot(temp_range, d50_range, 'b-', linewidth=2, label='Interpolated')

    # Add target cut points
    for target in target_cut_points:
        if min(valid_d50s) <= target <= max(valid_d50s):
            target_temp = temp_vs_d50(target)
            ax1.axhline(y=target, color='green', linestyle='--', alpha=0.7)
            ax1.axvline(x=target_temp, color='green', linestyle='--', alpha=0.7)
            ax1.plot(target_temp, target, 'go', markersize=8, markeredgecolor='black')
            ax1.text(target_temp+1, target+0.05, f'{target}nm', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

    ax1.set_xlabel('Initiator Temperature (°C)', fontsize=12)
    ax1.set_ylabel('D50 Cut Point (nm)', fontsize=12)
    ax1.set_title(f'D50 vs Initiator Temperature - {CPC_NAME}', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Set reasonable y-axis limits for D50
    y_min = min(min(valid_d50s), min(target_cut_points)) * 0.9
    y_max = max(max(valid_d50s), max(target_cut_points)) * 1.1
    ax1.set_ylim([y_min, y_max])

    # Plot 2: D50 vs Delta T
    ax2.scatter(delta_t_values, valid_d50s, color='blue', s=80, zorder=5,
               label='Measured D50', edgecolor='black', linewidth=1)

    delta_t_range = np.linspace(min(delta_t_values), max(delta_t_values), 100)
    d50_interpolated = [d50_vs_delta_t(dt) for dt in delta_t_range]
    ax2.plot(delta_t_range, d50_interpolated, 'r-', linewidth=2, label='Interpolated')

    # Add target cut points
    delta_t_vs_d50_func = interp1d(valid_d50s, delta_t_values, kind='linear', fill_value='extrapolate')
    for target in target_cut_points:
        target_dt = delta_t_vs_d50_func(target)
        if min(valid_d50s) <= target <= max(valid_d50s):
            # Within range - solid lines
            ax2.axhline(y=target, color='green', linestyle='-', alpha=0.8, linewidth=1.5)
            ax2.axvline(x=target_dt, color='green', linestyle='-', alpha=0.8, linewidth=1.5)
            ax2.plot(target_dt, target, 'go', markersize=8, markeredgecolor='black')
        else:
            # Outside range - dashed lines
            ax2.axhline(y=target, color='orange', linestyle='--', alpha=0.7)
            ax2.plot(target_dt, target, 'o', color='orange', markersize=8, markeredgecolor='black')

        ax2.text(target_dt+1, target+0.05, f'{target}nm', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3",
                         facecolor="lightgreen" if min(valid_d50s) <= target <= max(valid_d50s) else "orange",
                         alpha=0.8))

    ax2.set_xlabel('Delta T (°C)', fontsize=12)
    ax2.set_ylabel('D50 Cut Point (nm)', fontsize=12)
    ax2.set_title(f'D50 vs Delta T - {CPC_NAME}', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Set reasonable y-axis limits
    ax2.set_ylim([y_min, y_max])

    # Add text box with target info
    achievable = [t for t in target_cut_points if min(valid_d50s) <= t <= max(valid_d50s)]
    need_higher = [t for t in target_cut_points if t < min(valid_d50s)]

    info_text = f"Targets\n✓ Achievable: {achievable}\n⚠ Need higher ΔT: {need_higher}"
    ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))

    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(DATA_DIRECTORY, "Graphs")
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{FILE_DATE}_{CPC_NAME}_D50_Analysis_Targets.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")

    plt.show()

def save_analysis_results(valid_temps, valid_d50s, delta_t_values):
    """Save analysis results to CSV"""
    analysis_df = pd.DataFrame({
        'Initiator_Temperature_C': valid_temps,
        'Delta_T_C': delta_t_values,
        'D50_nm': valid_d50s
    })

    output_path = os.path.join(DATA_DIRECTORY, f"{FILE_DATE}_D50_analysis_{CPC_NAME}.csv")
    analysis_df.to_csv(output_path, index=False)
    print(f"Analysis data saved to: {output_path}")

def create_lookup_functions(d50_vs_delta_t, delta_t_vs_d50, temp_vs_d50):
    """Create and demonstrate lookup functions"""
    print(f"\nLOOKUP FUNCTIONS CREATED")
    print("="*40)

    def get_delta_t_for_d50(target_d50):
        """Get required Delta T for a target D50"""
        return float(delta_t_vs_d50(target_d50))

    def get_d50_for_delta_t(delta_t):
        """Get D50 for a given Delta T"""
        return float(d50_vs_delta_t(delta_t))

    def get_initiator_temp_for_d50(target_d50):
        """Get required initiator temperature for target D50"""
        return float(temp_vs_d50(target_d50))

    # Example usage
    print("Example usage:")
    try:
        print(f"  For D50 = 3.0 nm: ΔT = {get_delta_t_for_d50(3.0):.1f}°C, Init T = {get_initiator_temp_for_d50(3.0):.1f}°C")
        print(f"  For ΔT = 80°C: D50 = {get_d50_for_delta_t(80):.2f} nm")
    except:
        print("  Example calculations failed - check your data range")

    return get_delta_t_for_d50, get_d50_for_delta_t, get_initiator_temp_for_d50

if __name__ == "__main__":
    configure(parse_args())
    analyze_d50_temperature_relationship()
