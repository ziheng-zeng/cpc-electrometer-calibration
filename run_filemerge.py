#!/usr/bin/env python
"""
Electrometer (DMA) + CPC merge script that:
1. Reads and renames columns using inst_param (same as old merge code)
2. Parses datetime columns with timezone using def_time_col
3. Keeps ALL electrometer data points
4. Adds CPC data where timestamps match (within 1 second, using merge_asof)
5. Produces elec_* and cpc_* headers consistent with the old script
6. Saves output filename in the SAME style as the old script
"""

import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os
from datetime import datetime

import inst_param as inst  # <-- same module as the old script


def def_time_col(data: pd.DataFrame, col_header: str, timezone: str | None):
    """
    Parse datetime column with mixed format and apply timezone.
    Mirrors behavior of the old merge script, but avoids double-localizing.
    """
    # Use mixed format to handle inconsistent datetime formats
    data[col_header] = pd.to_datetime(data[col_header], format="mixed", errors="coerce")

    # Apply timezone if needed and not already tz-aware
    if timezone:
        if data[col_header].dt.tz is None:
            data[col_header] = data[col_header].dt.tz_localize(timezone)

    return data


def merge_files():
    """Main merge function - keeps all electrometer (DMA) data."""

    root = tk.Tk()
    root.withdraw()

    # ================= ELECTROMETER / DMA FILE =================
    dma_input = inst.read_settings["dma"]  # same key as old code

    print("\n1. Select ELECTROMETER/DMA file (this will be the base)...")
    elec_path = filedialog.askopenfilename(
        title="Select Electrometer/DMA File (BASE FILE)",
        filetypes=[dma_input["filetype"]],  # e.g. ('DMA Files', '*.csv')
    )
    if not elec_path:
        print("No electrometer file selected. Exiting.")
        return None

    print(f"Loading electrometer file: {os.path.basename(elec_path)}")
    elec_data = pd.read_csv(elec_path, index_col=False)

    # Rename columns using inst.headers["dma"] just like old code
    elec_data = elec_data.rename(columns=inst.headers["dma"])

    # Parse time column and apply timezone
    elec_datecol = dma_input["datecol"]  # e.g. "Time"
    elec_tzone = dma_input.get("tzone")
    elec_data = def_time_col(elec_data, elec_datecol, elec_tzone)

    # Create unified timestamp column for merging
    elec_data["timestamp"] = elec_data[elec_datecol]
    elec_time_start = elec_data["timestamp"].min()
    elec_time_end = elec_data["timestamp"].max()
    print(
        f"  Electrometer rows: {len(elec_data)} "
        f"(Time range: {elec_time_start} to {elec_time_end})"
    )

    # ================= CPC FILE =================
    cpc = inst.cpc  # instrument key, same as old code
    cpc_input = inst.read_settings[cpc]

    print("\n2. Select CPC file to merge in...")
    cpc_path = filedialog.askopenfilename(
        title="Select CPC File to Merge",
        filetypes=[cpc_input["filetype"]],
        initialdir=os.path.dirname(elec_path),
    )
    if not cpc_path:
        print("No CPC file selected. Exiting.")
        return None

    print(f"Loading CPC file: {os.path.basename(cpc_path)}")

    # Old code: header=0, names=inst.headers[cpc]
    cpc_data = pd.read_csv(
        cpc_path,
        index_col=False,
        header=0,
        names=inst.headers[cpc],
    )

    # Parse CPC datetime using inst settings
    cpc_datecol = cpc_input["datecol"]
    cpc_tzone = cpc_input.get("tzone")
    cpc_data = def_time_col(cpc_data, cpc_datecol, cpc_tzone)

    # Drop rows without valid datetime
    cpc_data = cpc_data[cpc_data[cpc_datecol].notna()].copy()
    cpc_data["timestamp"] = cpc_data[cpc_datecol]

    print(f"  CPC rows with valid timestamps: {len(cpc_data)}")

    # ================= CHECK OVERLAP =================
    overlap_start = max(elec_data["timestamp"].min(), cpc_data["timestamp"].min())
    overlap_end = min(elec_data["timestamp"].max(), cpc_data["timestamp"].max())

    if overlap_start <= overlap_end:
        overlap_seconds = (overlap_end - overlap_start).total_seconds()
        print(f"\n✓ Time overlap: {overlap_seconds:.0f} seconds")
    else:
        print("\n⚠ WARNING: No time overlap between files!")
        print("  CPC data will have NaN values for all rows")

    # ================= PREPARE DATA FOR MERGE =================
    # Prefix everything except 'timestamp' so we can still merge on that column.

    # Electrometer: prefix all columns except 'timestamp'
    elec_prefixed = elec_data.copy()
    elec_ts = elec_prefixed["timestamp"]
    elec_prefixed = elec_prefixed.drop(columns=["timestamp"]).add_prefix("elec_")
    elec_prefixed["timestamp"] = elec_ts

    # CPC: prefix all columns except 'timestamp'
    cpc_prefixed = cpc_data.copy()
    cpc_ts = cpc_prefixed["timestamp"]
    cpc_prefixed = cpc_prefixed.drop(columns=["timestamp"]).add_prefix("cpc_")
    cpc_prefixed["timestamp"] = cpc_ts

    # Sort by timestamp
    elec_prefixed = elec_prefixed.sort_values("timestamp")
    cpc_prefixed = cpc_prefixed.sort_values("timestamp")

    # ================= MERGE (KEEP ALL ELECTROMETER ROWS) =================
    print("\nMerging (keeping all electrometer rows)...")

    merged = pd.merge_asof(
        elec_prefixed,  # LEFT dataframe - keep all these rows
        cpc_prefixed,   # RIGHT dataframe - add these where they match
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=1.0),  # ±1 second
    )

    print(f"✓ Merged data: {len(merged)} rows (same as electrometer)")

    # Count how many rows have CPC data (using one typical CPC column)
    cpc_conc_col = "cpc_concentration"  # should match inst.headers[cpc]
    if cpc_conc_col in merged.columns:
        cpc_matched = merged[cpc_conc_col].notna().sum()
        print(
            f"  Rows with CPC data: {cpc_matched} "
            f"({100 * cpc_matched / len(merged):.1f}%)"
        )
        print(f"  Rows without CPC data: {len(merged) - cpc_matched}")

    # ================= SAVE OUTPUT (OLD NAMING STYLE) =================
    # EXACTLY mimic the old script:
    # output_filename = (
    #     PathNameDMA[0][-27:-17].replace("_", "")
    #     + "_"
    #     + PathNameDMA[0][-16:-8].replace("_", "")
    #     + "_joined_DMA_CPC"
    # )

    PathNameDMA0 = elec_path  # keep the same name the old script used
    output_filename = (
        PathNameDMA0[-27:-17].replace("_", "")
        + "_"
        + PathNameDMA0[-16:-8].replace("_", "")
        + "_joined_DMA_CPC"
    )

    output_folder = os.path.commonpath([elec_path, cpc_path])
    output_path = os.path.join(output_folder, output_filename + ".csv")

    merged.to_csv(output_path, index=False)
    print(f"\n✓ Saved: {output_filename}.csv")
    print(f"  Location: {output_folder}")

    # ================= QUICK DETECTION EFFICIENCY CHECK =================
    print("\n=== Quick Detection Efficiency Check ===")
    elec_conc_col = "elec_Electrometer Concentration"  # same style as old code

    if cpc_conc_col in merged.columns and elec_conc_col in merged.columns:
        valid_mask = (
            merged[cpc_conc_col].notna()
            & merged[elec_conc_col].notna()
            & (merged[elec_conc_col] != 0)
        )
        valid_count = valid_mask.sum()

        if valid_count > 0:
            det_eff = (
                merged.loc[valid_mask, cpc_conc_col]
                / merged.loc[valid_mask, elec_conc_col]
            )
            print(f"Points with valid efficiency: {valid_count}")
            print(f"Detection efficiency mean: {det_eff.mean():.3f}")
            print(f"Detection efficiency range: {det_eff.min():.3f} to {det_eff.max():.3f}")
        else:
            print("No valid points for efficiency calculation")

    print("\n✓ Process complete!")

    # Beep (non-fatal if winsound isn't available)
    try:
        import winsound
        winsound.Beep(440, 500)
    except Exception:
        pass

    return merged


def main():
    """Run merge once, then exit (no 'press Enter' prompt)."""
    print("=" * 60)
    print("ELECTROMETER-BASED MERGE TOOL")
    print("Keeps all electrometer rows, adds CPC where available")
    print("=" * 60)

    try:
        merge_files()
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
