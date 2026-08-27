# repeat_timeseries_gui.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Optional
import re
# ---- Tk UI ----
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

# ------------- USER-TWEAKABLE OPTIONS -------------
SMOOTH_MINUTES: Optional[float] = None  # e.g. 1.0 for 1-min moving average; None = off
RESAMPLE_RULE = "1min"                  # common grid for across-repeat CV
OUTPUT_ROOT: Optional[str] = None       # default = current working dir
TIMEZONE_AWARE = False                  # set True if your timestamps include tz info & you want to preserve it

VERBOSE = True  # set False to silence per-file debug lines

# —— Alignment settings ——
ALIGN_TO_STEP = True           # turn on/off alignment
ALIGN_STREAM = "cpc"           # "cpc" or "elec" -> which stream defines the step
PRE_WINDOW_SEC = 60            # show 60 s before the detected step
XLIM_RIGHT_SEC = None          # e.g., 600 for 10 min; None = auto
# Step detector parameters (robust defaults)
STEP_SMOOTH_SEC = 3.0          # smooth window (seconds) before derivative
DERIV_PCT_THRESHOLD = 95       # percentile of derivative used as cut for “step”
MIN_STEP_DELTA = 0             # optional absolute delta threshold to require (units of concentration); 0 = off

# Column name candidates (extend as needed)
ELEC_TIME_CANDS = ["elec_datetime", "Electrometer Time", "elec_time", "elecTimestamp"]
CPC_TIME_CANDS  = ["cpc_datetime", "cpc_instrument_datetime", "CPC Time", "cpc_time", "cpcTimestamp"]

CPC_CANDS = [
    "cpc_concentration","CPC_Conc","CPC conc","CPC","cpc","cpc_N","cpc_counts","CPC_N",
    "#/cm3 (CPC)","#/cm3_CPC","CPC #/cm3"
]
ELEC_CANDS = [
    "elec_concentration","Electrometer","electrometer","EM","EM_Conc","EM conc","Faraday",
    "#/cm3 (Electrometer)","#/cm3_Elec","Electrometer #/cm3","Elec #/cm3","Elec_Conc"
]


# ------------- HELPERS -------------
def ensure_outdir(root: Optional[str]) -> str:
    d = root or os.getcwd()
    out = os.path.join(d, "Graphs")
    os.makedirs(out, exist_ok=True)
    return out

def pick_col(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for nm in cands:
        if nm.lower() in lower:
            return lower[nm.lower()]
    return None

def to_datetime_col(s: pd.Series) -> pd.Series:
    # robust parse for string or numeric (s/ms since epoch)
    try:
        vals = s.dropna().values
        if len(vals) and np.issubdtype(s.dropna().values[:1].dtype, np.number):
            v = s.astype("float64")
            unit = "ms" if (v.dropna().abs() > 1e12).any() else "s"
            return pd.to_datetime(v, unit=unit, errors="coerce", utc=False)
        return pd.to_datetime(s, errors="coerce", utc=False)
    except Exception:
        return pd.to_datetime(s, errors="coerce", utc=False)

def _to_seconds_since_start(t: pd.Series) -> pd.Series:
    t = pd.to_datetime(t)
    t0 = t.iloc[0]
    return (t - t0).dt.total_seconds()

def detect_step_time(time_s: pd.Series, y: pd.Series,
                     smooth_sec: float = STEP_SMOOTH_SEC,
                     deriv_pct: float = DERIV_PCT_THRESHOLD,
                     min_step_delta: float = MIN_STEP_DELTA) -> Optional[float]:
    """
    Return the time (in seconds since series start) of the first major step-up.
    Heuristic:
      - time-based rolling mean to de-noise
      - central diff derivative
      - threshold at given percentile of positive derivative
      - earliest crossing that also yields a rise of >= min_step_delta within ~10 s
    """
    s = pd.Series(y).astype(float).copy()
    t = pd.Series(time_s).astype(float).copy()
    m = (~np.isnan(s)) & (~np.isnan(t))
    s, t = s[m], t[m]
    if len(s) < 5:
        return None

    # time-based smoothing
    df = pd.DataFrame({"t": t, "y": s}).sort_values("t")
    df = df.set_index(pd.to_timedelta(df["t"], unit="s"))
    win = f"{max(0.5, smooth_sec)}s"
    ys = df["y"].rolling(win, min_periods=1).mean().values
    ts = df["t"].values

    # derivative
    dy = np.diff(ys, prepend=ys[0])
    dt = np.diff(ts, prepend=ts[0])
    dt[dt == 0] = np.nan
    deriv = dy / dt
    deriv = np.nan_to_num(deriv, nan=0.0, posinf=0.0, neginf=0.0)

    # threshold
    pos_deriv = deriv[deriv > 0]
    if len(pos_deriv) == 0:
        return None
    thr = np.percentile(pos_deriv, deriv_pct)

    # earliest index crossing threshold and showing sustained rise soon after
    for i in range(len(deriv)):
        if deriv[i] >= thr:
            t0 = ts[i]
            # require a rise within next ~10 s if min_step_delta > 0
            if min_step_delta > 0:
                j = np.searchsorted(ts, t0 + 10.0)  # index ~10s after
                j = min(j, len(ys) - 1)
                if ys[j] - ys[i] < min_step_delta:
                    continue
            return float(t0)

    return None

def maybe_smooth_two_clocks(df: pd.DataFrame, minutes: Optional[float]) -> pd.DataFrame:
    if minutes is None or df.empty:
        return df

    out = df.copy()

    # Smooth EM on its own clock
    if "time_elec" in out and out["time_elec"].notna().any():
        em = out[["time_elec","elec"]].dropna(subset=["time_elec"]).set_index("time_elec").sort_index()
        em["elec"] = em["elec"].rolling(f"{minutes}min", min_periods=1).mean()
        out.loc[em.index, "elec"] = em["elec"].values

    # Smooth CPC on its own clock
    if "time_cpc" in out and out["time_cpc"].notna().any():
        cc = out[["time_cpc","cpc"]].dropna(subset=["time_cpc"]).set_index("time_cpc").sort_index()
        cc["cpc"] = cc["cpc"].rolling(f"{minutes}min", min_periods=1).mean()
        out.loc[cc.index, "cpc"] = cc["cpc"].values

    return out


def variability_stats(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan, np.nan, np.nan
    mean = s.mean()
    std = s.std(ddof=1) if len(s) > 1 else 0.0
    cv = std/mean if mean != 0 else np.nan
    return mean, std, cv

def _norm(s: str) -> str:
    """normalize a header: strip, collapse spaces/underscores, lower, remove NBSP"""
    s = str(s).replace("\u00A0", " ").strip().lower()
    s = re.sub(r"[\s_]+", " ", s)  # treat underscores like spaces
    return s

def _to_numeric_smart(s: pd.Series) -> pd.Series:
    # replace unicode minus, drop thousands commas, keep digits/.-+eE
    if s.dtype.kind in "biufc":
        return pd.to_numeric(s, errors="coerce")
    ss = (
        s.astype(str)
         .str.replace("\u2212", "-", regex=False)  # unicode minus
         .str.replace(",", "", regex=False)
         .str.replace(r"[^0-9eE\.\+\-]", "", regex=True)
    )
    return pd.to_numeric(ss, errors="coerce")

def to_datetime_col_robust(s: pd.Series) -> pd.Series:
    # numeric epoch (s or ms)?
    vals = s.dropna()
    if len(vals) and np.issubdtype(vals.values[:1].dtype, np.number):
        v = s.astype("float64")
        unit = "ms" if (v.dropna().abs() > 1e12).any() else "s"
        dt = pd.to_datetime(v, unit=unit, errors="coerce", utc=False)
    else:
        ss = s.astype(str).str.strip()
        # first try normal with inference
        dt = pd.to_datetime(ss, errors="coerce", utc=False, infer_datetime_format=True)
        if dt.notna().sum() == 0:
            # try dayfirst as a fallback
            dt = pd.to_datetime(ss, errors="coerce", utc=False, dayfirst=True)
    return dt

def _has_any_candidates(cols, cands_norm) -> bool:
    colset = {_norm(c) for c in cols}
    return any(_norm(c) in colset for c in cands_norm)

def _find_header_row_flex(path: str, encoding: str) -> Optional[int]:
    """Scan first ~50 lines to find the header row that contains any candidate names."""
    try:
        raw = pd.read_csv(path, sep=None, engine="python", header=None,
                          nrows=50, encoding=encoding, on_bad_lines="skip")
    except Exception:
        return None
    # all candidate names (normalized)
    all_cands = (
        ELEC_TIME_CANDS + CPC_TIME_CANDS + CPC_CANDS + ELEC_CANDS
    )
    all_cands_norm = {_norm(c) for c in all_cands}
    for i in range(len(raw)):
        row = raw.iloc[i].astype(str).tolist()
        if _has_any_candidates(row, all_cands_norm):
            return i
    return None

def _read_csv_flex(path: str) -> pd.DataFrame:
    """Robust CSV reader: infer delimiter, detect header row, try a couple encodings."""
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            df = pd.read_csv(path, sep=None, engine="python", encoding=enc,
                             on_bad_lines="skip")
            # if headers didn’t match, try to locate the header row
            all_cands = ELEC_TIME_CANDS + CPC_TIME_CANDS + CPC_CANDS + ELEC_CANDS
            if not _has_any_candidates(df.columns, {_norm(c) for c in all_cands}):
                header_row = _find_header_row_flex(path, enc)
                if header_row is not None:
                    df = pd.read_csv(path, sep=None, engine="python", encoding=enc,
                                     header=header_row, on_bad_lines="skip")
            return df
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err

def pick_col_norm(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Pick a column by normalized name (case/space/underscore-insensitive)."""
    cols_norm = {_norm(c): c for c in df.columns}
    for nm in candidates:
        n = _norm(nm)
        if n in cols_norm:
            return cols_norm[n]
    # light fuzzy: allow candidate to be contained in column or vice-versa
    for ncol, orig in cols_norm.items():
        for nm in candidates:
            ncan = _norm(nm)
            if ncan in ncol or ncol in ncan:
                return orig
    return None

def load_joint_csv(path: str) -> pd.DataFrame:
    df = _read_csv_flex(path)

    # pick time columns per instrument (normalized match)
    elec_t = pick_col_norm(df, ELEC_TIME_CANDS)
    cpc_t  = pick_col_norm(df,  CPC_TIME_CANDS)

    # pick concentration columns
    ccol = pick_col_norm(df, CPC_CANDS)
    ecol = pick_col_norm(df, ELEC_CANDS)

    if elec_t is None and cpc_t is None:
        raise ValueError(
            f"No time columns found in {os.path.basename(path)}.\n"
            f"Available columns: {list(df.columns)[:10]}..."
        )

    out = pd.DataFrame()

    # times
    out["time_elec"] = to_datetime_col_robust(df[elec_t]) if elec_t else pd.NaT
    out["time_cpc"]  = to_datetime_col_robust(df[cpc_t])  if cpc_t  else pd.NaT

    if not TIMEZONE_AWARE:
        for tcol in ("time_elec","time_cpc"):
            try:
                out[tcol] = out[tcol].dt.tz_localize(None)
            except Exception:
                pass

    # concentrations (robust numeric)
    out["elec"] = _to_numeric_smart(df[ecol]) if ecol else np.nan
    out["cpc"]  = _to_numeric_smart(df[ccol]) if ccol else np.nan

    # keep original row count; plotting filters NaTs
    if VERBOSE:
        print(
            f"[{os.path.basename(path)}] "
            f"EM time={out['time_elec'].notna().sum()}  EM conc={pd.notna(out['elec']).sum()}  "
            f"CPC time={out['time_cpc'].notna().sum()}  CPC conc={pd.notna(out['cpc']).sum()}  "
            f"| cols picked: elec_t={elec_t}, cpc_t={cpc_t}, elec={ecol}, cpc={ccol}"
        )
    return out


def plot_overlays_for_temp(temp: int, paths: List[str], cpc_name: str, outdir: str):
    if not paths:
        print(f"[{temp}°C] (skip) No files chosen.")
        return

    reps = []
    for i, p in enumerate(paths, 1):
        ts = load_joint_csv(p)
        ts["repeat"] = i
        ts["file"] = os.path.basename(p)
        ts = maybe_smooth_two_clocks(ts, SMOOTH_MINUTES)

        # ---- build relative-time axis per chosen stream (robust) ----
        t_rel_cpc = np.full(len(ts), np.nan)
        t_rel_elec = np.full(len(ts), np.nan)

        # valid subsets
        cpc_valid = ts.dropna(subset=["time_cpc"]).copy()
        em_valid = ts.dropna(subset=["time_elec"]).copy()
        cpc_has_y = ts.dropna(subset=["time_cpc", "cpc"]).shape[0] > 0
        em_has_y = ts.dropna(subset=["time_elec", "elec"]).shape[0] > 0

        # reference anchors
        cpc_t0_abs = cpc_valid["time_cpc"].iloc[0] if not cpc_valid.empty else None
        em_t0_abs = em_valid["time_elec"].iloc[0] if not em_valid.empty else None

        # detect steps (seconds since each stream's own start)
        cpc_step_s = None
        em_step_s = None
        if cpc_has_y:
            secs = _to_seconds_since_start(ts.loc[cpc_valid.index, "time_cpc"])
            cpc_step_s = detect_step_time(secs, ts.loc[cpc_valid.index, "cpc"])
        if em_has_y:
            secs = _to_seconds_since_start(ts.loc[em_valid.index, "time_elec"])
            em_step_s = detect_step_time(secs, ts.loc[em_valid.index, "elec"])

        if ALIGN_TO_STEP and ALIGN_STREAM == "cpc" and cpc_t0_abs is not None:
            # align both streams to CPC step if found; else to CPC series start
            anchor = cpc_step_s if cpc_step_s is not None else 0.0
            t_rel_cpc = (pd.to_datetime(ts["time_cpc"]) - cpc_t0_abs).dt.total_seconds() - anchor
            if em_t0_abs is not None:
                t_rel_elec = (pd.to_datetime(ts["time_elec"]) - cpc_t0_abs).dt.total_seconds() - anchor
        elif ALIGN_TO_STEP and ALIGN_STREAM == "elec" and em_t0_abs is not None:
            # align both streams to EM step if found; else to EM series start
            anchor = em_step_s if em_step_s is not None else 0.0
            t_rel_elec = (pd.to_datetime(ts["time_elec"]) - em_t0_abs).dt.total_seconds() - anchor
            if cpc_t0_abs is not None:
                t_rel_cpc = (pd.to_datetime(ts["time_cpc"]) - em_t0_abs).dt.total_seconds() - anchor
        else:
            # alignment off or missing anchor: fall back to each stream's own start
            if cpc_t0_abs is not None:
                t_rel_cpc = (pd.to_datetime(ts["time_cpc"]) - cpc_t0_abs).dt.total_seconds()
            if em_t0_abs is not None:
                t_rel_elec = (pd.to_datetime(ts["time_elec"]) - em_t0_abs).dt.total_seconds()

        ts["t_rel_cpc"] = t_rel_cpc
        ts["t_rel_elec"] = t_rel_elec
        reps.append(ts)

    data = pd.concat(reps, ignore_index=True)


    # Stats per repeat (unchanged logic, just ignoring NaNs)
    print(f"\n==== {cpc_name} | {temp}°C ====")
    for r, sub in data.groupby("repeat"):
        c_mean, c_std, c_cv = variability_stats(sub["cpc"])
        e_mean, e_std, e_cv = variability_stats(sub["elec"])
        print(f"Repeat {r:>2}: CPC mean={c_mean:.3g}, std={c_std:.3g}, CV={c_cv:.2%} | "
              f"Elec mean={e_mean:.3g}, std={e_std:.3g}, CV={e_cv:.2%}")

    # Across-repeat CV over common grid — compute separately for CPC vs EM using their own clocks
    try:
        # CPC grid
        resampled = []
        for r, sub in data.groupby("repeat"):
            cc = sub[["time_cpc","cpc"]].dropna(subset=["time_cpc"]).set_index("time_cpc").sort_index()
            if not cc.empty:
                rr = cc.resample(RESAMPLE_RULE).mean()
                rr["repeat"] = r
                resampled.append(rr)
        if resampled:
            grid = pd.concat(resampled).reset_index()
            cpc_piv = grid.pivot_table(index="time_cpc", columns="repeat", values="cpc")
            cpc_cv_t = cpc_piv.std(axis=1, ddof=1) / cpc_piv.mean(axis=1)
            print(f"Across-repeats CPC CV (median over time):  {np.nanmedian(cpc_cv_t)*100:0.1f}%")
    except Exception as e:
        print(f"(Note) CPC CV grid issue: {e}")

    try:
        # Electrometer grid
        resampled = []
        for r, sub in data.groupby("repeat"):
            em = sub[["time_elec","elec"]].dropna(subset=["time_elec"]).set_index("time_elec").sort_index()
            if not em.empty:
                rr = em.resample(RESAMPLE_RULE).mean()
                rr["repeat"] = r
                resampled.append(rr)
        if resampled:
            grid = pd.concat(resampled).reset_index()
            elec_piv = grid.pivot_table(index="time_elec", columns="repeat", values="elec")
            elec_cv_t = elec_piv.std(axis=1, ddof=1) / elec_piv.mean(axis=1)
            print(f"Across-repeats Elec CV (median over time): {np.nanmedian(elec_cv_t)*100:0.1f}%")
    except Exception as e:
        print(f"(Note) Electrometer CV grid issue: {e}")

    stamp = datetime.now().strftime("%Y%m%d")
    # ---- CPC aligned overlay ----
    cpc_any = ("cpc" in data and data["cpc"].notna().any() and
               (("t_rel_cpc" in data and data["t_rel_cpc"].notna().any()) or not ALIGN_TO_STEP))
    if cpc_any:
        plt.figure(figsize=(10, 4.6))
        for r, sub in data.groupby("repeat"):
            if ALIGN_TO_STEP:
                g = sub.dropna(subset=["t_rel_cpc", "cpc"]).sort_values("t_rel_cpc")
                x = g["t_rel_cpc"]
            else:
                g = sub.dropna(subset=["time_cpc", "cpc"]).sort_values("time_cpc")
                x = (g["time_cpc"] - g["time_cpc"].iloc[0]).dt.total_seconds()  # fallback: relative to series start
            if not g.empty:
                plt.plot(x, g["cpc"], linewidth=1.2, label=f"Repeat {r} – {g['file'].iloc[0]}")
        plt.title(f"{cpc_name} {temp}°C — CPC (aligned at step)")
        plt.xlabel("Time since step (s)");
        plt.ylabel("CPC concentration")
        plt.grid(True, alpha=0.4);
        plt.legend(ncol=2, fontsize=9)
        if ALIGN_TO_STEP:
            xmin = -float(PRE_WINDOW_SEC)
            xmax = XLIM_RIGHT_SEC if XLIM_RIGHT_SEC is not None else None
            plt.xlim(left=xmin, right=xmax)
        plt.tight_layout()
        out = os.path.join(outdir, f"{stamp}_{cpc_name}_{temp}C_CPC_timeseries_aligned.png")
        plt.savefig(out, dpi=220);
        print(f"Saved: {out}")
        plt.show()

    # ---- Electrometer aligned overlay ----
    elec_any = ("elec" in data and data["elec"].notna().any())

    if elec_any:
        plt.figure(figsize=(10, 4.6))
        for r, sub in data.groupby("repeat"):
            if ALIGN_TO_STEP:
                g = sub.dropna(subset=["t_rel_elec", "elec"]).sort_values("t_rel_elec")
                x = g["t_rel_elec"]
            else:
                g = sub.dropna(subset=["time_elec", "elec"]).sort_values("time_elec")
                x = (g["time_elec"] - g["time_elec"].iloc[0]).dt.total_seconds()
            if not g.empty:
                plt.plot(x, g["elec"], linewidth=1.2, label=f"Repeat {r} – {g['file'].iloc[0]}")
        plt.title(f"{cpc_name} {temp}°C — Electrometer (aligned at step)")
        plt.xlabel("Time since step (s)");
        plt.ylabel("Electrometer concentration")
        plt.grid(True, alpha=0.4);
        plt.legend(ncol=2, fontsize=9)
        if ALIGN_TO_STEP:
            xmin = -float(PRE_WINDOW_SEC)
            xmax = XLIM_RIGHT_SEC if XLIM_RIGHT_SEC is not None else None
            plt.xlim(left=xmin, right=xmax)
        plt.tight_layout()
        out = os.path.join(outdir, f"{stamp}_{cpc_name}_{temp}C_Elec_timeseries_aligned.png")
        plt.savefig(out, dpi=220);
        print(f"Saved: {out}")
        plt.show()





# ------------- MAIN (GUI FLOW) -------------
def main():
    root = tk.Tk()
    root.withdraw()

    # 1) CPC name
    cpc_name = simpledialog.askstring("CPC Name", "Enter CPC name (e.g., Ambrosia):")
    if not cpc_name:
        messagebox.showinfo("Canceled", "No CPC name provided.")
        return

    # 2) Temperatures list
    temps_str = simpledialog.askstring("Temperatures",
        "Enter initiator temperatures (comma-separated, e.g. 98,91,81):")
    if not temps_str:
        messagebox.showinfo("Canceled", "No temperatures provided.")
        return
    try:
        temps = [int(x.strip()) for x in temps_str.split(",") if x.strip() != ""]
        if not temps:
            raise ValueError
    except Exception:
        messagebox.showerror("Error", "Could not parse temperatures. Use comma-separated integers.")
        return

    outdir = ensure_outdir(OUTPUT_ROOT)

    messagebox.showinfo(
        "Select Files",
        "You will now select joint CSV files for each temperature.\n"
        "Each selection can include multiple files (one per repeat)."
    )

    files_per_temp: Dict[int, List[str]] = {}
    for t in temps:
        messagebox.showinfo(
            f"{t}°C",
            f"Select the joint CSV(s) for {t}°C (choose all repeats at once)."
        )
        paths = filedialog.askopenfilenames(
            title=f"Select joint CSV(s) for {t}°C",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        paths = list(paths)
        files_per_temp[t] = paths
        if not paths:
            print(f"[{t}°C] No files selected.")

    # Process & plot
    for t, paths in files_per_temp.items():
        try:
            plot_overlays_for_temp(t, paths, cpc_name, outdir)
        except Exception as e:
            messagebox.showerror("Error", f"Failed on {t}°C: {e}")

    messagebox.showinfo("Done", "All plots done. See the console for stats and the Graphs/ folder for PNGs.")

if __name__ == "__main__":
    main()
