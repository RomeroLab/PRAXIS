"""
Enzyme-kinetics fitting functions for the Bgl parent + variant MM curves.

The fitting core is identical between them; the two pipelines differ only in:

  * standard-curve loading
      - parents  -> `load_standard_curve_gain`     (per-gain cell positions; gain-80
                                                     for glucose, gain-100 for xyl/man)
      - variants -> `load_standard_curve_extended`  (single extended-range curve)
  * time ranges
      - parents  -> inline dict (min_hr, max_hr) -> pass seconds to calculate_v0
      - variants -> `load_time_ranges` / `get_time_range` from time_ranges.csv (minutes)

Velocity/fit functions are shared. `calculate_v0_for_compiled` supports optional
per-[S] time-range overrides (used by the variant pipeline).
"""
import datetime as dt
import os
import re

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress, t as t_dist


# ============================================================================
# UTILITIES
# ============================================================================
def excel_col_to_index(col_letter):
    """Excel column letter -> zero-based index."""
    index = 0
    for char in col_letter:
        index = index * 26 + (ord(char.upper()) - ord('A') + 1)
    return index - 1


def time_to_seconds(t):
    """Time -> seconds. Handles HH:MM:SS strings, datetime.time, timedelta, floats."""
    if pd.isna(t):
        return np.nan
    if isinstance(t, dt.time):
        return t.hour * 3600 + t.minute * 60 + t.second
    if isinstance(t, dt.timedelta):
        return t.total_seconds()
    if isinstance(t, str):
        parts = t.strip().split(':')
        try:
            if len(parts) == 3:
                h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
                return h * 3600 + m * 60 + s
            elif len(parts) == 2:
                m, s = int(parts[0]), int(parts[1])
                return m * 60 + s
        except ValueError:
            pass
    try:
        return float(t)
    except Exception:
        return np.nan


# ============================================================================
# STANDARD CURVES  (two loaders — do NOT cross-pollinate across gain settings)
# ============================================================================
def load_standard_curve_gain(filepath, substrate='glucose'):
    """Parent std curve: read fixed-gain cells from the raw plate export.

    glucose -> gain 80 rows; xylose/mannose -> gain 100 rows.
    Returns (slope, intercept, r_squared) for fluorescence -> mM.
    """
    if substrate.lower() == 'glucose':                      # gain 80
        cells = [('C', 60), ('C', 63), ('C', 66), ('C', 69), ('C', 72), ('C', 75), ('C', 78), ('C', 81),
                 ('D', 60), ('D', 63), ('D', 66), ('D', 69), ('D', 72), ('D', 75), ('D', 78), ('D', 81)]
    else:                                                    # gain 100
        cells = [('C', 61), ('C', 64), ('C', 67), ('C', 70), ('C', 73), ('C', 76), ('C', 79), ('C', 82),
                 ('D', 61), ('D', 64), ('D', 67), ('D', 70), ('D', 73), ('D', 76), ('D', 79), ('D', 82)]

    df_std = pd.read_excel(filepath, header=None)
    fluor = [df_std.iloc[r - 1, excel_col_to_index(c)] for c, r in cells]

    conc_mM = [5.0 / (2 ** i) for i in range(16)]           # 2-fold serial dilution from 5 mM
    std = pd.DataFrame({'concentration_mM': conc_mM, 'fluorescence': fluor})
    std['fluorescence'] = pd.to_numeric(std['fluorescence'], errors='coerce')
    std = std.dropna()
    slope, intercept, r, _, _ = linregress(std['fluorescence'], std['concentration_mM'])
    return slope, intercept, r ** 2


def load_standard_curve_extended(filepath):
    """Variant std curve: extended-range compiled curve (single gain for all substrates).

    Expects columns concentration_uM, rep1_fluorescence, rep2_fluorescence.
    Excludes the top point (>=62.5 uM, detector saturation).
    Returns (slope, intercept, r_squared) for fluorescence -> mM.
    """
    df = pd.read_excel(filepath)
    conc_uM = np.concatenate([df['concentration_uM'].values, df['concentration_uM'].values])
    fluor = np.concatenate([
        pd.to_numeric(df['rep1_fluorescence'], errors='coerce').values,
        pd.to_numeric(df['rep2_fluorescence'], errors='coerce').values,
    ])
    conc_mM = conc_uM / 1000.0
    valid = np.isfinite(conc_mM) & np.isfinite(fluor) & (conc_uM < 62.5)
    slope, intercept, r, _, _ = linregress(fluor[valid], conc_mM[valid])
    return slope, intercept, r ** 2


# ============================================================================
# TIME RANGES  (variant pipeline: from time_ranges.csv, in MINUTES)
# ============================================================================
def load_time_ranges(filepath):
    """Parse time_ranges.csv -> (defaults, overrides, enzyme_concs).

    Columns: enzyme, substrate, concentration, t_min_min, t_max_min, enzyme_nM (minutes).
      defaults     : {(enzyme, substrate): (t_min_sec, t_max_sec)}
      overrides    : {(enzyme, substrate, concentration): (t_min_sec, t_max_sec)}
      enzyme_concs : {(enzyme, substrate): nM}
    """
    defaults, overrides, enzyme_concs = {}, {}, {}
    df = pd.read_csv(filepath)
    for _, row in df.iterrows():
        enzyme = str(row['enzyme']).strip().lower()
        substrate = str(row['substrate']).strip().lower()

        enm = row.get('enzyme_nM')
        if enm is not None and not pd.isna(enm) and enm != '':
            enzyme_concs[(enzyme, substrate)] = float(enm)

        t_min, t_max = row.get('t_min_min'), row.get('t_max_min')
        if pd.isna(t_min) or pd.isna(t_max) or t_min == '' or t_max == '':
            continue
        tr = (float(t_min) * 60, float(t_max) * 60)

        conc = row.get('concentration')
        if pd.isna(conc) or conc == '':
            defaults[(enzyme, substrate)] = tr
        else:
            overrides[(enzyme, substrate, float(conc))] = tr
    return defaults, overrides, enzyme_concs


def get_time_range(enzyme, substrate, concentration, tr_defaults, tr_overrides):
    """Resolve the time range for one (enzyme, substrate, [S]); None => first 3 timepoints."""
    if (enzyme, substrate, concentration) in tr_overrides:
        return tr_overrides[(enzyme, substrate, concentration)]
    return tr_defaults.get((enzyme, substrate))


# ============================================================================
# DATA LOADING + BACKGROUND
# ============================================================================
def load_compiled_data(filepath):
    """Load a compiled {conc}_R{rep} file; adds a time_sec column."""
    df = pd.read_excel(filepath)
    df['time_sec'] = df[df.columns[0]].apply(time_to_seconds)
    return df


def load_background_raw(compiled_dir, substrates=('glu', 'xyl', 'man')):
    """Load negative-control frames neg_{sub}.xlsx -> {substrate: DataFrame}."""
    bkg = {}
    for sub in substrates:
        fp = os.path.join(compiled_dir, f"neg_{sub}.xlsx")
        if os.path.exists(fp):
            bkg[sub] = load_compiled_data(fp)
    return bkg


def calculate_v0_for_compiled(df, slope, intercept, time_range=None,
                              time_range_by_conc=None, bkg_df=None):
    """Initial velocities from compiled {conc}_R{rep} data.

    Background is subtracted at the fluorescence level before conversion to mM.
    `time_range` is (min_sec, max_sec) or None (first 3 timepoints); a per-[S]
    override in `time_range_by_conc` takes precedence.
    Returns a DataFrame: concentration, v0_uM_per_s, v0_sem_uM_per_s, rep_v0_uM_per_s,
    n_replicates, rep_r2s.
    """
    if time_range_by_conc is None:
        time_range_by_conc = {}
    time_col = df.columns[0]
    pattern = re.compile(r'([\d\.]+)_R(\d+)')

    conc_groups = {}
    for col in df.columns:
        if col in (time_col, 'time_sec'):
            continue
        m = pattern.match(str(col))
        if m:
            conc_groups.setdefault(float(m.group(1)), []).append(col)

    bkg_mean = {}
    if bkg_df is not None:
        bkg_time_col = bkg_df.columns[0]
        bkg_groups = {}
        for col in bkg_df.columns:
            if col in (bkg_time_col, 'time_sec'):
                continue
            m = pattern.match(str(col))
            if m:
                bkg_groups.setdefault(float(m.group(1)), []).append(col)
        for conc, cols in bkg_groups.items():
            vals = []
            for col in cols:
                bkg_df[col] = pd.to_numeric(bkg_df[col], errors='coerce')
                vals.append(bkg_df[col].values)
            bkg_mean[conc] = np.nanmean(vals, axis=0)

    results = []
    for conc in sorted(conc_groups.keys(), reverse=True):
        velocities, rep_r2s = [], []
        tr = time_range_by_conc.get(conc, time_range)

        for col in conc_groups[conc]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if tr is not None:
                mask = (df['time_sec'] >= tr[0]) & (df['time_sec'] <= tr[1])
                t = df.loc[mask, 'time_sec'].values
                y = df.loc[mask, col].values
                if bkg_df is not None and conc in bkg_mean:
                    bkg_mask = (bkg_df['time_sec'] >= tr[0]) & (bkg_df['time_sec'] <= tr[1])
                    bkg_y = bkg_mean[conc][bkg_mask.values]      # mask on the bkg frame, not the enzyme frame
                else:
                    bkg_y = None
            else:
                t = df['time_sec'].iloc[:3].values
                y = df[col].iloc[:3].values
                bkg_y = bkg_mean[conc][:3] if (bkg_df is not None and conc in bkg_mean) else None

            y_num = pd.to_numeric(pd.Series(y), errors='coerce').values
            if bkg_y is not None and len(bkg_y) == len(y_num):
                y_num = y_num - bkg_y

            valid = np.isfinite(y_num) & np.isfinite(t)
            t_clean, y_clean = t[valid], y_num[valid]
            if len(np.unique(t_clean)) >= 2:
                conc_mM = slope * y_clean + intercept
                s, _, r, _, _ = linregress(t_clean, conc_mM)
                rep_r2s.append(r ** 2)
                velocities.append(s)                        # mM/s

        v_mean = np.mean(velocities) if velocities else np.nan
        v_std = np.std(velocities, ddof=1) if len(velocities) > 1 else np.nan
        v_sem = v_std / np.sqrt(len(velocities)) if velocities else np.nan
        results.append({
            'concentration': conc,
            'v0_uM_per_s': v_mean * 1000,
            'v0_sem_uM_per_s': v_sem * 1000 if np.isfinite(v_sem) else np.nan,
            'rep_v0_uM_per_s': [v * 1000 for v in velocities],
            'n_replicates': len(velocities),
            'rep_r2s': rep_r2s,
        })
    return pd.DataFrame(results)


# ============================================================================
# MODELS
# ============================================================================
def substrate_inhibition(S, Vmax, Km, Ki):
    """v = Vmax*[S] / (Km + [S] + [S]^2/Ki)  (Haldane)."""
    S = np.asarray(S, dtype=float)
    return (Vmax * S) / (Km + S + (S ** 2 / Ki))


def michaelis_menten(S, Vmax, Km):
    """v = Vmax*[S] / (Km + [S])."""
    S = np.asarray(S, dtype=float)
    return (Vmax * S) / (Km + S)


# ============================================================================
# INITIAL SLOPE (model-independent kcat/Km)
# ============================================================================
def estimate_initial_slope(conc, v0, n_points=3):
    """OLS slope of v0 vs [S] over the n lowest-[S] points -> Vmax/Km (s^-1).

    Returns {slope, slope_SE, R2_linear, n_used} or None.
    """
    order = np.argsort(conc)
    x, y = conc[order], v0[order]
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = min(n_points, len(x))
    if n < 2:
        return None
    x, y = x[:n], y[:n]
    slope, intercept, r, _, se = linregress(x, y)
    if slope <= 0:
        return None
    return {'slope': slope, 'slope_SE': se, 'R2_linear': r ** 2, 'n_used': n}


# ============================================================================
# FITTING
# ============================================================================
def fit_substrate_inhibition(conc, v0, sem, enzyme_conc_uM):
    """Unconstrained 3-parameter substrate-inhibition fit (grid of initial guesses)."""
    valid = np.isfinite(conc) & np.isfinite(v0)
    conc, v0, sem = conc[valid], v0[valid], sem[valid]
    if len(conc) < 4:
        return None

    peak_idx = np.nanargmax(v0)
    peak_conc, peak_v0, max_conc = conc[peak_idx], v0[peak_idx], np.max(conc)
    lower = [1e-10, 0.01, 1.0]
    upper = [peak_v0 * 50, max_conc * 10, max_conc * 100]

    Vmax_guesses = peak_v0 * np.array([0.8, 1.2, 2.0, 5.0, 10.0])
    Km_guesses = np.unique(np.clip(
        [1, 5, 10, 25, 50, 100, 250, 500, peak_conc * 0.5, peak_conc, peak_conc * 2],
        lower[1], upper[1]))
    Ki_guesses = np.unique(np.clip(
        [50, 100, 250, 500, 1000, 2500, 5000, peak_conc * 2, peak_conc * 5, max_conc, max_conc * 5],
        lower[2], upper[2]))

    sem_valid = np.isfinite(sem) & (sem > 0)
    weight_options = [(None, False)]
    if np.all(sem_valid):
        weight_options.append((sem, True))

    best_popt = best_pcov = None
    best_ss = np.inf
    for sigma, abs_sigma in weight_options:
        for Vg in Vmax_guesses:
            for Kg in Km_guesses:
                for Ig in Ki_guesses:
                    try:
                        popt, pcov = curve_fit(
                            substrate_inhibition, conc, v0, p0=[float(Vg), float(Kg), float(Ig)],
                            sigma=sigma, absolute_sigma=abs_sigma, bounds=(lower, upper), maxfev=50000)
                        ss = np.sum((v0 - substrate_inhibition(conc, *popt)) ** 2)
                        if ss < best_ss:
                            best_ss, best_popt, best_pcov = ss, popt, pcov
                    except (RuntimeError, ValueError):
                        continue
    if best_popt is None:
        return None

    Vmax_fit, Km_fit, Ki_fit = best_popt
    param_se = np.sqrt(np.diag(best_pcov))
    pred = substrate_inhibition(conc, *best_popt)
    ss_tot = np.sum((v0 - np.mean(v0)) ** 2)
    r2 = 1 - np.sum((v0 - pred) ** 2) / ss_tot if ss_tot > 0 else np.nan

    dof = max(0, len(conc) - 3)
    t_val = t_dist.ppf(0.975, dof) if dof > 0 else np.nan
    ci = param_se * t_val if np.isfinite(t_val) else np.full(3, np.nan)

    kcat = Vmax_fit / enzyme_conc_uM if enzyme_conc_uM > 0 else np.nan
    kcat_se = param_se[0] / enzyme_conc_uM if enzyme_conc_uM > 0 else np.nan
    kcat_Km = (kcat / Km_fit) * 1e6 if Km_fit > 0 else np.nan
    if np.isfinite(kcat_Km) and kcat > 0 and Km_fit > 0 and np.isfinite(kcat_se) and np.isfinite(param_se[1]):
        rel = np.sqrt((kcat_se / kcat) ** 2 + (param_se[1] / Km_fit) ** 2)
        kcat_Km_se = kcat_Km * rel
    else:
        kcat_Km_se = np.nan

    return {
        'Vmax_uM_s': Vmax_fit, 'Vmax_SE': param_se[0], 'Vmax_CI': ci[0],
        'Km_uM': Km_fit, 'Km_SE': param_se[1], 'Km_CI': ci[1],
        'Ki_uM': Ki_fit, 'Ki_SE': param_se[2], 'Ki_CI': ci[2],
        'kcat_s': kcat, 'kcat_SE': kcat_se,
        'kcat_Km_M': kcat_Km, 'kcat_Km_SE': kcat_Km_se,
        'R2': r2, 'popt': best_popt, 'pcov': best_pcov,
        'conc': conc, 'v0': v0, 'sem': sem, 'n_points': len(conc),
    }


def fit_si_constrained(conc, v0, sem, enzyme_conc_uM, alpha, alpha_se):
    """2-parameter substrate-inhibition fit with Vmax/Km fixed to the initial slope `alpha`.

    Model: v = alpha*Km*S / (Km + S + S^2/Ki). Breaks the Vmax-Km correlation;
    kcat/Km is taken directly from alpha. Returns a 3-param popt for plotting.
    """
    valid = np.isfinite(conc) & np.isfinite(v0)
    conc, v0, sem = conc[valid], v0[valid], sem[valid]
    if len(conc) < 3:
        return None

    def model(S, Km, Ki):
        return (alpha * Km * S) / (Km + S + S ** 2 / Ki)

    peak_idx = np.nanargmax(v0)
    peak_conc, max_conc = conc[peak_idx], np.max(conc)
    lower = [0.01, 1.0]
    upper = [max_conc * 10, max_conc * 100]
    Km_guesses = np.unique(np.clip(
        [1, 5, 10, 25, 50, 100, 250, 500, peak_conc * 0.5, peak_conc, peak_conc * 2], lower[0], upper[0]))
    Ki_guesses = np.unique(np.clip(
        [50, 100, 250, 500, 1000, 2500, 5000, peak_conc * 2, peak_conc * 5, max_conc, max_conc * 5], lower[1], upper[1]))

    sem_valid = np.isfinite(sem) & (sem > 0)
    weight_options = [(None, False)]
    if np.all(sem_valid):
        weight_options.append((sem, True))

    best_popt = best_pcov = None
    best_ss = np.inf
    for sigma, abs_sigma in weight_options:
        for Kg in Km_guesses:
            for Ig in Ki_guesses:
                try:
                    popt, pcov = curve_fit(
                        model, conc, v0, p0=[float(Kg), float(Ig)],
                        sigma=sigma, absolute_sigma=abs_sigma, bounds=(lower, upper), maxfev=50000)
                    ss = np.sum((v0 - model(conc, *popt)) ** 2)
                    if ss < best_ss:
                        best_ss, best_popt, best_pcov = ss, popt, pcov
                except (RuntimeError, ValueError):
                    continue
    if best_popt is None:
        return None

    Km_fit, Ki_fit = best_popt
    Vmax_fit = alpha * Km_fit
    param_se = np.sqrt(np.diag(best_pcov))
    pred = model(conc, *best_popt)
    ss_tot = np.sum((v0 - np.mean(v0)) ** 2)
    r2 = 1 - np.sum((v0 - pred) ** 2) / ss_tot if ss_tot > 0 else np.nan

    Vmax_SE = alpha * param_se[0]
    kcat = Vmax_fit / enzyme_conc_uM if enzyme_conc_uM > 0 else np.nan
    kcat_se = Vmax_SE / enzyme_conc_uM if enzyme_conc_uM > 0 else np.nan
    kcat_Km = (alpha / enzyme_conc_uM) * 1e6 if enzyme_conc_uM > 0 else np.nan
    kcat_Km_se = (alpha_se / enzyme_conc_uM) * 1e6 if (enzyme_conc_uM > 0 and np.isfinite(alpha_se)) else np.nan

    dof = max(0, len(conc) - 2)
    t_val = t_dist.ppf(0.975, dof) if dof > 0 else np.nan
    ci = param_se * t_val if np.isfinite(t_val) else np.full(2, np.nan)

    return {
        'Vmax_uM_s': Vmax_fit, 'Vmax_SE': Vmax_SE,
        'Km_uM': Km_fit, 'Km_SE': param_se[0], 'Km_CI': ci[0],
        'Ki_uM': Ki_fit, 'Ki_SE': param_se[1], 'Ki_CI': ci[1],
        'kcat_s': kcat, 'kcat_SE': kcat_se,
        'kcat_Km_M': kcat_Km, 'kcat_Km_SE': kcat_Km_se,
        'R2': r2, 'popt': np.array([Vmax_fit, Km_fit, Ki_fit]), 'pcov': best_pcov,
        'conc': conc, 'v0': v0, 'sem': sem, 'n_points': len(conc), 'constrained': True,
    }
