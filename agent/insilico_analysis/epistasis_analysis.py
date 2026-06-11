#!/usr/bin/env python
"""
Epistasis analysis for in-silico benchmark datasets (GFP, GB1, PAB1, UBE4).

Prints (to stdout; no files written) a characterization of local and global
epistasis across all mutation orders, for the full landscape and the top
fitness-peak regions.

Analyses:
  1. Additive R² and global-epistasis R² (full landscape + top 5% / 1%)
  2. Additive- and global-model residuals / R² stratified by mutation count
  3. Face epistasis: magnitude / sign / reciprocal-sign classification over every
     measured 2D square of the genotype hypercube, at all mutation orders, using a
     local reference background (WT-anchored doubles are the special case R = WT),
     reported overall, by reference order |R|, and by fitness region.

Run:
  python insilico_analysis/epistasis_analysis.py
"""

import os, warnings, itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS = {
    "GFP": {
        "path": os.path.join(ROOT, "insilico_analysis/dataset_oracle/gfp/gfp_SeqFxnDataset.pkl"),
        "wt_fitness": 3.7,
        "wt": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
    },
    "GB1": {
        "path": os.path.join(ROOT, "insilico_analysis/dataset_oracle/gb1/gb1_SeqFxnDataset.pkl"),
        "wt_fitness": 0.0,
        "wt": "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
    },
    "PAB1": {
        "path": os.path.join(ROOT, "insilico_analysis/dataset_oracle/pab1/pab1_SeqFxnDataset.pkl"),
        "wt_fitness": 0.0,  # log-enrichment scale; Melamed et al. normalize WT enrichment to 1.0 => log(1)=0
        "wt": "GNIFIKNLHPDIDNKALYDTFSVFGDILSSKIATDENGKSKGFGFVHFEEEGAAKEAIDALNGMLLNGQEIYVAP",
    },
    "UBE4": {
        "path": os.path.join(ROOT, "insilico_analysis/dataset_oracle/ube4/ube4_SeqFxnDataset.pkl"),
        "wt_fitness": 0.084,  # log2-enrichment scale; Starita et al. report WT enrichment E=1.06 => log2(1.06)≈0.084
        "wt": "IEKFKLLAEKVEEIVAKNARAEIDYSDAPDEFRDPLMDTLMTDPVRLPSGTVMDRSIILRHLLNSPTDPFNRQMLTESMLEPVPELKEQIQAWMREKQSSDH",
    },
}

TOP_PCTS = [0.05, 0.01]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_mutations(mut_str):
    if pd.isna(mut_str) or mut_str == "" or mut_str == "WT":
        return []
    return mut_str.split(",")


def build_mutation_matrix(df):
    """One-hot encode individual mutations across all variants."""
    all_muts = set()
    parsed = []
    for m in df["mutations"]:
        p = parse_mutations(m)
        parsed.append(p)
        all_muts.update(p)
    all_muts = sorted(all_muts)
    mut_to_idx = {m: i for i, m in enumerate(all_muts)}

    X = np.zeros((len(df), len(all_muts)), dtype=np.float32)
    for i, p in enumerate(parsed):
        for m in p:
            X[i, mut_to_idx[m]] = 1.0
    return X, all_muts, mut_to_idx


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return 1 - ss_res / ss_tot


def fit_additive(X, y, regularize=False):
    model = Ridge(alpha=1.0) if regularize else LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    return model, y_pred, r2_score(y, y_pred)


def fit_global_epistasis(X, y):
    """Additive + isotonic nonlinear transform."""
    lin = LinearRegression()
    lin.fit(X, y)
    latent = lin.predict(X)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(latent, y)
    y_pred = iso.predict(latent)
    return latent, y_pred, r2_score(y, y_pred), lin, iso


def face_epistasis(df, wt_fitness, eps_tol=1e-8):
    """Epistasis over every measured 2D face (square) of the genotype hypercube.

    For a reference background R and two mutations a, b, the four corners
    R, R+a, R+b, R+a+b form a square. When all four are measured we compute
        epsilon = f(R+a+b) - f(R+a) - f(R+b) + f(R)
    and classify it relative to the local reference R:
        sign_a flips  <=>  sign(f(R+a) - f(R)) != sign(f(R+a+b) - f(R+b))   (and likewise b)
        reciprocal_sign : both a and b flip      sign : exactly one flips
        magnitude       : neither flips, |eps| > tol
    WT-anchored doubles are the special case R = WT (empty background). Each square
    is enumerated exactly once (its top corner R+a+b and the pair {a, b}).
    Returns a DataFrame with columns: ref_order (|R|), epsilon, type, f_top, f_ref.
    """
    fit = {}
    for mut, score in zip(df["mutations"], df["functional_score"]):
        fit[frozenset(parse_mutations(mut))] = score
    # Datasets often omit an explicit WT row; use the configured WT fitness as the
    # baseline so WT-anchored squares (the classic doubles) are included.
    fit.setdefault(frozenset(), wt_fitness)

    records = []
    for M, f_M in fit.items():
        if len(M) < 2:
            continue
        for a, b in itertools.combinations(sorted(M), 2):
            R, Ra, Rb = M - {a, b}, M - {b}, M - {a}
            if R not in fit or Ra not in fit or Rb not in fit:
                continue
            f_R, f_Ra, f_Rb = fit[R], fit[Ra], fit[Rb]
            eps = f_M - f_Ra - f_Rb + f_R

            da_alone, da_in_b = f_Ra - f_R, f_M - f_Rb   # effect of a on R vs on R+b
            db_alone, db_in_a = f_Rb - f_R, f_M - f_Ra   # effect of b on R vs on R+a
            sign_a = (np.sign(da_alone) != np.sign(da_in_b)) if (da_alone != 0 and da_in_b != 0) else False
            sign_b = (np.sign(db_alone) != np.sign(db_in_a)) if (db_alone != 0 and db_in_a != 0) else False

            if sign_a and sign_b:
                etype = "reciprocal_sign"
            elif sign_a or sign_b:
                etype = "sign"
            elif abs(eps) > eps_tol:
                etype = "magnitude"
            else:
                etype = "none"

            records.append(dict(ref_order=len(R), epsilon=eps, type=etype, f_top=f_M, f_ref=f_R))
    return pd.DataFrame(records)


def epistasis_type_fractions(eps_df):
    n = len(eps_df)
    if n == 0:
        return {t: float("nan") for t in ["magnitude", "sign", "reciprocal_sign"]}
    return {t: (eps_df["type"] == t).sum() / n for t in ["magnitude", "sign", "reciprocal_sign"]}


def face_fractions_by_ref_order(eps_df):
    """Class fractions and counts stratified by reference order |R| (variant order = |R| + 2)."""
    if len(eps_df) == 0:
        return pd.DataFrame()
    rows = []
    for ro, g in eps_df.groupby("ref_order"):
        n = len(g)
        rows.append(dict(
            ref_order=int(ro), variant_order=int(ro) + 2, n=n,
            frac_magnitude=(g["type"] == "magnitude").sum() / n,
            frac_sign=(g["type"] == "sign").sum() / n,
            frac_reciprocal_sign=(g["type"] == "reciprocal_sign").sum() / n,
            mean_abs_eps=g["epsilon"].abs().mean(),
        ))
    return pd.DataFrame(rows).sort_values("ref_order").reset_index(drop=True)


def per_order_stats(num_muts, y, y_pred_add, y_pred_global, label="full"):
    """Compute residual stats and R² per mutation order."""
    resid_add = y - y_pred_add
    resid_global = y - y_pred_global
    rows = []
    for k in sorted(np.unique(num_muts)):
        mask = num_muts == k
        n_k = mask.sum()
        if n_k < 2:
            continue
        ra_k, rg_k = resid_add[mask], resid_global[mask]
        rows.append(dict(
            subset=label, num_mutations=k, n=n_k,
            add_rmse=np.sqrt(np.mean(ra_k ** 2)), add_r2=r2_score(y[mask], y_pred_add[mask]),
            global_rmse=np.sqrt(np.mean(rg_k ** 2)), global_r2=r2_score(y[mask], y_pred_global[mask]),
        ))
    return pd.DataFrame(rows)


def _print_face_fractions(label, eps_df):
    if len(eps_df) == 0:
        print(f"  {label} (n=0): no complete squares")
        return
    f = epistasis_type_fractions(eps_df)
    print(f"  {label} (n={len(eps_df):,}):  mean|ε|={eps_df['epsilon'].abs().mean():.4f}   "
          f"magnitude {f['magnitude']:.1%}   sign {f['sign']:.1%}   reciprocal_sign {f['reciprocal_sign']:.1%}")


# ---------------------------------------------------------------------------
# Per-dataset analysis
# ---------------------------------------------------------------------------

def analyze(name, cfg):
    print(f"\n{'=' * 64}\n  {name}\n{'=' * 64}")
    df = pd.read_pickle(cfg["path"])
    y = df["functional_score"].values
    num_muts = df["num_mutations"].values
    X, all_muts, _ = build_mutation_matrix(df)
    print(f"Variants: {len(df):,}   Unique mutations: {len(all_muts):,}")
    print(f"Mutation orders: {dict(sorted(zip(*np.unique(num_muts, return_counts=True))))}")

    # ── Full landscape: additive & global epistasis ──────────────
    add_model, y_add, r2_add = fit_additive(X, y)
    _, y_global, r2_global, lin_model, iso_model = fit_global_epistasis(X, y)
    f_wt = add_model.intercept_
    print(f"\nConfigured WT fitness: {cfg['wt_fitness']:.4f}   (additive-model intercept estimate: {f_wt:.4f})")
    print(f"Additive R²:          {r2_add:.4f}")
    print(f"Global epistasis R²:  {r2_global:.4f}  (gain {r2_global - r2_add:+.4f})")

    order_full = per_order_stats(num_muts, y, y_add, y_global, label="full")
    print("\nPer-order breakdown (full landscape):")
    print(order_full[["num_mutations", "n", "add_r2", "add_rmse", "global_r2", "global_rmse"]].to_string(index=False))

    # ── Face epistasis over all measured squares ─────────────────
    eps_df = face_epistasis(df, cfg["wt_fitness"])
    fr = epistasis_type_fractions(eps_df)
    print(f"\nFace epistasis — all measured squares (local reference):")
    _print_face_fractions("overall", eps_df)
    by_ref = face_fractions_by_ref_order(eps_df)
    if len(by_ref) > 0:
        print("  by reference order |R|  (variant order = |R| + 2):")
        print("  " + by_ref.to_string(index=False).replace("\n", "\n  "))

    # ── Top fitness-peak regions ─────────────────────────────────
    summary = dict(name=name, n_variants=len(df), n_mutations=len(all_muts),
                   f_wt=round(float(f_wt), 4), r2_add=round(r2_add, 4),
                   r2_global=round(r2_global, 4), gain=round(r2_global - r2_add, 4),
                   face_n=len(eps_df), face_mean_abs=round(eps_df["epsilon"].abs().mean(), 4) if len(eps_df) else None,
                   face_sign=round(fr["sign"], 4), face_recip=round(fr["reciprocal_sign"], 4))

    for pct in TOP_PCTS:
        tag = f"top{int(pct * 100)}"
        threshold = np.percentile(y, 100 * (1 - pct))
        mask = y >= threshold
        print(f"\nTop {pct:.0%}  (n={mask.sum():,}, fitness >= {threshold:.3f})")

        # full models evaluated on the region
        r2_add_on = r2_score(y[mask], add_model.predict(X[mask]))
        r2_glob_on = r2_score(y[mask], iso_model.predict(lin_model.predict(X[mask])))
        print(f"  Full additive model on region:  R² = {r2_add_on:.4f}")
        print(f"  Full global   model on region:  R² = {r2_glob_on:.4f}")

        # refit on the region
        X_top, y_top = X[mask], y[mask]
        col_mask = X_top.std(axis=0) > 0
        if col_mask.sum() > 0 and len(y_top) > 10:
            _, _, r2_add_top = fit_additive(X_top[:, col_mask], y_top, regularize=True)
            _, _, r2_global_top, _, _ = fit_global_epistasis(X_top[:, col_mask], y_top)
            print(f"  Refit additive R² (Ridge):      {r2_add_top:.4f}")
            print(f"  Refit global epistasis R²:      {r2_global_top:.4f}  (gain {r2_global_top - r2_add_top:+.4f})")
        else:
            r2_add_top = r2_global_top = float("nan")

        # faces whose top corner sits in the region
        eps_top = eps_df[eps_df["f_top"] >= threshold] if len(eps_df) else eps_df
        if len(eps_top) > 0:
            print("  Face epistasis in region:")
            _print_face_fractions("region", eps_top)
            ft = epistasis_type_fractions(eps_top)
        else:
            ft = {"sign": float("nan"), "reciprocal_sign": float("nan")}

        summary[f"r2_add_{tag}"] = round(r2_add_top, 4) if not np.isnan(r2_add_top) else None
        summary[f"r2_global_{tag}"] = round(r2_global_top, 4) if not np.isnan(r2_global_top) else None
        summary[f"face_n_{tag}"] = len(eps_top)
        summary[f"face_sign_{tag}"] = round(ft["sign"], 4) if not np.isnan(ft["sign"]) else None
        summary[f"face_recip_{tag}"] = round(ft["reciprocal_sign"], 4) if not np.isnan(ft["reciprocal_sign"]) else None

    return summary


def print_summary(summaries):
    print("\n" + "=" * 80 + "\nSUMMARY (all datasets)\n" + "=" * 80)
    print(pd.DataFrame(summaries).set_index("name").T.to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    summaries = [analyze(name, cfg) for name, cfg in DATASETS.items()]
    print_summary(summaries)


if __name__ == "__main__":
    main()
