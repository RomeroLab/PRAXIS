"""Top-1% epistasis fingerprint for the main figure.

For each protein, at the top-1% fitness peak region, report:
  - Additive R²   : R² of a purely additive (sum-of-effects) model, refit on the region (Ridge).
  - Monotone R²   : R² of a single monotone (isotonic) transform of the additive score, refit on
                    the region. Gap above Additive R² = nonspecific ("global") epistasis;
                    residual (1 - Monotone R²) = specific epistasis + noise.
  - Noise-floored face partition (% of peak-region faces): no_epistasis / magnitude / sign /
                    reciprocal_sign, using an order-aware r-trick noise floor (see below).

The full diagnostic version (per-order breakdown, whole-landscape fits, specific-epistasis/HOC,
raw partitions, top-5%) is preserved in epistasis_analysis_full.py.bak.
"""

import os, warnings, itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(rel):
    return os.path.join(ROOT, "insilico_analysis/dataset_oracle", rel)


# r_replicate = replicate reliability for the r-trick noise floor: sigma_single^2 = (1-r)*Var(singles).
# sigma_by_order = published per-order measurement noise where it differs from the single-mutant value
# (read-depth driven, not derivable from singles). Only GB1 has order-resolved noise published.
DATASETS = {
    "GFP": {
        "path": _p("gfp/gfp_SeqFxnDataset.pkl"), "wt_fitness": 3.7,
        # Sarkisyan WT-replicate noise sigma^2=0.0097 over Var(singles)=0.403 => r = 0.976.
        "r_replicate": 0.976,
    },
    "GB1": {
        "path": _p("gb1/gb1_SeqFxnDataset.pkl"), "wt_fitness": 0.0,
        # Olson 2014 triplicate single-mutant correlation R>0.996 => sigma_single ≈ 0.123.
        # GB1 faces' "both" corner is a DOUBLE mutant, noisier from lower read depth — a read-depth
        # effect (Var(doubles)=5.2 ≈ Var(singles)=3.8), NOT signal variance, so unrecoverable from
        # singles. Supply Otwinowski 2018's order-2 noise (0.52) directly => face floor ≈ 0.56.
        "r_replicate": 0.996, "sigma_by_order": {2: 0.52},
    },
    "PAB1": {
        "path": _p("pab1/pab1_SeqFxnDataset.pkl"), "wt_fitness": 0.0,
        # Melamed synonymous-variant control: noise variance ~0.25 over Var(singles)=2.68 => r = 0.907
        # (sigma_single ~0.5). Doubles noise not separately published => floor is a (safe) lower bound.
        "r_replicate": 0.907,
    },
    "UBE4": {
        "path": _p("ube4/ube4_SeqFxnDataset.pkl"), "wt_fitness": 0.084,
        # Enrich2 (Rubin 2017) E4B replicate correlation r=0.894 (r^2=0.80).
        "r_replicate": 0.894,
    },
}

TOP_PCT = 0.01      # peak region
NOISE_Z = 1.0       # floor at 1 face-sigma


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_mutations(mut_str):
    if pd.isna(mut_str) or mut_str in ("", "WT"):
        return []
    return mut_str.split(",")


def build_mutation_matrix(df):
    """One-hot encode individual mutations across all variants."""
    parsed = [parse_mutations(m) for m in df["mutations"]]
    all_muts = sorted({m for p in parsed for m in p})
    idx = {m: i for i, m in enumerate(all_muts)}
    X = np.zeros((len(df), len(all_muts)), dtype=np.float32)
    for i, p in enumerate(parsed):
        for m in p:
            X[i, idx[m]] = 1.0
    return X


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float("nan") if ss_tot == 0 else 1 - ss_res / ss_tot


def additive_r2(X, y):
    """R² of a Ridge additive model fit on (X, y)."""
    m = Ridge(alpha=1.0).fit(X, y)
    return r2_score(y, m.predict(X))


def monotone_r2(X, y):
    """R² of an additive score passed through a single monotone (isotonic) transform."""
    latent = LinearRegression().fit(X, y).predict(X)
    iso = IsotonicRegression(out_of_bounds="clip").fit(latent, y)
    return r2_score(y, iso.predict(latent))


def face_epistasis(df, wt_fitness):
    """Epistasis on every fully-measured 2D face (square). For a background R and mutations a, b:
        epsilon = f(R+a+b) - f(R+a) - f(R+b) + f(R)
    with marginal effects (da0/da1 for a, db0/db1 for b) for the noise-floored sign classification.
    ref_order = |R| (corner orders are |R|, |R|+1, |R|+1, |R|+2); f_top = f(R+a+b)."""
    fit = {frozenset(parse_mutations(m)): s for m, s in zip(df["mutations"], df["functional_score"])}
    fit.setdefault(frozenset(), wt_fitness)
    rec = []
    for M, f_M in fit.items():
        if len(M) < 2:
            continue
        for a, b in itertools.combinations(sorted(M), 2):
            R, Ra, Rb = M - {a, b}, M - {b}, M - {a}
            if R not in fit or Ra not in fit or Rb not in fit:
                continue
            f_R, f_Ra, f_Rb = fit[R], fit[Ra], fit[Rb]
            rec.append(dict(ref_order=len(R), f_top=f_M, epsilon=f_M - f_Ra - f_Rb + f_R,
                            da0=f_Ra - f_R, da1=f_M - f_Rb, db0=f_Rb - f_R, db1=f_M - f_Ra))
    return pd.DataFrame(rec)


def floor_partition(eps_df, tau):
    """4-way partition (none / magnitude / sign / reciprocal_sign) with a floor tau (scalar or
    per-face array) on BOTH the interaction term (|eps|<=tau -> none) and the marginal effects
    (a sign-flip counts only if both the alone- and in-background-effects exceed tau)."""
    n = len(eps_df)
    if n == 0:
        return None
    eps = eps_df["epsilon"].to_numpy()
    a0, a1 = eps_df["da0"].to_numpy(), eps_df["da1"].to_numpy()
    b0, b1 = eps_df["db0"].to_numpy(), eps_df["db1"].to_numpy()
    none = np.abs(eps) <= tau
    flip_a = (np.abs(a0) > tau) & (np.abs(a1) > tau) & (np.sign(a0) != np.sign(a1))
    flip_b = (np.abs(b0) > tau) & (np.abs(b1) > tau) & (np.sign(b0) != np.sign(b1))
    recip = (~none) & flip_a & flip_b
    sign = (~none) & (flip_a ^ flip_b)
    mag = (~none) & (~flip_a) & (~flip_b)
    return {"n": n, "none": int(none.sum()), "magnitude": int(mag.sum()),
            "sign": int(sign.sum()), "reciprocal_sign": int(recip.sum())}


# ---------------------------------------------------------------------------
# Per-dataset fingerprint
# ---------------------------------------------------------------------------

def fingerprint(name, cfg):
    df = pd.read_pickle(cfg["path"])
    y = df["functional_score"].values
    num_muts = df["num_mutations"].values
    X = build_mutation_matrix(df)

    # Order-aware per-face noise floor (uniform method):
    #   sigma_eps^2(face) = sigma^2[|R|] + 2*sigma^2[|R|+1] + sigma^2[|R|+2]
    # sigma_single from the r-trick; higher orders from cfg["sigma_by_order"] else sigma_single.
    eps_df = face_epistasis(df, cfg["wt_fitness"])
    var_singles = float(np.var(y[num_muts == 1]))
    sigma_single = float(np.sqrt(max(0.0, (1.0 - cfg["r_replicate"]) * var_singles)))
    sbo = {int(k): float(v) for k, v in cfg.get("sigma_by_order", {}).items()}
    sig = lambda k: sbo.get(int(k), sigma_single)
    face_sigma = lambda o: float(np.sqrt(sig(o) ** 2 + 2 * sig(o + 1) ** 2 + sig(o + 2) ** 2))
    eps_df["tau"] = NOISE_Z * np.array([face_sigma(int(o)) for o in eps_df["ref_order"].to_numpy()])

    # Top-1% peak region: refit additive & monotone R² on the region; noise-floor its faces.
    threshold = np.percentile(y, 100 * (1 - TOP_PCT))
    X_top, y_top = X[y >= threshold], y[y >= threshold]
    col = X_top.std(axis=0) > 0
    add = additive_r2(X_top[:, col], y_top)
    mono = monotone_r2(X_top[:, col], y_top)

    eps_top = eps_df[eps_df["f_top"] >= threshold]
    p = floor_partition(eps_top, eps_top["tau"].to_numpy())
    n = p["n"]
    return dict(
        protein=name, n_faces=n,
        add_R2=round(add, 3), mono_R2=round(mono, 3), gain=round(mono - add, 3),
        no_epi=round(100 * p["none"] / n, 1), magnitude=round(100 * p["magnitude"] / n, 1),
        sign=round(100 * p["sign"] / n, 1), reciprocal=round(100 * p["reciprocal_sign"] / n, 1),
    )


def main():
    rows = [fingerprint(name, cfg) for name, cfg in DATASETS.items()]
    print("Top-1% epistasis fingerprint (peak region)\n")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
