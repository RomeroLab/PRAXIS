#!/usr/bin/env python
"""
Epistasis analysis for in-silico benchmark datasets (GFP, GB1, PAB1, UBE4).

Quantifies local (pairwise) and global epistasis across ALL mutation orders:
  - The full landscape
  - The top 5% fitness peak region

Key analyses:
  1. Additive R² and global epistasis R² (full landscape + top 5%)
  2. Additive-model residuals stratified by mutation count (all orders)
  3. R² stratified by mutation count
  4. Pairwise epistasis classification (sign / reciprocal sign) for doubles

Outputs:
  - insilico_analysis/figures/epistasis_summary.png       (scatter + global epistasis)
  - insilico_analysis/figures/epistasis_by_order.png      (residuals & R² by mutation count)
  - insilico_analysis/figures/epistasis_pairwise.png      (pairwise ε distributions)
  - insilico_analysis/figures/epistasis_table.csv         (all metrics)
  - insilico_analysis/figures/epistasis_by_order_table.csv (per-order breakdown)
"""

import os, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.isotonic import IsotonicRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS = {
    "GFP": {
        "path": os.path.join(ROOT, "insilico_analysis/dataset_oracle/gfp/gfp_SeqFxnDataset.pkl"),
        "wt": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
    },
    "GB1": {
        "path": os.path.join(ROOT, "insilico_analysis/dataset_oracle/gb1/gb1_SeqFxnDataset.pkl"),
        "wt": "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
    },
    "PAB1": {
        "path": os.path.join(ROOT, "insilico_analysis/dataset_oracle/pab1/pab1_SeqFxnDataset.pkl"),
        "wt": "GNIFIKNLHPDIDNKALYDTFSVFGDILSSKIATDENGKSKGFGFVHFEEEGAAKEAIDALNGMLLNGQEIYVAP",
    },
    "UBE4": {
        "path": os.path.join(ROOT, "insilico_analysis/dataset_oracle/ube4/ube4_SeqFxnDataset.pkl"),
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


def pairwise_epistasis(df, f_wt):
    """Compute ε = f(AB) - f(A) - f(B) + f(WT) for all doubles with both singles."""
    singles = df[df["num_mutations"] == 1]
    single_map = {}
    for _, row in singles.iterrows():
        ms = parse_mutations(row["mutations"])
        if len(ms) == 1:
            single_map[ms[0]] = row["functional_score"]

    doubles = df[df["num_mutations"] == 2]
    records = []
    for _, row in doubles.iterrows():
        ms = parse_mutations(row["mutations"])
        if len(ms) != 2:
            continue
        m1, m2 = ms
        if m1 not in single_map or m2 not in single_map:
            continue

        f_ab = row["functional_score"]
        f_a, f_b = single_map[m1], single_map[m2]
        eps = f_ab - f_a - f_b + f_wt

        da_alone = f_a - f_wt
        db_alone = f_b - f_wt
        da_in_b = f_ab - f_b
        db_in_a = f_ab - f_a

        sign_a = (np.sign(da_alone) != np.sign(da_in_b)) if (da_alone != 0 and da_in_b != 0) else False
        sign_b = (np.sign(db_alone) != np.sign(db_in_a)) if (db_alone != 0 and db_in_a != 0) else False

        if sign_a and sign_b:
            etype = "reciprocal_sign"
        elif sign_a or sign_b:
            etype = "sign"
        elif abs(eps) > 1e-8:
            etype = "magnitude"
        else:
            etype = "none"

        records.append(dict(
            mut1=m1, mut2=m2,
            f_ab=f_ab, f_a=f_a, f_b=f_b, f_wt=f_wt,
            epsilon=eps, type=etype,
        ))
    return pd.DataFrame(records)


def epistasis_type_fractions(eps_df):
    n = len(eps_df)
    if n == 0:
        return {t: float("nan") for t in ["magnitude", "sign", "reciprocal_sign"]}
    return {t: (eps_df["type"] == t).sum() / n for t in ["magnitude", "sign", "reciprocal_sign"]}


# ---------------------------------------------------------------------------
# Per-order analysis: residuals and R² stratified by mutation count
# ---------------------------------------------------------------------------

def per_order_stats(num_muts, y, y_pred_add, y_pred_global, label="full"):
    """Compute residual stats and R² per mutation order."""
    resid_add = y - y_pred_add
    resid_global = y - y_pred_global
    orders = sorted(np.unique(num_muts))

    rows = []
    for k in orders:
        mask = num_muts == k
        n_k = mask.sum()
        if n_k < 2:
            continue
        y_k = y[mask]
        ra_k = resid_add[mask]
        rg_k = resid_global[mask]
        rows.append(dict(
            subset=label,
            num_mutations=k,
            n=n_k,
            # Additive residuals
            add_mae=np.abs(ra_k).mean(),
            add_rmse=np.sqrt(np.mean(ra_k ** 2)),
            add_mean_resid=ra_k.mean(),
            add_r2=r2_score(y_k, y_pred_add[mask]),
            # Global epistasis residuals
            global_mae=np.abs(rg_k).mean(),
            global_rmse=np.sqrt(np.mean(rg_k ** 2)),
            global_r2=r2_score(y_k, y_pred_global[mask]),
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-dataset analysis
# ---------------------------------------------------------------------------

def analyze(name, cfg):
    print(f"\n{'=' * 60}\n  {name}\n{'=' * 60}")
    df = pd.read_pickle(cfg["path"])
    y = df["functional_score"].values
    num_muts = df["num_mutations"].values
    X, all_muts, mut_to_idx = build_mutation_matrix(df)
    print(f"Variants: {len(df):,}   Unique mutations: {len(all_muts):,}")
    print(f"Mutation orders: {dict(sorted(zip(*np.unique(num_muts, return_counts=True))))}")

    # ── Full landscape ────────────────────────────────────────────
    add_model, y_add, r2_add = fit_additive(X, y)
    latent, y_global, r2_global, lin_model, iso_model = fit_global_epistasis(X, y)
    f_wt = add_model.intercept_
    print(f"Estimated WT fitness (intercept): {f_wt:.4f}")
    print(f"Additive R²:          {r2_add:.4f}")
    print(f"Global epistasis R²:  {r2_global:.4f}  (gain {r2_global - r2_add:+.4f})")

    # Per-order breakdown (full)
    order_full = per_order_stats(num_muts, y, y_add, y_global, label="full")
    print(f"\nPer-order breakdown (full landscape):")
    print(order_full[["num_mutations", "n", "add_r2", "add_rmse", "global_r2", "global_rmse"]].to_string(index=False))

    # Pairwise epistasis
    eps_df = pairwise_epistasis(df, f_wt)
    pw = {}
    if len(eps_df) > 0:
        pw["n"] = len(eps_df)
        pw["mean_abs"] = eps_df["epsilon"].abs().mean()
        pw["median_abs"] = eps_df["epsilon"].abs().median()
        pw["std"] = eps_df["epsilon"].std()
        pw.update(epistasis_type_fractions(eps_df))
        print(f"\nPairwise ε  (n={pw['n']:,}):")
        print(f"  mean|ε|={pw['mean_abs']:.4f}  median|ε|={pw['median_abs']:.4f}  std={pw['std']:.4f}")
        print(f"  magnitude {pw['magnitude']:.1%}   sign {pw['sign']:.1%}   reciprocal_sign {pw['reciprocal_sign']:.1%}")

    # ── Top subsets ──────────────────────────────────────────────
    tops = {}  # keyed by pct
    for pct in TOP_PCTS:
        pct_label = f"top{int(pct*100)}pct"
        threshold = np.percentile(y, 100 * (1 - pct))
        mask = y >= threshold
        X_top, y_top = X[mask], y[mask]
        num_muts_top = num_muts[mask]
        print(f"\nTop {pct:.0%}  (n={mask.sum():,}, fitness >= {threshold:.3f})")

        # Evaluate FULL models on top subset
        r2_add_on_top = r2_score(y_top, add_model.predict(X_top))
        y_global_on_top = iso_model.predict(lin_model.predict(X_top))
        r2_global_on_top = r2_score(y_top, y_global_on_top)
        print(f"  Full additive model on top:  R² = {r2_add_on_top:.4f}")
        print(f"  Full global model on top:    R² = {r2_global_on_top:.4f}")

        # Refit on top subset
        col_mask = X_top.std(axis=0) > 0
        X_top_filt = X_top[:, col_mask]
        if X_top_filt.shape[1] > 0 and len(y_top) > 10:
            _, y_add_top, r2_add_top = fit_additive(X_top_filt, y_top, regularize=True)
            _, y_global_top, r2_global_top, _, _ = fit_global_epistasis(X_top_filt, y_top)
            print(f"  Refit additive R² (Ridge):       {r2_add_top:.4f}")
            print(f"  Refit global epistasis R²:       {r2_global_top:.4f}  (gain {r2_global_top - r2_add_top:+.4f})")

            order_top = per_order_stats(num_muts_top, y_top, y_add_top, y_global_top, label=pct_label)
            if len(order_top) > 0:
                print(f"\n  Per-order breakdown (top {pct:.0%}):")
                print("  " + order_top[["num_mutations", "n", "add_r2", "add_rmse", "global_r2", "global_rmse"]].to_string(index=False).replace("\n", "\n  "))
        else:
            r2_add_top = r2_global_top = float("nan")
            order_top = pd.DataFrame()

        # Pairwise ε restricted to this top subset
        eps_top = eps_df[eps_df["f_ab"] >= threshold] if len(eps_df) > 0 else pd.DataFrame()
        pw_top = {}
        if len(eps_top) > 0:
            pw_top["n"] = len(eps_top)
            pw_top["mean_abs"] = eps_top["epsilon"].abs().mean()
            pw_top["median_abs"] = eps_top["epsilon"].abs().median()
            pw_top["std"] = eps_top["epsilon"].std()
            pw_top.update(epistasis_type_fractions(eps_top))
            print(f"  Pairwise ε in top {pct:.0%} (n={pw_top['n']:,}):")
            print(f"    mean|ε|={pw_top['mean_abs']:.4f}  median|ε|={pw_top['median_abs']:.4f}")
            print(f"    magnitude {pw_top['magnitude']:.1%}   sign {pw_top['sign']:.1%}   reciprocal_sign {pw_top['reciprocal_sign']:.1%}")

        tops[pct] = dict(
            threshold=threshold,
            r2_add_on_top=r2_add_on_top,
            r2_global_on_top=r2_global_on_top,
            r2_add_top=r2_add_top,
            r2_global_top=r2_global_top,
            gain_top=r2_global_top - r2_add_top if not np.isnan(r2_global_top) else float("nan"),
            pw_top=pw_top, eps_top=eps_top,
            order_top=order_top,
        )

    return dict(
        name=name,
        n_variants=len(df), n_mutations=len(all_muts),
        r2_add=r2_add, r2_global=r2_global,
        gain=r2_global - r2_add,
        f_wt=f_wt,
        pw=pw, eps_df=eps_df,
        tops=tops,
        latent=latent, y=y, y_add=y_add, y_global=y_global,
        num_muts=num_muts,
        order_full=order_full,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_scatter_and_global(results, out_path):
    """Figure 1: additive-vs-observed scatter & global-epistasis nonlinear mapping."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for j, name in enumerate(results):
        r = results[name]
        y = r["y"]
        latent = r["latent"]
        y_global = r["y_global"]
        # Use the smallest top pct for highlighting
        threshold = r["tops"][min(TOP_PCTS)]["threshold"]

        ax = axes[0, j]
        top_mask = y >= threshold
        ax.scatter(latent[~top_mask], y[~top_mask], s=1, alpha=0.15, c="steelblue", rasterized=True)
        ax.scatter(latent[top_mask], y[top_mask], s=3, alpha=0.4, c="crimson", label=f"top {min(TOP_PCTS):.0%}", rasterized=True)
        mn, mx = latent.min(), latent.max()
        ax.plot([mn, mx], [mn, mx], "k--", lw=0.8, label="y=x")
        ax.set_xlabel("Additive prediction")
        ax.set_ylabel("Observed fitness")
        ax.set_title(f"{name}  (add R²={r['r2_add']:.3f})")
        ax.legend(fontsize=7, loc="upper left")

        ax = axes[1, j]
        order = np.argsort(latent)
        ax.scatter(latent, y, s=1, alpha=0.1, c="grey", rasterized=True)
        ax.plot(latent[order], y_global[order], c="darkorange", lw=2, label="isotonic g(·)")
        ax.plot([mn, mx], [mn, mx], "k--", lw=0.8)
        ax.set_xlabel("Latent additive score")
        ax.set_ylabel("Observed fitness")
        ax.set_title(f"Global epistasis  (R²={r['r2_global']:.3f})")
        ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"\nSaved {out_path}")


def plot_by_order(results, out_path):
    """Figure 2: additive RMSE and R² stratified by mutation count."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    colors_full = "steelblue"
    top_colors = {0.05: "crimson", 0.01: "darkorange"}

    for j, name in enumerate(results):
        r = results[name]
        of = r["order_full"]

        # Row 0: RMSE by mutation count
        ax = axes[0, j]
        ax.plot(of["num_mutations"], of["add_rmse"], "o-", c=colors_full, label="Additive (full)", ms=5)
        ax.plot(of["num_mutations"], of["global_rmse"], "s--", c=colors_full, alpha=0.6, label="Global (full)", ms=4)
        for pct in TOP_PCTS:
            ot = r["tops"][pct]["order_top"]
            if len(ot) > 0:
                c = top_colors.get(pct, "green")
                ax.plot(ot["num_mutations"], ot["add_rmse"], "o-", c=c, label=f"Additive (top {pct:.0%})", ms=5)
                ax.plot(ot["num_mutations"], ot["global_rmse"], "s--", c=c, alpha=0.6, label=f"Global (top {pct:.0%})", ms=4)
        ax.set_xlabel("Number of mutations")
        ax.set_ylabel("RMSE (residual)")
        ax.set_title(f"{name} — Prediction error by order")
        ax.legend(fontsize=5)
        ax.set_xticks(of["num_mutations"].values)

        # Row 1: R² by mutation count (clipped to [-1, 1] for readability)
        ax = axes[1, j]
        ax.plot(of["num_mutations"], of["add_r2"].clip(-1, 1), "o-", c=colors_full, label="Additive (full)", ms=5)
        ax.plot(of["num_mutations"], of["global_r2"].clip(-1, 1), "s--", c=colors_full, alpha=0.6, label="Global (full)", ms=4)
        for pct in TOP_PCTS:
            ot = r["tops"][pct]["order_top"]
            if len(ot) > 0:
                c = top_colors.get(pct, "green")
                ax.plot(ot["num_mutations"], ot["add_r2"].clip(-1, 1), "o-", c=c, label=f"Additive (top {pct:.0%})", ms=5)
                ax.plot(ot["num_mutations"], ot["global_r2"].clip(-1, 1), "s--", c=c, alpha=0.6, label=f"Global (top {pct:.0%})", ms=4)
        ax.set_xlabel("Number of mutations")
        ax.set_ylabel("R² (clipped to [-1, 1])")
        ax.set_title(f"{name} — R² by order")
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0, color="grey", ls=":", lw=0.5)
        ax.legend(fontsize=5)
        ax.set_xticks(of["num_mutations"].values)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_pairwise(results, out_path):
    """Figure 3: pairwise ε distributions (full landscape + top subsets)."""
    n_rows = 1 + len(TOP_PCTS)
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 4.5 * n_rows))

    for j, name in enumerate(results):
        r = results[name]

        plot_data = [("Full landscape", r["eps_df"])]
        for pct in TOP_PCTS:
            plot_data.append((f"Top {pct:.0%}", r["tops"][pct]["eps_top"]))

        for row, (label, eps_df) in enumerate(plot_data):
            ax = axes[row, j]
            if len(eps_df) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{name} — {label}")
                continue

            eps = eps_df["epsilon"].values
            lo, hi = np.percentile(eps, [1, 99])
            bins = np.linspace(lo, hi, 60)
            ax.hist(eps, bins=bins, color="steelblue", edgecolor="white", lw=0.3)
            ax.axvline(0, color="k", ls="--", lw=0.8)
            ax.axvline(eps.mean(), color="crimson", ls="-", lw=1.2, label=f"mean={eps.mean():.3f}")

            fracs = epistasis_type_fractions(eps_df)
            info = (f"n={len(eps_df):,}\n"
                    f"|ε| mean={np.abs(eps).mean():.3f}\n"
                    f"sign={fracs['sign']:.1%}\n"
                    f"recip={fracs['reciprocal_sign']:.1%}")
            ax.text(0.97, 0.95, info, transform=ax.transAxes,
                    va="top", ha="right", fontsize=7,
                    bbox=dict(boxstyle="round", fc="white", alpha=0.8))
            ax.set_xlabel("ε (epistasis)")
            ax.set_ylabel("Count")
            ax.set_title(f"{name} — {label}")
            ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def save_tables(results, fig_dir):
    """Save summary CSV and per-order CSV."""
    # -- Summary table --
    rows = []
    for name in results:
        r = results[name]
        pw = r["pw"]
        row = dict(
            dataset=name,
            n_variants=r["n_variants"],
            n_mutations=r["n_mutations"],
            est_wt_fitness=round(r["f_wt"], 4),
            r2_additive=round(r["r2_add"], 4),
            r2_global_epistasis=round(r["r2_global"], 4),
            nonlinearity_gain=round(r["gain"], 4),
            pairwise_n=pw.get("n", 0),
            pairwise_mean_abs_eps=round(pw["mean_abs"], 4) if "mean_abs" in pw else None,
            pairwise_frac_sign=round(pw.get("sign", float("nan")), 4),
            pairwise_frac_reciprocal=round(pw.get("reciprocal_sign", float("nan")), 4),
        )
        for pct in TOP_PCTS:
            tag = f"{int(pct*100)}"
            t = r["tops"][pct]
            pt = t["pw_top"]
            row[f"r2_add_refit_top{tag}"] = round(t["r2_add_top"], 4) if not np.isnan(t["r2_add_top"]) else None
            row[f"r2_global_refit_top{tag}"] = round(t["r2_global_top"], 4) if not np.isnan(t["r2_global_top"]) else None
            row[f"nonlinearity_gain_top{tag}"] = round(t["gain_top"], 4) if not np.isnan(t["gain_top"]) else None
            row[f"pairwise_n_top{tag}"] = pt.get("n", 0)
            row[f"pairwise_mean_abs_eps_top{tag}"] = round(pt["mean_abs"], 4) if "mean_abs" in pt else None
            row[f"pairwise_frac_sign_top{tag}"] = round(pt.get("sign", float("nan")), 4)
            row[f"pairwise_frac_reciprocal_top{tag}"] = round(pt.get("reciprocal_sign", float("nan")), 4)
        rows.append(row)
    df_summary = pd.DataFrame(rows)
    p1 = os.path.join(fig_dir, "epistasis_table.csv")
    df_summary.to_csv(p1, index=False)
    print(f"\nSaved {p1}")
    print("\n" + "=" * 80 + "\nSUMMARY TABLE\n" + "=" * 80)
    print(df_summary.T.to_string())

    # -- Per-order table --
    order_frames = []
    for name in results:
        r = results[name]
        odf = r["order_full"]
        if len(odf) > 0:
            tmp = odf.copy()
            tmp.insert(0, "dataset", name)
            order_frames.append(tmp)
        for pct in TOP_PCTS:
            ot = r["tops"][pct]["order_top"]
            if len(ot) > 0:
                tmp = ot.copy()
                tmp.insert(0, "dataset", name)
                order_frames.append(tmp)
    if order_frames:
        df_order = pd.concat(order_frames, ignore_index=True)
        p2 = os.path.join(fig_dir, "epistasis_by_order_table.csv")
        df_order.to_csv(p2, index=False)
        print(f"\nSaved {p2}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    fig_dir = os.path.join(ROOT, "insilico_analysis", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    results = {}
    for name, cfg in DATASETS.items():
        results[name] = analyze(name, cfg)

    plot_scatter_and_global(results, os.path.join(fig_dir, "epistasis_summary.png"))
    plot_by_order(results, os.path.join(fig_dir, "epistasis_by_order.png"))
    plot_pairwise(results, os.path.join(fig_dir, "epistasis_pairwise.png"))
    save_tables(results, fig_dir)


if __name__ == "__main__":
    main()
