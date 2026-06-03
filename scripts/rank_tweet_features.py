"""Rank tweet features (v6 already-shipped + v7 new) by marginal predictive power.

Two-stage funnel:
  Stage 1 (intra-feature pruning):
    Pairwise correlation matrix of all candidate columns (v6 features × 2 keywords +
    new v7 features × 2 keywords) over the 2009-2022 monthly panel. Greedy drop any
    column whose |corr| with an already-kept column ≥ CORR_THRESHOLD (default 0.9).
    Walk order: new v7 columns first, then v6 — this biases the kept set toward the
    newer columns when there's a tie, surfacing the rare features that survive.

  Stage 2 (target alignment via Ridge proxy):
    Forward-walk 24-vintage Ridge nowcast using DFM-derived target (Actual_Q at
    each vintage). For each surviving feature:
       baseline:  target_t ~ target_{t-3M}        (AR proxy)
       candidate: target_t ~ target_{t-3M} + f_t
    Score = RMSE(candidate) - RMSE(baseline). Negative = improvement.
    Lower is better.

Output:
  notebooks/analysis_outputs/tweet_feature_ranking.csv
  Columns: feature, kept_after_pruning, baseline_rmse, candidate_rmse,
           rmse_delta, abs_corr_with_target.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = Path("/home/btiu/Documents/Research/TweetsNowcast/data")
OUTPUT_DIR = Path("/home/btiu/Documents/Research/TweetsNowcast/notebooks/analysis_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

V7_PATH = DATA_DIR / "PH_Tweets_v7.csv"
DFM_PATH = Path("/home/btiu/Documents/Research/TweetsNowcast/Results/DFM_Opt_1010_W1000_PHL_GDP_SA_E_L1-0_L2-0_summary.csv")

CORR_THRESHOLD = 0.9

# 24-vintage test window matching the LSTM experiments.
TEST_START = "2019-07-31"
TEST_END = "2021-06-30"

# v6 already-shipped features. Used to flag (origin=v6/v7) but NOT to exclude.
V6_FEATURE_NAMES = {
    "TBraw", "VADERraw", "TBstance", "VADERstance",
    "TBstanceweight", "VADERstanceweight", "CR_lognorm",
    "TBraw_log", "VADERraw_log", "TBstance_log", "VADERstance_log",
    "TBstanceweight_log", "VADERstanceweight_log", "CR_lognorm_log",
    "TBraw_log_stl", "VADERraw_log_stl", "TBstance_log_stl", "VADERstance_log_stl",
    "TBstanceweight_log_stl", "VADERstanceweight_log_stl", "CR_lognorm_log_stl",
    "TBraw_stl", "VADERraw_stl", "TBstance_stl", "VADERstance_stl",
    "TBstanceweight_stl", "VADERstanceweight_stl", "CR_lognorm_stl",
}


def load_features_wide():
    """Load v7 (which contains v6 + new cols) and pivot. Returns (wide DataFrame, origin map).
    `origin` is a dict feature_full_name → 'v6' or 'v7'. Walk order puts v7 first."""
    df = pd.read_csv(V7_PATH, parse_dates=["date"])
    all_feats = [c for c in df.columns if c not in {"date", "keyword"}]
    new_feats = [c for c in all_feats if c not in V6_FEATURE_NAMES]
    v6_feats = [c for c in all_feats if c in V6_FEATURE_NAMES]
    ordered = new_feats + v6_feats  # v7 first so they get priority in pruning

    long = df[["date", "keyword"] + ordered]
    wide = long.pivot(index="date", columns="keyword", values=ordered)
    wide.columns = [f"{k}__{f}" for f, k in wide.columns]
    wide = wide.sort_index()

    # Re-order columns to match the v7-first walk order (pivot may reorder).
    walk_order = [f"{k}__{f}" for f in ordered for k in ["PE", "PU+"] if f"{k}__{f}" in wide.columns]
    wide = wide[walk_order]

    origin = {col: ("v6" if col.split("__", 1)[1] in V6_FEATURE_NAMES else "v7")
              for col in wide.columns}
    return wide, origin


def load_target():
    """Load DFM Actual_Q per vintage. Returns Series indexed by vintage month."""
    df = pd.read_csv(DFM_PATH, parse_dates=["date"])
    return df.set_index("date")["Actual_Q"].sort_index()


def correlation_prune(features_wide, threshold=CORR_THRESHOLD):
    """Greedy correlation pruning. Walk features in their natural order; keep one
    if it doesn't correlate ≥ threshold with any already-kept feature."""
    corr = features_wide.corr().abs()
    kept, dropped_for = [], {}
    for col in features_wide.columns:
        if all(corr.loc[col, k] < threshold for k in kept):
            kept.append(col)
        else:
            # Note which kept feature absorbed this one (highest correlation).
            absorber = max((k for k in kept if corr.loc[col, k] >= threshold),
                           key=lambda k: corr.loc[col, k])
            dropped_for[col] = (absorber, float(corr.loc[col, absorber]))
    return kept, dropped_for, corr


def forward_walk_rmse(target, feature_series, vintages, lag_quarters=1):
    """Forward-walk Ridge regression nowcast over `vintages`.

    For each vintage v: train Ridge on rows < v with X = [target_{v-3M}, feature_v]
    and y = target_v; predict for v; collect squared error. Returns RMSE.
    If feature_series is None, runs AR-only baseline.
    """
    lag_months = 3 * lag_quarters
    target_lag = target.shift(lag_months // 1)  # shift by months in DatetimeIndex
    # Actually target is vintage-indexed monthly; shift by lag_months month-positions.
    target_lag = target.shift(lag_months, freq="MS").reindex(target.index, method="nearest", tolerance=pd.Timedelta("15D"))
    # Simpler: rebuild a monthly DatetimeIndex from target then shift positionally.
    target_lag = pd.Series(target.values, index=target.index).shift(lag_months // 1 * 1)  # placeholder
    # Implement positional shift on the sorted vintage series (each row is a month-end).
    s = target.sort_index()
    target_lag = s.shift(lag_months)  # positional shift over monthly index → lag_quarters quarters back

    rows = []
    for v in vintages:
        if v not in target.index:
            continue
        y_t = target.loc[v]
        x_lag = target_lag.loc[v] if v in target_lag.index else np.nan
        if pd.isna(y_t) or pd.isna(x_lag):
            continue
        row = {"vintage": v, "y": y_t, "target_lag": x_lag}
        if feature_series is not None:
            f_v = feature_series.get(v, np.nan)
            row["feat"] = f_v
        rows.append(row)
    df_all = pd.DataFrame(rows).dropna()

    if df_all.empty:
        return np.nan

    feat_cols = ["target_lag"] + (["feat"] if feature_series is not None else [])
    errors = []
    for v in vintages:
        train = df_all[df_all["vintage"] < v]
        test = df_all[df_all["vintage"] == v]
        if len(train) < 4 or test.empty:
            continue
        # Scale features for Ridge stability.
        scaler = StandardScaler().fit(train[feat_cols])
        Xtr = scaler.transform(train[feat_cols])
        Xte = scaler.transform(test[feat_cols])
        model = Ridge(alpha=1.0).fit(Xtr, train["y"])
        y_pred = model.predict(Xte)
        errors.append((y_pred[0] - test["y"].iloc[0]) ** 2)
    return float(np.sqrt(np.mean(errors))) if errors else np.nan


def main():
    print(f"Loading {V7_PATH} ...")
    features_wide, origin = load_features_wide()
    n_v6 = sum(1 for o in origin.values() if o == "v6")
    n_v7 = sum(1 for o in origin.values() if o == "v7")
    print(f"  monthly panel: {features_wide.shape[0]} months × {features_wide.shape[1]} columns "
          f"({n_v7} v7 + {n_v6} v6)")

    print(f"\nLoading target from {DFM_PATH} ...")
    target = load_target()
    print(f"  target series: {len(target)} vintages, {target.index.min().date()} → {target.index.max().date()}")

    print(f"\n=== Stage 1: correlation pruning at |r| ≥ {CORR_THRESHOLD} ===")
    kept, dropped_for, _ = correlation_prune(features_wide, CORR_THRESHOLD)
    print(f"  kept: {len(kept)} of {features_wide.shape[1]}")
    print("  dropped (mapped to kept absorber):")
    for col, (absorber, r) in dropped_for.items():
        print(f"    {col:<48s} → {absorber} (|r|={r:.3f})")

    # Vintages for the forward-walk test (24 LSTM-matched).
    test_vintages = pd.date_range(TEST_START, TEST_END, freq="ME")
    test_vintages = [v for v in test_vintages if v in target.index]
    print(f"\n=== Stage 2: forward-walk Ridge over {len(test_vintages)} test vintages ===")

    baseline_rmse = forward_walk_rmse(target, None, test_vintages)
    print(f"  baseline RMSE (AR lag-1Q only):       {baseline_rmse:.4f}")

    rows = []
    for col in kept:
        feat_series = features_wide[col]
        cand_rmse = forward_walk_rmse(target, feat_series, test_vintages)
        delta = cand_rmse - baseline_rmse if pd.notna(cand_rmse) else np.nan
        # Univariate signal: abs correlation of feature with target over all overlapping vintages.
        overlap = pd.concat([feat_series, target], axis=1).dropna()
        abs_corr = abs(overlap.corr().iloc[0, 1]) if len(overlap) > 5 else np.nan
        rows.append({
            "feature": col, "origin": origin[col], "kept": True,
            "baseline_rmse": baseline_rmse, "candidate_rmse": cand_rmse,
            "rmse_delta": delta, "abs_corr_target": abs_corr,
        })

    # Also record the pruned features so the ranking table is complete.
    for col, (absorber, r) in dropped_for.items():
        rows.append({
            "feature": col, "origin": origin[col], "kept": False,
            "baseline_rmse": baseline_rmse, "candidate_rmse": np.nan,
            "rmse_delta": np.nan, "abs_corr_target": np.nan,
            "pruned_for": absorber, "pruned_corr": r,
        })

    out = pd.DataFrame(rows)
    out_sorted = out.sort_values(["kept", "rmse_delta"], ascending=[False, True])
    out_path = OUTPUT_DIR / "tweet_feature_ranking.csv"
    out_sorted.to_csv(out_path, index=False)

    print(f"\n=== Ranking (lower rmse_delta = better; ⬇ improvement vs baseline) ===")
    keep_only = out_sorted[out_sorted["kept"]].copy()
    keep_only["rmse_delta_pretty"] = keep_only["rmse_delta"].map(lambda x: f"{x:+.4f}" if pd.notna(x) else "n/a")
    print(keep_only[["feature", "origin", "candidate_rmse", "rmse_delta_pretty", "abs_corr_target"]].round(4).to_string(index=False))
    print(f"\nFull table → {out_path}")


if __name__ == "__main__":
    main()
