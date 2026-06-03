"""Build PH_Tweets_v7.csv: extends v6 with new BART-derived stance ratios,
VADER dispersion, and engagement-weighted aggregates.

Inputs:
  data/0_raw/{PE,PU+}-*.csv          per-tweet, with tweet_public_metrics
  data/1_interim/analyzed-{PE,PU+}-*.csv   per-tweet, with vader_scores + bart_stance
  data/PH_Tweets_v6.csv              existing monthly features (kept as-is)

Output:
  data/PH_Tweets_v7.csv              v6 + new columns, same long format
                                     (one row per (date, keyword))

New columns (per keyword):
  BART stance ratios (direct from argmax stance label):
    bart_good_share     % stance == 'philippine economy good'
    bart_bad_share      % stance == 'philippine economy bad'
    bart_net_stance     good_share - bad_share

  Soft BART probabilities (mean over all tweets, captures gradient signal):
    bart_pgood_mean     mean(P(good))
    bart_pbad_mean      mean(P(bad))

  Dispersion:
    vader_std           std-dev of VADER compound (polarization proxy)

  Engagement-weighted (weight = log(1 + likes + retweets + replies + quotes)):
    ew_vader_mean       weighted mean VADER compound
    ew_bart_net_stance  weighted net stance

For each new column we emit:
  raw + _stl (STL residual)
  Plus _log + _log_stl for count-like columns (currently none — vader_std is
  bounded in [0, 1] so log doesn't reshape it meaningfully; reserve this for
  future count-based aggregates like total monthly engagement).

Conventions match v6: STL applied on monthly index, period=12.
"""
import ast
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR = Path("/home/btiu/Documents/Research/TweetsNowcast/data")
RAW_DIR = DATA_DIR / "0_raw"
INTERIM_DIR = DATA_DIR / "1_interim"

RAW_FILES = {
    "PE":  RAW_DIR / "PE-2006-03-21T00:00:00Z-2023-01-01T00:00:00Z.csv",
    "PU+": RAW_DIR / "PU+-2006-03-21T00:00:00Z-2023-01-01T00:00:00Z.csv",
}
ANALYZED_FILES = {
    "PE":  INTERIM_DIR / "analyzed-PE-2006-03-21T00:00:00Z-2023-01-01T00:00:00Z.csv",
    "PU+": INTERIM_DIR / "analyzed-PU+-2006-03-21T00:00:00Z-2023-01-01T00:00:00Z.csv",
}

V6_PATH = DATA_DIR / "PH_Tweets_v6.csv"
V7_PATH = DATA_DIR / "PH_Tweets_v7.csv"

# Conservative window matching v6 (Jan 2009 → Dec 2022 by inspection).
START_MONTH = "2009-01-31"
END_MONTH   = "2022-12-31"

# Count-like columns get an additional _log + _log_stl emission.
# Reserved for future count-based aggregates (e.g., monthly engagement totals).
# vader_std is bounded in [0, 1] and roughly symmetric, so log doesn't help.
COUNT_LIKE_COLS: set[str] = set()


def parse_dict(s, key, default=np.nan):
    """Parse a stringified-dict column and pull one key. Returns default on any failure."""
    if not isinstance(s, str):
        return default
    try:
        d = ast.literal_eval(s)
        return d.get(key, default)
    except (ValueError, SyntaxError):
        return default


def parse_engagement(s):
    """Return total engagement count (likes + retweets + replies + quotes)."""
    if not isinstance(s, str):
        return 0
    try:
        d = ast.literal_eval(s)
        return int(d.get("like_count", 0)) + int(d.get("retweet_count", 0)) \
             + int(d.get("reply_count", 0)) + int(d.get("quote_count", 0))
    except (ValueError, SyntaxError):
        return 0


def load_keyword(keyword):
    """Join raw (engagement) and analyzed (vader, bart) per-tweet on tweet_id."""
    print(f"  loading raw {keyword} ...", flush=True)
    raw = pd.read_csv(RAW_FILES[keyword], usecols=["tweet_id", "tweet_public_metrics"])
    raw["engagement"] = raw["tweet_public_metrics"].apply(parse_engagement)
    raw = raw[["tweet_id", "engagement"]]

    print(f"  loading analyzed {keyword} ...", flush=True)
    ana = pd.read_csv(
        ANALYZED_FILES[keyword],
        usecols=["tweet_id", "created_at", "vader_scores", "bart_stance"],
        parse_dates=["created_at"],
    )
    ana["vader_compound"] = ana["vader_scores"].apply(lambda s: parse_dict(s, "compound", 0.0))
    ana["bart_stance_label"] = ana["bart_stance"].apply(lambda s: parse_dict(s, "stance", "none"))
    ana["bart_p_good"] = ana["bart_stance"].apply(lambda s: parse_dict(s, "philippine economy good", 0.0))
    ana["bart_p_bad"]  = ana["bart_stance"].apply(lambda s: parse_dict(s, "philippine economy bad", 0.0))
    ana = ana.drop(columns=["vader_scores", "bart_stance"])

    df = ana.merge(raw, on="tweet_id", how="left")
    df["engagement"] = df["engagement"].fillna(0).astype(int)
    df["log_engagement"] = np.log1p(df["engagement"])  # weight; 0-engagement tweet still counts as weight=0
    # Avoid all-zero weights collapsing the weighted mean; floor weight at log1p(0)=0 → bump to small epsilon.
    df["weight"] = df["log_engagement"].clip(lower=1e-6)
    df["month"] = df["created_at"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp(how="end").dt.normalize()
    return df


def monthly_features(df):
    """Aggregate per-tweet → monthly. Returns DataFrame indexed by month."""
    g = df.groupby("month")

    out = pd.DataFrame(index=g.size().index)

    # Direct stance ratios
    stance = df.groupby(["month", "bart_stance_label"]).size().unstack("bart_stance_label", fill_value=0)
    totals = stance.sum(axis=1).replace(0, np.nan)
    out["bart_good_share"] = stance.get("philippine economy good", 0) / totals
    out["bart_bad_share"]  = stance.get("philippine economy bad",  0) / totals
    out["bart_net_stance"] = out["bart_good_share"] - out["bart_bad_share"]

    # Soft probabilities
    out["bart_pgood_mean"] = g["bart_p_good"].mean()
    out["bart_pbad_mean"]  = g["bart_p_bad"].mean()

    # Dispersion
    out["vader_std"] = g["vader_compound"].std()

    # Engagement-weighted
    def wmean(sub, col):
        w = sub["weight"]
        return float((sub[col] * w).sum() / w.sum()) if w.sum() > 0 else np.nan

    ew_vader = g.apply(lambda sub: wmean(sub, "vader_compound"))
    out["ew_vader_mean"] = ew_vader

    # Engagement-weighted net stance: build per-tweet (+1 if good, -1 if bad, 0 else), weight, average.
    df = df.copy()
    df["stance_sign"] = 0.0
    df.loc[df["bart_stance_label"] == "philippine economy good", "stance_sign"] = 1.0
    df.loc[df["bart_stance_label"] == "philippine economy bad",  "stance_sign"] = -1.0
    out["ew_bart_net_stance"] = df.groupby("month").apply(lambda sub: wmean(sub, "stance_sign"))

    out.index = pd.DatetimeIndex(out.index, name="date")
    return out


def stl_residual(series, period=12):
    """STL residual on a monthly series. NaN-safe via linear interpolation before fit."""
    s = series.copy()
    if s.dropna().shape[0] < 2 * period:
        return pd.Series(np.nan, index=s.index)
    filled = s.interpolate(method="linear", limit_direction="both")
    try:
        stl = STL(filled, period=period, robust=True).fit()
        return filled - stl.trend - stl.seasonal  # residual
    except Exception as e:
        print(f"    STL failed ({type(e).__name__}: {e}); returning NaNs")
        return pd.Series(np.nan, index=s.index)


def add_transforms(df, count_like_cols):
    """For each base column, emit _stl. For count-like cols additionally emit _log and _log_stl."""
    out = df.copy()
    for col in df.columns:
        out[f"{col}_stl"] = stl_residual(df[col])
        if col in count_like_cols:
            logged = np.log(df[col].clip(lower=1e-9))
            out[f"{col}_log"] = logged
            out[f"{col}_log_stl"] = stl_residual(logged)
    return out


def main():
    print(f"Loading v6 from {V6_PATH} ...")
    v6 = pd.read_csv(V6_PATH, parse_dates=["date"])
    print(f"  v6 shape: {v6.shape}, keywords: {v6['keyword'].unique().tolist()}")

    new_long = []
    for keyword in ["PE", "PU+"]:
        print(f"\n=== {keyword} ===")
        tweets = load_keyword(keyword)
        print(f"  joined tweets: {len(tweets):,}")
        feats = monthly_features(tweets)
        # Restrict to the v6 window so we align on join
        feats = feats.loc[(feats.index >= START_MONTH) & (feats.index <= END_MONTH)]
        feats = add_transforms(feats, COUNT_LIKE_COLS)
        feats = feats.reset_index()
        feats["keyword"] = keyword
        new_long.append(feats)

    new_long = pd.concat(new_long, ignore_index=True)
    print(f"\nNew features: {new_long.shape[0]} rows × {new_long.shape[1] - 2} cols (excluding date+keyword)")
    print(f"New column list: {[c for c in new_long.columns if c not in ('date', 'keyword')]}")

    # Merge into v6 on (date, keyword). Use left join — keep v6 rows that don't get new features (shouldn't happen if window matches).
    v7 = v6.merge(new_long, on=["date", "keyword"], how="left", validate="one_to_one")
    print(f"\nv7 shape: {v7.shape}, added columns: {v7.shape[1] - v6.shape[1]}")
    print(f"NaN rate in new columns:")
    new_cols = [c for c in v7.columns if c not in v6.columns]
    print(v7[new_cols].isna().mean().round(3).to_string())

    v7.to_csv(V7_PATH, index=False)
    print(f"\nSaved {V7_PATH} ({v7.shape[0]} rows × {v7.shape[1]} cols)")


if __name__ == "__main__":
    main()
