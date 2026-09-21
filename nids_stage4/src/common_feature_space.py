"""
common_feature_space.py
========================
Stage 4 - Component 1: Cross-Dataset Feature Alignment

The core obstacle to Transfer Learning across CIC-IDS2017 and UNSW-NB15 is
that they don't share a column schema at all — CIC-IDS2017 has
`flow_duration`, `total_fwd_packets`, `flow_bytes_s`, ...; UNSW-NB15 has
`dur`, `spkts`, `sload`, ... A model trained on one dataset's 20 features
simply cannot be applied to the other's differently-named, differently-
scaled columns.

Real transfer learning across heterogeneous NIDS datasets requires mapping
BOTH datasets into a shared, semantically-aligned feature space first. This
module does that: it defines a small set of common flow-level concepts that
both datasets independently measure (duration, forward/backward packet and
byte counts, rates, inter-arrival timing) and provides a per-dataset
extractor that renames/derives each dataset's raw columns into that common
space.

This alignment step is itself a legitimate, reportable contribution — most
student NIDS projects skip it entirely and simply re-train from scratch per
dataset, which is NOT transfer learning, just independent training.

Common feature space (9 features)
----------------------------------
    duration            - flow/connection duration
    fwd_packets          - packets sent (source -> destination)
    bwd_packets          - packets received (destination -> source)
    fwd_bytes             - bytes sent
    bwd_bytes             - bytes received
    byte_rate             - bytes per second for the flow
    packet_rate           - packets per second for the flow
    fwd_iat_mean          - mean inter-arrival time, forward direction
    total_bytes            - fwd_bytes + bwd_bytes (derived, for a simple
                              volume signal both datasets support well)

Known limitations (be upfront about these in your report)
-----------------------------------------------------------
- UNSW-NB15's `sload` (bits/sec, source load) is used as a proxy for
  `byte_rate` — it's not an exact match to CIC-IDS2017's `flow_bytes_s`,
  but both represent "how much traffic volume per unit time" and are the
  closest semantically-aligned fields available.
- UNSW-NB15's `sinpkt` (inter-packet arrival time, source) is used as a
  proxy for CIC-IDS2017's `flow_iat_mean` for the same reason.
- This is a DELIBERATE, DOCUMENTED simplification, not silent data
  fabrication — every mapped field is a genuine measurement from the
  source dataset, just under a different name/unit convention. Flag this
  explicitly as a "future work: richer common feature engineering" note
  in your report; it's honest and expected for an undergrad project.
"""

import numpy as np
import pandas as pd


COMMON_FEATURES = [
    "duration", "fwd_packets", "bwd_packets", "fwd_bytes", "bwd_bytes",
    "byte_rate", "packet_rate", "fwd_iat_mean", "total_bytes",
]


def _clip_and_clean(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Replace inf with NaN, impute with column median — same policy as Stage 1."""
    df = df.copy()
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
    for c in cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    return df


def extract_common_features_cic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map CIC-IDS2017's raw (data_loader-cleaned) columns into the common
    feature space. Expects columns as produced by data_loader.load_cic_ids2017().
    """
    out = pd.DataFrame(index=df.index)
    out["duration"] = df["flow_duration"]
    out["fwd_packets"] = df["total_fwd_packets"]
    out["bwd_packets"] = df["total_backward_packets"]
    out["fwd_bytes"] = df["total_length_of_fwd_packets"]
    out["bwd_bytes"] = df["total_length_of_bwd_packets"]
    out["byte_rate"] = df["flow_bytes_s"]
    out["packet_rate"] = df["flow_packets_s"]
    out["fwd_iat_mean"] = df["flow_iat_mean"]
    out["total_bytes"] = out["fwd_bytes"] + out["bwd_bytes"]

    out = _clip_and_clean(out, COMMON_FEATURES)
    out["label"] = df["label"].values
    out["attack_type"] = df["attack_type"].values
    return out


def extract_common_features_unsw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map UNSW-NB15's raw (data_loader-cleaned) columns into the common
    feature space. Expects columns as produced by data_loader.load_unsw_nb15().

    Note the two documented proxy mappings (byte_rate <- sload,
    fwd_iat_mean <- sinpkt) explained in the module docstring above.
    """
    out = pd.DataFrame(index=df.index)
    out["duration"] = df["dur"]
    out["fwd_packets"] = df["spkts"]
    out["bwd_packets"] = df["dpkts"]
    out["fwd_bytes"] = df["sbytes"]
    out["bwd_bytes"] = df["dbytes"]
    out["byte_rate"] = df["sload"]          # proxy — see module docstring
    out["packet_rate"] = df["rate"]
    out["fwd_iat_mean"] = df["sinpkt"]       # proxy — see module docstring
    out["total_bytes"] = out["fwd_bytes"] + out["bwd_bytes"]

    out = _clip_and_clean(out, COMMON_FEATURES)
    out["label"] = df["label"].values
    out["attack_type"] = df["attack_type"].values
    return out


EXTRACTORS = {
    "cic_ids2017": extract_common_features_cic,
    "unsw_nb15": extract_common_features_unsw,
}


def extract_common_features(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Dispatch to the correct extractor. Returns common features + label + attack_type."""
    dataset_name = dataset_name.lower().strip()
    if dataset_name not in EXTRACTORS:
        raise ValueError(f"No common-feature extractor defined for '{dataset_name}'. "
                          f"Available: {list(EXTRACTORS.keys())}")
    return EXTRACTORS[dataset_name](df)
