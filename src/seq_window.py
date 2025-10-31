"""Window utilities for turning tabular data into supervised sequences."""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_sequences(X, y, seq_len: int = 24, horizon: int = 1):
    """Create sequences for single-step forecasting."""
    X_values = X.values if hasattr(X, "values") else X
    y_values = y.values if hasattr(y, "values") else y

    xs, ys = [], []
    for idx in range(seq_len, len(X_values) - horizon + 1):
        xs.append(X_values[idx - seq_len : idx, :])
        ys.append(y_values[idx + horizon - 1])
    return np.array(xs), np.array(ys)


def to_sequences_multi(X, y, seq_len: int = 24, horizon: int = 24):
    """Create sequences for multi-step forecasting."""
    X_values = X.values if hasattr(X, "values") else X
    y_values = y.values if hasattr(y, "values") else y

    xs, ys = [], []
    for idx in range(seq_len, len(X_values) - horizon + 1):
        xs.append(X_values[idx - seq_len : idx, :])
        ys.append(y_values[idx : idx + horizon])
    return np.array(xs), np.array(ys)
