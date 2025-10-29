import numpy as np
import pandas as pd


def to_sequences(X, y, seq_len=24, horizon=1):
Xv = X.values if hasattr(X, 'values') else X
yv = y.values if hasattr(y, 'values') else y
xs, ys = [], []
for i in range(seq_len, len(Xv)-horizon+1):
xs.append(Xv[i-seq_len:i, :])
ys.append(yv[i + horizon - 1])
return np.array(xs), np.array(ys)


def to_sequences_multi(X, y, seq_len=24, horizon=24):
Xv = X.values if hasattr(X, 'values') else X
yv = y.values if hasattr(y, 'values') else y
xs, ys = [], []
for i in range(seq_len, len(Xv)-horizon+1):
xs.append(Xv[i-seq_len:i, :])
ys.append(yv[i:i+horizon])
return np.array(xs), np.array(ys)