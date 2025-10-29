import pandas as pd
from .config import RESAMPLE_RULE, TARGET, MAX_LAG, ROLL_WINDOWS


def load_raw(path: str) -> pd.DataFrame:
df = pd.read_csv(path, sep=';', na_values='?', low_memory=False)
# Combine Date + Time (dayfirst=True for UCI format)
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
df = df.drop(columns=['Date', 'Time']).sort_values('datetime').set_index('datetime')
df = df.apply(pd.to_numeric, errors='coerce').dropna()
return df


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
return df.resample(RESAMPLE_RULE).mean().dropna()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
df = df.copy()
for L in range(1, MAX_LAG+1):
df[f'{TARGET}_lag_{L}'] = df[TARGET].shift(L)
for w in ROLL_WINDOWS:
df[f'{TARGET}_roll_mean_{w}'] = df[TARGET].rolling(w).mean()
df[f'{TARGET}_roll_std_{w}'] = df[TARGET].rolling(w).std()
df['hour'] = df.index.hour
df['weekday'] = df.index.weekday
df['month'] = df.index.month
return df.dropna()