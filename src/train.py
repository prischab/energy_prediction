import os, json, joblib
from .data_prep import load_raw, resample_hourly, build_features
from .rfe_select import run_rfe
from .seq_window import to_sequences, to_sequences_multi
from .model_lstm import build_lstm, fit_model


def main(horizon_key="1h"):
if not os.path.exists(DATA_PATH):
raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")


print("🚀 Loading data…")
df = build_features(resample_hourly(load_raw(DATA_PATH)))
y = df[TARGET]
X = df.drop(columns=[TARGET])


print("🔎 RFE selecting features…")
selected, ranking = run_rfe(X, y, n_features=N_FEATURES_RFE, step=5)
X = X[selected]


split = int(0.8 * len(X))
Xtr, Xte = X.iloc[:split], X.iloc[split:]
ytr, yte = y.iloc[:split], y.iloc[split:]


sx, sy = MinMaxScaler(), MinMaxScaler()
Xtr_s = sx.fit_transform(Xtr)
Xte_s = sx.transform(Xte)


horizon = HORIZONS[horizon_key]
if horizon == 1:
ytr_s = sy.fit_transform(ytr.values.reshape(-1,1)).ravel()
yte_s = sy.transform(yte.values.reshape(-1,1)).ravel()
Xtr_seq, ytr_seq = to_sequences(pd.DataFrame(Xtr_s, index=Xtr.index, columns=X.columns),
pd.Series(ytr_s, index=ytr.index), seq_len=SEQ_LEN, horizon=1)
Xte_seq, yte_seq = to_sequences(pd.DataFrame(Xte_s, index=Xte.index, columns=X.columns),
pd.Series(yte_s, index=yte.index), seq_len=SEQ_LEN, horizon=1)
out_dim = 1
else:
ytr_s = sy.fit_transform(ytr.values.reshape(-1,1)).ravel()
yte_s = sy.transform(yte.values.reshape(-1,1)).ravel()
Xtr_seq, ytr_seq = to_sequences_multi(pd.DataFrame(Xtr_s, index=Xtr.index, columns=X.columns),
pd.Series(ytr_s, index=ytr.index), seq_len=SEQ_LEN, horizon=horizon)
Xte_seq, yte_seq = to_sequences_multi(pd.DataFrame(Xte_s, index=Xte.index, columns=X.columns),
pd.Series(yte_s, index=yte.index), seq_len=SEQ_LEN, horizon=horizon)
out_dim = horizon


model = build_lstm(n_features=X.shape[1], seq_len=SEQ_LEN, out_dim=out_dim)
vcut = int(0.9 * len(Xtr_seq))
print("🏋️ Training LSTM…")
_ = fit_model(model, Xtr_seq[:vcut], ytr_seq[:vcut], Xtr_seq[vcut:], ytr_seq[vcut:], epochs=30, batch=64)


os.makedirs(MODEL_DIR, exist_ok=True)
model.save(os.path.join(MODEL_DIR, 'lstm_model.keras'))
joblib.dump(sx, os.path.join(MODEL_DIR, 'scaler_X.pkl'))
joblib.dump(sy, os.path.join(MODEL_DIR, 'scaler_y.pkl'))
with open(os.path.join(MODEL_DIR, 'meta.json'), 'w') as f:
json.dump({
'seq_len': SEQ_LEN,
'feature_order': X.columns.tolist(),
'horizon': horizon,
'target': TARGET,
'selected_rfe': selected,
'ranking': ranking
}, f, indent=2)
print("💾 Saved artifacts to ./models")


if __name__ == "__main__":
main("1h")