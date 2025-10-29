import os, json, joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .config import DATA_PATH, MODEL_DIR
from .data_prep import load_raw, resample_hourly, build_features
import tensorflow as tf


def eval_last_test_window():
model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'lstm_model.keras'))
sx = joblib.load(os.path.join(MODEL_DIR, 'scaler_X.pkl'))
sy = joblib.load(os.path.join(MODEL_DIR, 'scaler_y.pkl'))
meta = json.load(open(os.path.join(MODEL_DIR, 'meta.json')))


df = build_features(resample_hourly(load_raw(DATA_PATH)))
y = df[meta['target']]
X = df[meta['feature_order']]


split = int(0.8 * len(X))
Xte, yte = X.iloc[split:], y.iloc[split:]
Xte_s = sx.transform(Xte)
yte_s = sy.transform(yte.values.reshape(-1,1)).ravel()


from .seq_window import to_sequences
X_seq, y_seq = to_sequences(pd.DataFrame(Xte_s, index=Xte.index, columns=X.columns),
pd.Series(yte_s, index=yte.index), seq_len=meta['seq_len'], horizon=meta['horizon'])


preds = model.predict(X_seq, verbose=0)
preds_inv = sy.inverse_transform(preds)
y_inv = sy.inverse_transform(y_seq.reshape(-1,1))


mae = mean_absolute_error(y_inv.flatten(), preds_inv.flatten())
rmse = mean_squared_error(y_inv.flatten(), preds_inv.flatten(), squared=False)
return float(mae), float(rmse)


if __name__ == "__main__":
print(eval_last_test_window())