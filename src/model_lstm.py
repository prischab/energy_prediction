from tensorflow.keras import layers, models, callbacks, optimizers


def build_lstm(n_features, seq_len=24, out_dim=1):
inp = layers.Input(shape=(seq_len, n_features))
x = layers.LSTM(64, return_sequences=True)(inp)
x = layers.LSTM(32)(x)
x = layers.Dense(32, activation='relu')(x)
out = layers.Dense(out_dim)(x)
model = models.Model(inp, out)
model.compile(optimizer=optimizers.Adam(1e-3), loss='mae', metrics=['mape'])
return model


def fit_model(model, Xtr, ytr, Xva, yva, epochs=50, batch=64):
es = callbacks.EarlyStopping(patience=8, restore_best_weights=True)
rl = callbacks.ReduceLROnPlateau(patience=4, factor=0.5)
hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs,
batch_size=batch, callbacks=[es, rl], verbose=2)
return hist