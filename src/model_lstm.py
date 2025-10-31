"""Model-building utilities for the LSTM forecaster."""

from __future__ import annotations

from typing import Any


def _require_tf_keras() -> Any:
    """Return the ``tf.keras`` module, raising a friendly error if unavailable."""

    try:
        from tensorflow import keras
    except ImportError as exc:  # pragma: no cover - dependency provided at runtime
        raise ImportError(
            "TensorFlow is required to build or load the LSTM model. "
            "Install it with `pip install tensorflow`."
        ) from exc

    return keras


def build_lstm(n_features: int, seq_len: int = 24, out_dim: int = 1):
    """Construct a simple stacked LSTM network."""

    keras = _require_tf_keras()
    layers = keras.layers
    models = keras.models
    optimizers = keras.optimizers

    inputs = layers.Input(shape=(seq_len, n_features))
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.LSTM(32)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(out_dim)(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(1e-3), loss="mae", metrics=["mape"])
    return model


def fit_model(model, X_train, y_train, X_valid, y_valid, epochs: int = 50, batch: int = 64):
    """Train the provided model with early stopping and learning rate scheduling."""

    keras = _require_tf_keras()
    callbacks = keras.callbacks

    early_stop = callbacks.EarlyStopping(patience=8, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(patience=4, factor=0.5)
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        epochs=epochs,
        batch_size=batch,
        callbacks=[early_stop, reduce_lr],
        verbose=2,
    )
    return history
