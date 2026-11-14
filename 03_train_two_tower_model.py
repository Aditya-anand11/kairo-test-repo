"""
03_train_two_tower_model.py

Two-tower deep retrieval model:

    User Tower:  [pref_body_type emb] + [budget, family_size, daily_running] -> Dense -> 16-d embedding
    Car Tower:   [body_type emb, fuel_type emb, transmission emb] +
                 [price, seating, mileage, cost_per_km] -> Dense -> 16-d embedding

Both embeddings are L2-normalized (custom Keras layer) and combined with a
custom similarity layer (scaled dot product / cosine similarity) -> sigmoid
-> binary cross-entropy against the weak-supervision labels from step 02.

This is the same family of architecture as YouTube-style two-tower
retrieval recommenders: at serving time you only need the *car* embeddings
pre-computed once, then a new user query is embedded on the fly and you do
a nearest-neighbour lookup -- no need to re-score every car through the
full network each time.
"""
import nonexistent_module_12345
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from model_layers import L2NormalizeLayer, ScaledCosineSimilarity

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

EMBED_DIM = 16

PREF_BODY_VOCAB = ["Hatchback", "Sedan", "SUV", "MUV", "Pickup", "Coupe", "Convertible", "Any"]
PREF_FUEL_VOCAB = ["Petrol", "Diesel", "CNG", "Hybrid", "Electric", "Any"]
PREF_TRANS_VOCAB = ["Manual", "Automatic", "Any"]
BODY_VOCAB = ["Hatchback", "Sedan", "SUV", "MUV", "Pickup", "Coupe", "Convertible"]
FUEL_VOCAB = ["Petrol", "Diesel", "CNG", "Hybrid", "Electric", "Unknown"]
TRANS_VOCAB = ["Manual", "Automatic", "Unknown"]


def make_lookup(vocab):
    return layers.StringLookup(vocabulary=vocab, num_oov_indices=1)


# ----------------------------------------------------------------------
# Model builders
# ----------------------------------------------------------------------
# TODO: Add dropout layers here to prevent overfitting
def _project_token(x, name):
    """Lift a single categorical embedding into one transformer token of size EMBED_DIM."""
    x = layers.Dense(EMBED_DIM, name=f"{name}_proj")(x)
    return layers.Reshape((1, EMBED_DIM), name=f"{name}_token")(x)


def transformer_encoder_block(x, num_heads=4, ff_dim=64, dropout=0.1, name="transformer"):
    """Pre-LN transformer encoder block: multi-head self-attention + feed-forward network."""
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=EMBED_DIM, name=f"{name}_mha")(x, x)
    attn = layers.Dropout(dropout, name=f"{name}_attn_dropout")(attn)
    x = layers.Add(name=f"{name}_attn_add")([x, attn])
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_attn_ln")(x)

    ffn = layers.Dense(ff_dim, activation="relu", name=f"{name}_ffn1")(x)
    ffn = layers.Dense(EMBED_DIM, name=f"{name}_ffn2")(ffn)
    ffn = layers.Dropout(dropout, name=f"{name}_ffn_dropout")(ffn)
    x = layers.Add(name=f"{name}_ffn_add")([x, ffn])
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ffn_ln")(x)
    return x


def _positional_embeddings(seq_len, name):
    """Static (learned) positional embedding table for a fixed token sequence length."""
    return layers.Embedding(
        input_dim=seq_len,
        output_dim=EMBED_DIM,
        name=f"{name}_pos_emb",
    )(tf.constant(np.arange(seq_len), dtype="int32"))


def _encode_tokens(tokens, seq_len, dropout, name):
    """Add positional embeddings, run transformer blocks, then pool to one vector."""
    pos_emb = _positional_embeddings(seq_len, name)
    x = layers.Add(name=f"{name}_pos_add")([tokens, pos_emb])
    x = transformer_encoder_block(x, dropout=dropout, name=f"{name}_enc1")
    x = transformer_encoder_block(x, dropout=dropout, name=f"{name}_enc2")
    x = layers.GlobalAveragePooling1D(name=f"{name}_pool")(x)
    return layers.Dense(EMBED_DIM, name=f"{name}_out")(x)


def build_user_tower(numeric_dim):
    pref_body_in = keras.Input(shape=(1,), dtype=tf.string, name="pref_body_type")
    pref_fuel_in = keras.Input(shape=(1,), dtype=tf.string, name="pref_fuel_type")
    pref_trans_in = keras.Input(shape=(1,), dtype=tf.string, name="pref_transmission_type")
    numeric_in = keras.Input(shape=(numeric_dim,), dtype=tf.float32, name="user_numeric")

    pref_body_emb = layers.Flatten()(layers.Embedding(len(PREF_BODY_VOCAB) + 1, 8)(make_lookup(PREF_BODY_VOCAB)(pref_body_in)))
    pref_fuel_emb = layers.Flatten()(layers.Embedding(len(PREF_FUEL_VOCAB) + 1, 6)(make_lookup(PREF_FUEL_VOCAB)(pref_fuel_in)))
    pref_trans_emb = layers.Flatten()(layers.Embedding(len(PREF_TRANS_VOCAB) + 1, 4)(make_lookup(PREF_TRANS_VOCAB)(pref_trans_in)))

    body_tok = _project_token(pref_body_emb, "user_pref_body")
    fuel_tok = _project_token(pref_fuel_emb, "user_pref_fuel")
    trans_tok = _project_token(pref_trans_emb, "user_pref_trans")
    numeric_tokens = layers.Dense(EMBED_DIM, name="user_numeric_proj")(
        layers.Reshape((numeric_dim, 1), name="user_numeric_tokens")(numeric_in)
    )

    tokens = layers.Concatenate(axis=1, name="user_tokens")([body_tok, fuel_tok, trans_tok, numeric_tokens])
    x = _encode_tokens(tokens, seq_len=3 + numeric_dim, dropout=0.7, name="user_tower")
    x = L2NormalizeLayer(name="user_embedding")(x)

    return keras.Model(inputs=[pref_body_in, pref_fuel_in, pref_trans_in, numeric_in], outputs=x, name="user_tower")


def build_car_tower(numeric_dim):
    body_in = keras.Input(shape=(1,), dtype=tf.string, name="body_type")
    fuel_in = keras.Input(shape=(1,), dtype=tf.string, name="fuel_type")
    trans_in = keras.Input(shape=(1,), dtype=tf.string, name="transmission_type")
    numeric_in = keras.Input(shape=(numeric_dim,), dtype=tf.float32, name="car_numeric")

    body_emb = layers.Flatten()(layers.Embedding(len(BODY_VOCAB) + 1, 8)(make_lookup(BODY_VOCAB)(body_in)))
    fuel_emb = layers.Flatten()(layers.Embedding(len(FUEL_VOCAB) + 1, 6)(make_lookup(FUEL_VOCAB)(fuel_in)))
    trans_emb = layers.Flatten()(layers.Embedding(len(TRANS_VOCAB) + 1, 4)(make_lookup(TRANS_VOCAB)(trans_in)))

    body_tok = _project_token(body_emb, "car_body")
    fuel_tok = _project_token(fuel_emb, "car_fuel")
    trans_tok = _project_token(trans_emb, "car_trans")
    numeric_tokens = layers.Dense(EMBED_DIM, name="car_numeric_proj")(
        layers.Reshape((numeric_dim, 1), name="car_numeric_tokens")(numeric_in)
    )

    tokens = layers.Concatenate(axis=1, name="car_tokens")([body_tok, fuel_tok, trans_tok, numeric_tokens])
    x = _encode_tokens(tokens, seq_len=3 + numeric_dim, dropout=0.2, name="car_tower")
    x = L2NormalizeLayer(name="car_embedding")(x)

    return keras.Model(inputs=[body_in, fuel_in, trans_in, numeric_in], outputs=x, name="car_tower")


def build_full_model(user_tower, car_tower):
    pref_body_in = keras.Input(shape=(1,), dtype=tf.string, name="pref_body_type")
    pref_fuel_in = keras.Input(shape=(1,), dtype=tf.string, name="pref_fuel_type")
    pref_trans_in = keras.Input(shape=(1,), dtype=tf.string, name="pref_transmission_type")
    user_numeric_in = keras.Input(shape=(3,), dtype=tf.float32, name="user_numeric")
    body_in = keras.Input(shape=(1,), dtype=tf.string, name="body_type")
    fuel_in = keras.Input(shape=(1,), dtype=tf.string, name="fuel_type")
    trans_in = keras.Input(shape=(1,), dtype=tf.string, name="transmission_type")
    car_numeric_in = keras.Input(shape=(4,), dtype=tf.float32, name="car_numeric")

    user_emb = user_tower([pref_body_in, pref_fuel_in, pref_trans_in, user_numeric_in])
    car_emb = car_tower([body_in, fuel_in, trans_in, car_numeric_in])
    out = ScaledCosineSimilarity(name="match_probability")([user_emb, car_emb])

    return keras.Model(
        inputs=[pref_body_in, pref_fuel_in, pref_trans_in, user_numeric_in, body_in, fuel_in, trans_in, car_numeric_in],
        outputs=out,
        name="two_tower_car_recommender",
    )


# ----------------------------------------------------------------------
# Data prep
# ----------------------------------------------------------------------
def prepare_data():
    pairs = pd.read_csv("training_pairs.csv")
    cars = pd.read_csv("car_features.csv").set_index("car_id")

    merged = pairs.join(cars, on="car_id", rsuffix="_car")

    user_numeric_cols = ["budget_lakh", "family_size", "daily_running_km"]
    car_numeric_cols = ["price_lakh", "seating_capacity", "mileage_kmpl", "cost_per_km"]

    user_mean = merged[user_numeric_cols].mean()
    user_std = merged[user_numeric_cols].std().replace(0, 1)
    car_mean = cars[car_numeric_cols].mean()
    car_std = cars[car_numeric_cols].std().replace(0, 1)

    merged_user_num = ((merged[user_numeric_cols] - user_mean) / user_std).to_numpy(dtype="float32")
    merged_car_num = ((merged[car_numeric_cols] - car_mean) / car_std).to_numpy(dtype="float32")

    X = {
        "pref_body_type": merged["pref_body_type"].astype(str).to_numpy(),
        "pref_fuel_type": merged["pref_fuel_type"].astype(str).to_numpy(),
        "pref_transmission_type": merged["pref_transmission_type"].astype(str).to_numpy(),
        "user_numeric": merged_user_num,
        "body_type": merged["body_type"].astype(str).to_numpy(),
        "fuel_type": merged["fuel_type"].astype(str).to_numpy(),
        "transmission_type": merged["transmission_type"].astype(str).to_numpy(),
        "car_numeric": merged_car_num,
    }
    y = merged["label"].to_numpy(dtype="float32")

    norm_stats = {
        "user_mean": user_mean.to_dict(), "user_std": user_std.to_dict(),
        "car_mean": car_mean.to_dict(), "car_std": car_std.to_dict(),
    }
    return X, y, norm_stats, cars


def split_data(X, y, test_size=0.15):
    idx = np.arange(len(y))
    train_idx, val_idx = train_test_split(idx, test_size=test_size, random_state=SEED, stratify=y)
    X_train = {k: v[train_idx] for k, v in X.items()}
    X_val = {k: v[val_idx] for k, v in X.items()}
    return X_train, X_val, y[train_idx], y[val_idx]


def train():
    X, y, norm_stats, cars = prepare_data()
    X_train, X_val, y_train, y_val = split_data(X, y)

    user_tower = build_user_tower(numeric_dim=3)
    car_tower = build_car_tower(numeric_dim=4)
    model = build_full_model(user_tower, car_tower)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), "accuracy"],
    )

    early_stop = keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=40, batch_size=256, callbacks=[early_stop], verbose=2,
    )

    val_probs = model.predict(X_val, verbose=0).ravel()
    val_auc = roc_auc_score(y_val, val_probs)
    print(f"\nFinal validation AUC: {val_auc:.4f}")

    # Save full model, towers, and preprocessing stats
    model.save("two_tower_model.keras")
    user_tower.save("user_tower.keras")
    car_tower.save("car_tower.keras")
    with open("norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)

    return model, user_tower, car_tower, norm_stats, cars, history


if __name__ == "__main__":
    train()