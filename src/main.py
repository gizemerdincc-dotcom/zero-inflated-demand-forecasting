# zinn_full_pipeline.py
# Clean main pipeline for Zero-Inflated Neural Network (ZINN)

import os
import random
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models, optimizers, losses, backend as K
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ======================================================
# ================= CONFIG =============================
# ======================================================
FILE_PATH = r"C:\deneme\NNtrial1.xlsx"   # <-- kendi dosya yolunu kontrol et
OUTDIR = r"C:\Users\gizem\PyCharmMiscProject\zinn_outputs"

SEQ_LEN = 14
MIN_SERIES_LEN = 20
TEST_SIZE = 0.2

BATCH_SIZE = 32
EPOCHS = 80
LR = 1e-3
ALPHA_MSE = 1.0
EMBED_DIM = 8

DATE_COL = "Date"
CUSTOMER_COL = "Customer"
PRODUCT_COL = "Product"
TARGET_COL = "EDI"


# ======================================================
# =============== ATTENTION LAYER ======================
# ======================================================
class AttentionLayer(layers.Layer):
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )

    def call(self, inputs):
        e = K.squeeze(K.tanh(K.dot(inputs, self.W)), axis=-1)
        a = K.softmax(e)
        a = K.expand_dims(a, axis=-1)
        return K.sum(inputs * a, axis=1)


# ======================================================
# =============== MASKED LOSS ==========================
# ======================================================
def masked_mse(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, 0.0), K.floatx())
    eps = K.epsilon()
    se = K.square(y_true - y_pred) * mask
    return K.sum(se) / (K.sum(mask) + eps)


# ======================================================
# =============== DATA PREPARATION =====================
# ======================================================
def load_and_prepare_raw_data(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path)

    # Column normalization if needed
    rename_map = {}
    if "CustomerID" in df.columns:
        rename_map["CustomerID"] = CUSTOMER_COL
    if "ProductID" in df.columns:
        rename_map["ProductID"] = PRODUCT_COL
    if "DateColumn" in df.columns:
        rename_map["DateColumn"] = DATE_COL

    if rename_map:
        df = df.rename(columns=rename_map)

    required_cols = [CUSTOMER_COL, PRODUCT_COL, DATE_COL, TARGET_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[CUSTOMER_COL] = df[CUSTOMER_COL].astype(str)
    df[PRODUCT_COL] = df[PRODUCT_COL].astype(str)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")

    df = df.dropna(subset=[DATE_COL, TARGET_COL, CUSTOMER_COL, PRODUCT_COL]).copy()
    df = df.sort_values([CUSTOMER_COL, PRODUCT_COL, DATE_COL]).reset_index(drop=True)

    return df


def prepare_customer_product_data(df: pd.DataFrame, customer_id: str, product_id: str) -> pd.DataFrame | None:
    df_cp = df[(df[CUSTOMER_COL] == customer_id) & (df[PRODUCT_COL] == product_id)].copy()
    if df_cp.empty:
        return None

    df_cp = df_cp.sort_values(DATE_COL).copy()

    # Lagged demand (safe)
    df_cp["Lag1"] = df_cp[TARGET_COL].shift(1)
    df_cp["Lag2"] = df_cp[TARGET_COL].shift(2)

    # Date-based features
    df_cp["DayOfWeek"] = df_cp[DATE_COL].dt.dayofweek.astype(int)
    df_cp["WeekOfYear"] = df_cp[DATE_COL].dt.isocalendar().week.astype(int)
    df_cp["Month"] = df_cp[DATE_COL].dt.month.astype(int)

    # Target
    df_cp["Target"] = df_cp[TARGET_COL].astype(float)

    # Drop rows with NaN caused by lagging
    df_cp = df_cp.dropna().reset_index(drop=True)

    return df_cp if not df_cp.empty else None


def build_feature_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    all_features = []
    processed = 0
    skipped = 0

    # Only observed customer-product pairs
    observed_pairs = (
        df_raw[[CUSTOMER_COL, PRODUCT_COL]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    for customer_id, product_id in observed_pairs:
        df_feat = prepare_customer_product_data(df_raw, customer_id, product_id)
        if df_feat is not None and not df_feat.empty:
            all_features.append(df_feat)
            processed += 1
        else:
            skipped += 1

    if not all_features:
        raise ValueError("No valid feature dataframe could be created.")

    final_df = pd.concat(all_features, ignore_index=True)
    print(f"Feature creation finished. Processed series: {processed}, Skipped series: {skipped}")
    return final_df


# ======================================================
# =============== SAMPLE CREATION ======================
# ======================================================
def build_samples(final_df: pd.DataFrame, seq_len: int, min_series_len: int) -> pd.DataFrame:
    static_cols = ["Lag1", "Lag2", "DayOfWeek", "WeekOfYear", "Month"]

    samples = []

    grouped = final_df.groupby([CUSTOMER_COL, PRODUCT_COL], sort=False)

    for (cust, prod), g in grouped:
        g = g.sort_values(DATE_COL).reset_index(drop=True)

        if len(g) < min_series_len:
            continue

        values = g["Target"].values.astype(float)

        for i in range(len(g)):
            seq = values[max(0, i - seq_len):i]
            if len(seq) < seq_len:
                seq = np.pad(seq, (seq_len - len(seq), 0), constant_values=0.0)

            static_vals = g.loc[i, static_cols].values.astype(float)

            samples.append({
                "cust": str(cust),
                "prod": str(prod),
                "seq": seq,
                "static": static_vals,
                "y": float(values[i]),
                "y_bin": int(values[i] != 0),
                "date": g.loc[i, DATE_COL]
            })

    if not samples:
        raise ValueError("No samples could be built. Check MIN_SERIES_LEN and data coverage.")

    return pd.DataFrame(samples)


# ======================================================
# =============== MAIN EXPERIMENT ======================
# ======================================================
def run_experiment(seed: int = 42):
    # ---------- Reproducibility ----------
    os.makedirs(OUTDIR, exist_ok=True)

    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print(f"\nRunning ZINN | SEED = {seed}")

    # ---------- Load raw data ----------
    df_raw = load_and_prepare_raw_data(FILE_PATH)

    # ---------- Feature dataframe ----------
    final_df = build_feature_dataframe(df_raw)

    # ---------- Build samples ----------
    df_s = build_samples(final_df, seq_len=SEQ_LEN, min_series_len=MIN_SERIES_LEN)

    # ---------- Encode categoricals ----------
    le_c = LabelEncoder()
    le_p = LabelEncoder()

    df_s["cust_i"] = le_c.fit_transform(df_s["cust"])
    df_s["prod_i"] = le_p.fit_transform(df_s["prod"])

    # ---------- Train / Test split (time-based within each series) ----------
    train_idx, test_idx = [], []

    for _, g in df_s.groupby(["cust", "prod"], sort=False):
        g = g.sort_values("date")
        cut = int(len(g) * (1 - TEST_SIZE))

        # Safety: avoid empty test or train
        if cut <= 0 or cut >= len(g):
            continue

        train_idx.extend(g.index[:cut].tolist())
        test_idx.extend(g.index[cut:].tolist())

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Train/test split failed. Check TEST_SIZE and sample counts.")

    # ---------- Prepare arrays ----------
    X_seq_all = np.stack(df_s["seq"].values).astype(np.float32)
    X_stat_all = np.stack(df_s["static"].values).astype(np.float32)
    y_all = df_s["y"].values.astype(np.float32)
    y_bin_all = df_s["y_bin"].values.astype(np.float32)

    cust_all = df_s["cust_i"].values.astype(np.int32)
    prod_all = df_s["prod_i"].values.astype(np.int32)

    # ---------- Normalize static features ----------
    scaler_stat = StandardScaler()
    X_stat_train = scaler_stat.fit_transform(X_stat_all[train_idx])
    X_stat_test = scaler_stat.transform(X_stat_all[test_idx])

    # ---------- Normalize sequence inputs (FIXED) ----------
    seq_mean = X_seq_all[train_idx].mean()
    seq_std = X_seq_all[train_idx].std() + 1e-6

    X_seq_train = (X_seq_all[train_idx] - seq_mean) / seq_std
    X_seq_test = (X_seq_all[test_idx] - seq_mean) / seq_std

    # ---------- Labels ----------
    y_train = y_all[train_idx]
    y_test = y_all[test_idx]

    y_bin_train = y_bin_all[train_idx]
    y_bin_test = y_bin_all[test_idx]

    cust_train = cust_all[train_idx]
    cust_test = cust_all[test_idx]

    prod_train = prod_all[train_idx]
    prod_test = prod_all[test_idx]

    # ---------- Model ----------
    in_c = layers.Input(shape=(), dtype="int32", name="cust_input")
    in_p = layers.Input(shape=(), dtype="int32", name="prod_input")
    in_s = layers.Input(shape=(X_stat_train.shape[1],), name="static_input")
    in_q = layers.Input(shape=(SEQ_LEN,), name="seq_input")

    emb_c = layers.Flatten()(layers.Embedding(len(le_c.classes_), EMBED_DIM)(in_c))
    emb_p = layers.Flatten()(layers.Embedding(len(le_p.classes_), EMBED_DIM)(in_p))

    xq = layers.Reshape((SEQ_LEN, 1))(in_q)
    xq = layers.GRU(64, return_sequences=True)(xq)
    xq = AttentionLayer()(xq)

    xs = layers.Dense(64, activation="relu")(in_s)

    x = layers.Concatenate()([emb_c, emb_p, xq, xs])
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)

    out_bin = layers.Dense(1, activation="sigmoid", name="p_nonzero")(x)
    out_amt = layers.Dense(1, activation="relu", name="amount")(x)

    model = models.Model(
        inputs=[in_c, in_p, in_s, in_q],
        outputs=[out_bin, out_amt]
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR),
        loss={
            "p_nonzero": losses.BinaryCrossentropy(),
            "amount": masked_mse
        },
        loss_weights={
            "p_nonzero": 1.0,
            "amount": ALPHA_MSE
        }
    )

    # ---------- Train ----------
    X_train = [cust_train, prod_train, X_stat_train, X_seq_train]
    Y_train = [y_bin_train, y_train]

    X_test = [cust_test, prod_test, X_stat_test, X_seq_test]

    model.fit(
        X_train,
        Y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    # ---------- Predict ----------
    p_bin, p_amt = model.predict(X_test, verbose=0)
    pred = p_bin.flatten() * p_amt.flatten()

    # ---------- Metrics ----------
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    zero_mask = (y_test == 0)
    nonzero_mask = (y_test > 0)

    mae_zero = (
        mean_absolute_error(y_test[zero_mask], pred[zero_mask])
        if zero_mask.sum() > 0 else np.nan
    )
    mae_nonzero = (
        mean_absolute_error(y_test[nonzero_mask], pred[nonzero_mask])
        if nonzero_mask.sum() > 0 else np.nan
    )

    print(f"Seed {seed}")
    print(f"Train size      : {len(train_idx)}")
    print(f"Test size       : {len(test_idx)}")
    print(f"MAE             : {mae:.4f}")
    print(f"RMSE            : {rmse:.4f}")
    print(f"MAE_zero        : {mae_zero:.4f}")
    print(f"MAE_nonzero     : {mae_nonzero:.4f}")

    return {
        "seed": seed,
        "mae": mae,
        "rmse": rmse,
        "mae_zero": mae_zero,
        "mae_nonzero": mae_nonzero,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
    }


# ======================================================
# ================= ENTRY POINT ========================
# ======================================================
if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    results = run_experiment(seed)
    print("\nFinal Results:")
    for k, v in results.items():
        print(f"{k}: {v}")
