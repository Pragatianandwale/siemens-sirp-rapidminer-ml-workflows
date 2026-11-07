from pathlib import Path

script = r'''"""
SmartFall LSTM v3
Participant-wise split, class-presence checks, training, and evaluation.
Dataset expected at ../data/smartfall_rnn_clean_ready_to_use.csv
All outputs use smartfall_lstm_v3_ prefix to preserve earlier files.
"""

from pathlib import Path
import json
import random
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_FILENAME = "smartfall_rnn_clean_ready_to_use.csv"
PREFIX = "smartfall_lstm_v3_"
SEQUENCE_LENGTH = 5
FEATURES = [
    "gait_speed",
    "step_time_variability",
    "turn_duration",
    "sway_rms",
    "accel_rms",
]
TARGET = "scenario_label"
GROUP = "person_id"
DAY = "day"

candidates = [
    PROJECT_DIR / "data" / DATA_FILENAME,
    SCRIPT_DIR / "data" / DATA_FILENAME,
    PROJECT_DIR / DATA_FILENAME,
    SCRIPT_DIR / DATA_FILENAME,
]
DATA_PATH = next((p for p in candidates if p.exists()), None)
if DATA_PATH is None:
    raise FileNotFoundError(
        "Dataset not found. Checked these paths:\n" +
        "\n".join(str(p) for p in candidates)
    )

print("Dataset:", DATA_PATH)
print("Output folder:", SCRIPT_DIR)

df = pd.read_csv(DATA_PATH)
required = FEATURES + [TARGET, GROUP, DAY]
missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(
        f"Missing required columns: {missing}\n"
        f"Available columns: {list(df.columns)}"
    )

df = df.dropna(subset=required).copy()
df[DAY] = pd.to_numeric(df[DAY], errors="coerce")
for col in FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=[DAY] + FEATURES).copy()
df[DAY] = df[DAY].astype(float)
df[TARGET] = df[TARGET].astype(str)
df[GROUP] = df[GROUP].astype(str)

if df[GROUP].nunique() < 5:
    raise ValueError(
        "At least five distinct participants are recommended for "
        "participant-wise train/validation/test splitting."
    )

encoder = LabelEncoder()
df[TARGET] = encoder.fit_transform(df[TARGET])
class_names = [str(c) for c in encoder.classes_]
n_classes = len(class_names)
if n_classes < 2:
    raise ValueError("At least two scenario_label classes are required.")

print("Rows after cleaning:", len(df))
print("Participants:", df[GROUP].nunique())
print("Classes:", class_names)
print("Class row counts:")
print(df[TARGET].value_counts().sort_index().rename(index=dict(enumerate(class_names))))

# Create rolling sequences separately for each participant.
X_list, y_list, group_list, day_list = [], [], [], []
for person, part in df.groupby(GROUP):
    part = part.sort_values(DAY)
    values = part[FEATURES].to_numpy(dtype=np.float32)
    labels = part[TARGET].to_numpy(dtype=np.int64)
    days = part[DAY].to_numpy(dtype=float)

    for end in range(SEQUENCE_LENGTH - 1, len(part)):
        start = end - SEQUENCE_LENGTH + 1
        window_days = days[start:end + 1]
        # Skip gaps only when day values are integer-like.
        if np.allclose(window_days, np.round(window_days)):
            if not np.all(np.diff(window_days) == 1):
                continue
        X_list.append(values[start:end + 1])
        y_list.append(labels[end])
        group_list.append(person)
        day_list.append(days[end])

if not X_list:
    raise ValueError(
        "No sequences created. Check that each participant has enough "
        "rows and that day values are consecutive."
    )

X = np.asarray(X_list, dtype=np.float32)
y = np.asarray(y_list, dtype=np.int64)
groups = np.asarray(group_list)
end_days = np.asarray(day_list)
all_classes = np.arange(n_classes)

print(f"Sequences: {len(X)}")
print("Sequence counts by class:")
print(pd.Series(y).value_counts().sort_index().rename(index=dict(enumerate(class_names))))

# Search for participant-disjoint splits where every class appears in
# train, validation, and test. Fail safely if the dataset cannot support it.
split_found = False
train_idx = val_idx = test_idx = None
for seed in range(2000):
    outer = GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=seed
    )
    trainval_idx, candidate_test = next(outer.split(X, y, groups))

    if not np.array_equal(np.unique(y[candidate_test]), all_classes):
        continue

    inner = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=seed + 10000
    )
    train_rel, val_rel = next(
        inner.split(
            X[trainval_idx],
            y[trainval_idx],
            groups[trainval_idx]
        )
    )
    candidate_train = trainval_idx[train_rel]
    candidate_val = trainval_idx[val_rel]

    if (
        np.array_equal(np.unique(y[candidate_train]), all_classes)
        and np.array_equal(np.unique(y[candidate_val]), all_classes)
    ):
        train_idx = candidate_train
        val_idx = candidate_val
        test_idx = candidate_test
        split_found = True
        print("Valid participant-wise split found. Seed:", seed)
        break

if not split_found:
    distribution = df.groupby(GROUP)[TARGET].apply(
        lambda s: sorted(encoder.inverse_transform(np.unique(s)).tolist())
    )
    print("\nClasses present for each participant:")
    print(distribution.to_string())
    raise ValueError(
        "\nCould not find a participant-wise split containing every class "
        "in train, validation, and test. No model was trained. The class "
        "distribution across participants must be reviewed; do not use a "
        "random row split because it can leak participant information."
    )

# Confirm there is no participant overlap.
train_people = set(groups[train_idx])
val_people = set(groups[val_idx])
test_people = set(groups[test_idx])
assert train_people.isdisjoint(val_people)
assert train_people.isdisjoint(test_people)
assert val_people.isdisjoint(test_people)

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

print("Train participants:", sorted(train_people))
print("Validation participants:", sorted(val_people))
print("Test participants:", sorted(test_people))
print("Train sequences:", len(train_idx))
print("Validation sequences:", len(val_idx))
print("Test sequences:", len(test_idx))

# Fit preprocessing on training participants only.
scaler = StandardScaler()
scaler.fit(X_train.reshape(-1, len(FEATURES)))

def scale_sequences(a):
    return scaler.transform(
        a.reshape(-1, len(FEATURES))
    ).reshape(a.shape).astype(np.float32)

X_train = scale_sequences(X_train)
X_val = scale_sequences(X_val)
X_test = scale_sequences(X_test)

classes_present = np.unique(y_train)
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes_present,
    y=y_train
)
class_weights = {
    int(label): float(weight)
    for label, weight in zip(classes_present, weights)
}
print("Class weights:", class_weights)

# Compact model to reduce overfitting on this small dataset.
model = Sequential([
    Input(shape=(SEQUENCE_LENGTH, len(FEATURES))),
    LSTM(
        24,
        dropout=0.20,
        recurrent_dropout=0.0,
        kernel_regularizer=l2(0.001)
    ),
    Dropout(0.25),
    Dense(12, activation="relu", kernel_regularizer=l2(0.001)),
    Dropout(0.15),
    Dense(n_classes, activation="softmax"),
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0007),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

best_path = SCRIPT_DIR / f"{PREFIX}best.keras"
final_path = SCRIPT_DIR / f"{PREFIX}final.keras"
scaler_path = SCRIPT_DIR / f"{PREFIX}scaler.joblib"
encoder_path = SCRIPT_DIR / f"{PREFIX}label_encoder.joblib"
pred_path = SCRIPT_DIR / f"{PREFIX}predictions.csv"
metrics_path = SCRIPT_DIR / f"{PREFIX}metrics.json"
report_path = SCRIPT_DIR / f"{PREFIX}classification_report.txt"
cm_path = SCRIPT_DIR / f"{PREFIX}confusion_matrix.png"
history_path = SCRIPT_DIR / f"{PREFIX}training_history.png"

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=6,
        min_lr=0.00001,
        verbose=1
    ),
    ModelCheckpoint(
        str(best_path),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=8,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

model.save(final_path)
eval_model = tf.keras.models.load_model(best_path) if best_path.exists() else model

probabilities = eval_model.predict(X_test, verbose=0)
predictions_encoded = np.argmax(probabilities, axis=1)

accuracy = float(accuracy_score(y_test, predictions_encoded))
macro_precision = float(precision_score(
    y_test, predictions_encoded, average="macro", zero_division=0
))
macro_recall = float(recall_score(
    y_test, predictions_encoded, average="macro", zero_division=0
))
macro_f1 = float(f1_score(
    y_test, predictions_encoded, average="macro", zero_division=0
))
report = classification_report(
    y_test,
    predictions_encoded,
    labels=all_classes,
    target_names=class_names,
    zero_division=0
)

print("\nTEST RESULTS — PARTICIPANT-WISE HOLDOUT")
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Macro precision: {macro_precision:.4f}")
print(f"Macro recall: {macro_recall:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(report)

with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
    f.write(f"Macro precision: {macro_precision:.4f}\n")
    f.write(f"Macro recall: {macro_recall:.4f}\n")
    f.write(f"Macro F1: {macro_f1:.4f}\n\n")
    f.write(report)

cm = confusion_matrix(y_test, predictions_encoded, labels=all_classes)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, interpolation="nearest")
fig.colorbar(im, ax=ax)
ax.set(
    xticks=np.arange(n_classes),
    yticks=np.arange(n_classes),
    xticklabels=class_names,
    yticklabels=class_names,
    xlabel="Predicted label",
    ylabel="True label",
    title="SmartFall LSTM v3 — Participant-Wise Test Confusion Matrix"
)
plt.setp(
    ax.get_xticklabels(),
    rotation=35,
    ha="right",
    rotation_mode="anchor"
)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center")
fig.tight_layout()
fig.savefig(cm_path, dpi=160)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history.get("accuracy", []), label="Train accuracy")
axes[0].plot(history.history.get("val_accuracy", []), label="Validation accuracy")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[1].plot(history.history.get("loss", []), label="Train loss")
axes[1].plot(history.history.get("val_loss", []), label="Validation loss")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()
fig.tight_layout()
fig.savefig(history_path, dpi=160)
plt.close(fig)

prediction_table = pd.DataFrame({
    GROUP: groups[test_idx],
    "sequence_end_day": end_days[test_idx],
    "actual_label": encoder.inverse_transform(y_test),
    "predicted_label": encoder.inverse_transform(predictions_encoded),
})
for index, name in enumerate(class_names):
    safe_name = "".join(ch if ch.isalnum() else "_" for ch in name)
    prediction_table[f"probability_{safe_name}"] = probabilities[:, index]
prediction_table.to_csv(pred_path, index=False)

joblib.dump(scaler, scaler_path)
joblib.dump(encoder, encoder_path)

metrics = {
    "dataset_path": str(DATA_PATH),
    "output_directory": str(SCRIPT_DIR),
    "sequence_length": SEQUENCE_LENGTH,
    "features": FEATURES,
    "classes": class_names,
    "rows_after_cleaning": int(len(df)),
    "participants": int(df[GROUP].nunique()),
    "sequences": int(len(X)),
    "train_sequences": int(len(train_idx)),
    "validation_sequences": int(len(val_idx)),
    "test_sequences": int(len(test_idx)),
    "train_participants": sorted(train_people),
    "validation_participants": sorted(val_people),
    "test_participants": sorted(test_people),
    "test_accuracy": accuracy,
    "macro_precision": macro_precision,
    "macro_recall": macro_recall,
    "macro_f1": macro_f1,
    "note": (
        "Use of simulated/synthetic data does not establish real-world "
        "clinical performance. Holdout results may vary with participant split."
    ),
}
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("\nNew files saved in:", SCRIPT_DIR)
for output_path in [
    best_path, final_path, scaler_path, encoder_path, pred_path,
    metrics_path, report_path, cm_path, history_path
]:
    print(" -", output_path.name)
print("Training completed.")
'''
path = Path("/mnt/data/train_lstm_v3.py")
path.write_text(script, encoding="utf-8")
print(f"Created {path} ({len(script.splitlines())} lines)")
