
"""
SmartFall LSTM v3
Participant-wise training, validation, and testing.
Outputs are saved with the smartfall_lstm_v3_ prefix.
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
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

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

# --------------------------------------------------
# 2. FIND DATASET
# --------------------------------------------------

candidates = [
    PROJECT_DIR / "data" / DATA_FILENAME,
    SCRIPT_DIR / "data" / DATA_FILENAME,
    PROJECT_DIR / DATA_FILENAME,
    SCRIPT_DIR / DATA_FILENAME,
]

DATA_PATH = next((p for p in candidates if p.is_file()), None)

if DATA_PATH is None:
    raise FileNotFoundError(
        "Dataset not found. Checked:\n"
        + "\n".join(str(p) for p in candidates)
    )

SCRIPT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("SMARTFALL LSTM V3")
print("=" * 60)
print("Dataset:", DATA_PATH)
print("Output folder:", SCRIPT_DIR)

# --------------------------------------------------
# 3. LOAD AND VALIDATE DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

required = FEATURES + [TARGET, GROUP, DAY]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(
        f"Missing columns: {missing}\n"
        f"Available columns: {df.columns.tolist()}"
    )

df = df.dropna(subset=required).copy()

for col in FEATURES + [DAY]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=FEATURES + [DAY]).copy()

# Avoid non-finite feature values.
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=FEATURES + [DAY]).copy()

df[TARGET] = df[TARGET].astype(str).str.strip()
df[GROUP] = df[GROUP].astype(str).str.strip()

df = df[
    (df[TARGET] != "")
    & (df[GROUP] != "")
].copy()

if df.empty:
    raise ValueError("No valid rows remain after cleaning.")

# Check duplicate participant-day records.
duplicates = df.duplicated(subset=[GROUP, DAY], keep=False)

if duplicates.any():
    print(
        "\nWARNING: Duplicate participant/day records found:",
        int(duplicates.sum()),
    )
    print(
        "Review these records. The script will retain them, "
        "but duplicate days may affect sequence quality."
    )

if df[GROUP].nunique() < 5:
    raise ValueError(
        "At least five distinct participants are required "
        "for this attempted participant-wise split."
    )

encoder = LabelEncoder()
df[TARGET] = encoder.fit_transform(df[TARGET])

class_names = [str(c) for c in encoder.classes_]
n_classes = len(class_names)
all_classes = np.arange(n_classes)

if n_classes < 2:
    raise ValueError("At least two target classes are required.")

print("\nRows after cleaning:", len(df))
print("Unique participants:", df[GROUP].nunique())
print("Classes:", class_names)
print("\nClass row counts:")
print(
    df[TARGET]
    .value_counts()
    .sort_index()
    .rename(index=dict(enumerate(class_names)))
)

# --------------------------------------------------
# 4. CREATE SEQUENCES WITHIN EACH PARTICIPANT
# --------------------------------------------------

X_list = []
y_list = []
group_list = []
day_list = []

for person, part in df.groupby(GROUP):
    part = part.sort_values(DAY).reset_index(drop=True)

    # Duplicate days make the sequence ambiguous.
    if part[DAY].duplicated().any():
        print(
            f"WARNING: Skipping participant {person}: "
            "duplicate day values."
        )
        continue

    values = part[FEATURES].to_numpy(dtype=np.float32)
    labels = part[TARGET].to_numpy(dtype=np.int64)
    days = part[DAY].to_numpy(dtype=float)

    for end in range(SEQUENCE_LENGTH - 1, len(part)):
        start = end - SEQUENCE_LENGTH + 1
        window_days = days[start:end + 1]

        # If all day values are integer-like, require consecutive days.
        if np.allclose(window_days, np.round(window_days)):
            if not np.all(np.diff(window_days) == 1):
                continue

        X_list.append(values[start:end + 1])
        y_list.append(labels[end])
        group_list.append(person)
        day_list.append(days[end])

if not X_list:
    raise ValueError(
        "No sequences were created. Check participant row counts, "
        "day values, and SEQUENCE_LENGTH."
    )

X = np.asarray(X_list, dtype=np.float32)
y = np.asarray(y_list, dtype=np.int64)
groups = np.asarray(group_list)
end_days = np.asarray(day_list)

print("\nTotal sequences:", len(X))
print("Sequence counts by class:")
print(
    pd.Series(y)
    .value_counts()
    .sort_index()
    .rename(index=dict(enumerate(class_names)))
)

# --------------------------------------------------
# 5. FIND PARTICIPANT-DISJOINT SPLIT
# --------------------------------------------------

# Test: 25% of participants.
# Validation: approximately 20% of remaining participants.
# Every class must occur in all three partitions.

split_found = False
train_idx = val_idx = test_idx = None

for seed in range(2000):
    outer = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=seed,
    )

    trainval_idx, candidate_test = next(
        outer.split(X, y, groups)
    )

    if not np.array_equal(
        np.unique(y[candidate_test]), all_classes
    ):
        continue

    inner = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=seed + 10000,
    )

    train_rel, val_rel = next(
        inner.split(
            X[trainval_idx],
            y[trainval_idx],
            groups[trainval_idx],
        )
    )

    candidate_train = trainval_idx[train_rel]
    candidate_val = trainval_idx[val_rel]

    if not np.array_equal(
        np.unique(y[candidate_train]), all_classes
    ):
        continue

    if not np.array_equal(
        np.unique(y[candidate_val]), all_classes
    ):
        continue

    train_idx = candidate_train
    val_idx = candidate_val
    test_idx = candidate_test
    split_found = True

    print("\nValid participant-wise split found.")
    print("Split seed:", seed)
    break

if not split_found:
    print("\nClasses available per participant:")

    class_distribution = df.groupby(GROUP)[TARGET].apply(
        lambda s: [
            class_names[int(i)]
            for i in sorted(s.unique())
        ]
    )

    print(class_distribution.to_string())

    raise ValueError(
        "\nCould not find a participant-wise split with every "
        "class in train, validation, and test after 2000 attempts. "
        "No model was trained. Review the distribution of classes "
        "across participants. Do not use a random row split to "
        "force a result."
    )

train_people = set(groups[train_idx])
val_people = set(groups[val_idx])
test_people = set(groups[test_idx])

assert train_people.isdisjoint(val_people)
assert train_people.isdisjoint(test_people)
assert val_people.isdisjoint(test_people)

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

print("\nTrain participants:", sorted(train_people))
print("Validation participants:", sorted(val_people))
print("Test participants:", sorted(test_people))

print("\nSequence counts:")
print("Train:", len(train_idx))
print("Validation:", len(val_idx))
print("Test:", len(test_idx))

# --------------------------------------------------
# 6. SCALE USING TRAINING DATA ONLY
# --------------------------------------------------

scaler = StandardScaler()

scaler.fit(
    X_train.reshape(-1, len(FEATURES))
)


def scale_sequences(data):
    transformed = scaler.transform(
        data.reshape(-1, len(FEATURES))
    )

    return transformed.reshape(data.shape).astype(np.float32)


X_train = scale_sequences(X_train)
X_val = scale_sequences(X_val)
X_test = scale_sequences(X_test)

# --------------------------------------------------
# 7. CLASS WEIGHTS
# --------------------------------------------------

weights = compute_class_weight(
    class_weight="balanced",
    classes=all_classes,
    y=y_train,
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(all_classes, weights)
}

print("\nClass weights:", class_weights)

# --------------------------------------------------
# 8. BUILD LSTM
# --------------------------------------------------

model = Sequential([
    Input(shape=(SEQUENCE_LENGTH, len(FEATURES))),
    LSTM(
        24,
        dropout=0.20,
        recurrent_dropout=0.0,
        kernel_regularizer=l2(0.001),
    ),
    Dropout(0.25),
    Dense(
        12,
        activation="relu",
        kernel_regularizer=l2(0.001),
    ),
    Dropout(0.15),
    Dense(n_classes, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0007
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# --------------------------------------------------
# 9. OUTPUT PATHS
# --------------------------------------------------

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
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=6,
        min_lr=0.00001,
        verbose=1,
    ),
    ModelCheckpoint(
        filepath=str(best_path),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
]

# --------------------------------------------------
# 10. TRAIN
# --------------------------------------------------

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=8,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1,
)

if not best_path.is_file():
    raise RuntimeError(
        "The best model checkpoint was not created."
    )

# Load the best validation-loss checkpoint.
best_model = tf.keras.models.load_model(best_path)

# Save the same best weights as the final model so the final
# model corresponds to the model evaluated below.
best_model.save(final_path)

# --------------------------------------------------
# 11. TEST EVALUATION
# --------------------------------------------------

probabilities = best_model.predict(X_test, verbose=0)
predictions_encoded = np.argmax(probabilities, axis=1)

accuracy = float(
    accuracy_score(y_test, predictions_encoded)
)

macro_precision = float(
    precision_score(
        y_test,
        predictions_encoded,
        average="macro",
        zero_division=0,
    )
)

macro_recall = float(
    recall_score(
        y_test,
        predictions_encoded,
        average="macro",
        zero_division=0,
    )
)

macro_f1 = float(
    f1_score(
        y_test,
        predictions_encoded,
        average="macro",
        zero_division=0,
    )
)

report = classification_report(
    y_test,
    predictions_encoded,
    labels=all_classes,
    target_names=class_names,
    zero_division=0,
)

cm = confusion_matrix(
    y_test,
    predictions_encoded,
    labels=all_classes,
)

print("\n" + "=" * 60)
print("PARTICIPANT-WISE TEST RESULTS")
print("=" * 60)
print(f"Accuracy:        {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Macro precision: {macro_precision:.4f}")
print(f"Macro recall:    {macro_recall:.4f}")
print(f"Macro F1:        {macro_f1:.4f}")
print("\nClassification report:")
print(report)

with open(report_path, "w", encoding="utf-8") as file:
    file.write(
        f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n"
    )
    file.write(f"Macro precision: {macro_precision:.4f}\n")
    file.write(f"Macro recall: {macro_recall:.4f}\n")
    file.write(f"Macro F1: {macro_f1:.4f}\n\n")
    file.write(report)

# --------------------------------------------------
# 12. CONFUSION MATRIX GRAPH
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))

image = ax.imshow(cm, interpolation="nearest")
fig.colorbar(image, ax=ax)

ax.set(
    xticks=np.arange(n_classes),
    yticks=np.arange(n_classes),
    xticklabels=class_names,
    yticklabels=class_names,
    xlabel="Predicted label",
    ylabel="True label",
    title="SmartFall LSTM v3 - Participant-Wise Test",
)

plt.setp(
    ax.get_xticklabels(),
    rotation=35,
    ha="right",
    rotation_mode="anchor",
)

for row in range(cm.shape[0]):
    for col in range(cm.shape[1]):
        ax.text(
            col,
            row,
            str(cm[row, col]),
            ha="center",
            va="center",
        )

fig.tight_layout()
fig.savefig(cm_path, dpi=160)
plt.close(fig)

# --------------------------------------------------
# 13. TRAINING HISTORY GRAPH
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(
    history.history.get("accuracy", []),
    label="Train accuracy",
)
axes[0].plot(
    history.history.get("val_accuracy", []),
    label="Validation accuracy",
)
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(
    history.history.get("loss", []),
    label="Train loss",
)
axes[1].plot(
    history.history.get("val_loss", []),
    label="Validation loss",
)
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()

fig.tight_layout()
fig.savefig(history_path, dpi=160)
plt.close(fig)

# --------------------------------------------------
# 14. EXPORT PREDICTIONS FOR RAPIDMINER
# --------------------------------------------------

prediction_table = pd.DataFrame({
    GROUP: groups[test_idx],
    "sequence_end_day": end_days[test_idx],
    "actual_label": encoder.inverse_transform(y_test),
    "predicted_label": encoder.inverse_transform(
        predictions_encoded
    ),
})

for index, class_name in enumerate(class_names):
    safe_name = "".join(
        char if char.isalnum() else "_"
        for char in class_name
    )

    prediction_table[f"probability_{safe_name}"] = (
        probabilities[:, index]
    )

prediction_table.to_csv(pred_path, index=False)

# --------------------------------------------------
# 15. SAVE PREPROCESSING OBJECTS AND METRICS
# --------------------------------------------------

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
    "total_sequences": int(len(X)),
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
    "confusion_matrix": cm.tolist(),
    "note": (
        "Results are from a participant-wise holdout. "
        "If the data is synthetic or simulated, these results "
        "do not establish real-world clinical performance."
    ),
}

with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

# --------------------------------------------------
# 16. FINISH
# --------------------------------------------------

print("\nTraining and evaluation completed.")
print("New files saved in:", SCRIPT_DIR)

for output_path in [
    best_path,
    final_path,
    scaler_path,
    encoder_path,
    pred_path,
    metrics_path,
    report_path,
    cm_path,
    history_path,
]:
    print(" -", output_path.name)

print("\nRapidMiner import file:")
print(pred_path)
