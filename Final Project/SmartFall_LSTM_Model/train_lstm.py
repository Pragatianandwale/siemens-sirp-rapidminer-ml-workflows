from pathlib import Path

script = r'''import os
import json
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# -------------------- FOLDERS --------------------
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "data", "smartfall_rnn_clean_ready_to_use.csv")
RESULTS = os.path.join(BASE, "results")
MODEL_DIR = os.path.join(BASE, "model")
GRAPHS = os.path.join(BASE, "graphs")

for folder in (RESULTS, MODEL_DIR, GRAPHS):
    os.makedirs(folder, exist_ok=True)

# -------------------- SETTINGS --------------------
SEQUENCE_LENGTH = 5
TEST_SIZE = 0.25
VALIDATION_SIZE = 0.20
EPOCHS = 100
BATCH_SIZE = 8
SEED = 42

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

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

# -------------------- LOAD AND CHECK DATA --------------------
if not os.path.isfile(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, sep=None, engine="python")
df.columns = df.columns.str.strip()

required = FEATURES + [TARGET, GROUP, DAY]
missing = [column for column in required if column not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=required).copy()
df[DAY] = pd.to_numeric(df[DAY], errors="raise")
df = df.sort_values([GROUP, DAY]).reset_index(drop=True)

if df.duplicated([GROUP, DAY]).any():
    raise ValueError("Duplicate person_id/day rows found; resolve them before training.")
if not np.isfinite(df[FEATURES].to_numpy(dtype=float)).all():
    raise ValueError("Movement features contain non-finite values.")

print("Dataset shape:", df.shape)
print("Participants:", df[GROUP].nunique())
print("Labels:\n", df[TARGET].value_counts())

# -------------------- BUILD SEQUENCES --------------------
# A sequence uses five consecutive days from one participant.
# Its target is the scenario label on the final day.
X_list, y_list, person_list, day_list = [], [], [], []

for person, part in df.groupby(GROUP, sort=False):
    part = part.sort_values(DAY).reset_index(drop=True)
    values = part[FEATURES].to_numpy(dtype=np.float32)
    labels = part[TARGET].astype(str).to_numpy()
    days = part[DAY].to_numpy()

    for end in range(SEQUENCE_LENGTH - 1, len(part)):
        start = end - SEQUENCE_LENGTH + 1
        if not np.all(np.diff(days[start:end + 1]) == 1):
            continue
        X_list.append(values[start:end + 1])
        y_list.append(labels[end])
        person_list.append(str(person))
        day_list.append(days[end])

if len(X_list) < 20:
    raise ValueError(
        f"Only {len(X_list)} valid sequences found. "
        "Check whether participants have five consecutive days."
    )

X = np.asarray(X_list, dtype=np.float32)
y_text = np.asarray(y_list)
groups = np.asarray(person_list)
target_days = np.asarray(day_list)

encoder = LabelEncoder()
y = encoder.fit_transform(y_text)
classes = np.arange(len(encoder.classes_))
all_classes = set(classes)

if len(classes) < 2:
    raise ValueError("At least two scenario classes are required.")

print("Sequence data shape:", X.shape)
print("Sequence labels:\n", pd.Series(y_text).value_counts())

# -------------------- GROUP-AWARE SPLIT HELPERS --------------------
# Split by participant, not by individual sequence, to prevent leakage.
def choose_group_split(indices, test_fraction, required_classes, seed_offset=0, attempts=500):
    """Return local train/test indices, preferring every class in both sets."""
    local_y = y[indices]
    local_groups = groups[indices]
    splitter = GroupShuffleSplit(
        n_splits=attempts,
        test_size=test_fraction,
        random_state=SEED + seed_offset,
    )

    best = None
    best_score = -1
    for local_train, local_test in splitter.split(
        np.zeros(len(indices)), local_y, local_groups
    ):
        train_classes = set(local_y[local_train])
        test_classes = set(local_y[local_test])

        # Training must contain every class so the model can learn each one.
        if train_classes != required_classes:
            continue

        score = len(test_classes)
        if score > best_score:
            best = (local_train, local_test)
            best_score = score

        if test_classes == required_classes:
            return local_train, local_test, True

    if best is None:
        raise ValueError(
            "Could not create a participant-level split with every class in "
            "the training set. More representative participant data is needed."
        )
    return best[0], best[1], False

all_indices = np.arange(len(X))
outer_train_local, test_local, test_has_all_classes = choose_group_split(
    all_indices, TEST_SIZE, all_classes, seed_offset=0
)
trainval_idx = all_indices[outer_train_local]
test_idx = all_indices[test_local]

# Make a separate participant-level validation split only from outer training.
inner_train_local, val_local, val_has_all_classes = choose_group_split(
    trainval_idx, VALIDATION_SIZE, all_classes, seed_offset=1000
)
train_idx = trainval_idx[inner_train_local]
val_idx = trainval_idx[val_local]

train_people = set(groups[train_idx])
val_people = set(groups[val_idx])
test_people = set(groups[test_idx])

assert train_people.isdisjoint(val_people)
assert train_people.isdisjoint(test_people)
assert val_people.isdisjoint(test_people)

if not test_has_all_classes:
    warnings.warn(
        "The held-out test participants do not contain every class. "
        "Interpret per-class metrics for absent classes cautiously."
    )
if not val_has_all_classes:
    warnings.warn(
        "The validation participants do not contain every class. "
        "Validation accuracy may be unstable; do not use it as a final score."
    )

print("Training sequences:", len(train_idx), "| participants:", sorted(train_people))
print("Validation sequences:", len(val_idx), "| participants:", sorted(val_people))
print("Testing sequences:", len(test_idx), "| participants:", sorted(test_people))
print("Train class counts:\n", pd.Series(y[train_idx]).map(
    dict(enumerate(encoder.classes_))
).value_counts())
print("Validation class counts:\n", pd.Series(y[val_idx]).map(
    dict(enumerate(encoder.classes_))
).value_counts())

# -------------------- SCALE WITHOUT DATA LEAKAGE --------------------
scaler = StandardScaler()
n_features = X.shape[2]
scaler.fit(X[train_idx].reshape(-1, n_features))

X_scaled = scaler.transform(X.reshape(-1, n_features)).reshape(X.shape).astype(np.float32)
X_train, X_val, X_test = X_scaled[train_idx], X_scaled[val_idx], X_scaled[test_idx]
y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

# Balance classes in the training set only.
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train,
)
class_weights = {int(label): float(weight) for label, weight in zip(np.unique(y_train), weights)}
print("Class weights:", class_weights)

# -------------------- SMALLER, REGULARIZED LSTM --------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, n_features)),
    tf.keras.layers.LSTM(
        16,
        dropout=0.20,
        recurrent_dropout=0.0,
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
    ),
    tf.keras.layers.Dropout(0.35),
    tf.keras.layers.Dense(
        8,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
    ),
    tf.keras.layers.Dense(len(encoder.classes_), activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=0.00001,
        verbose=1,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "smartfall_lstm_best.keras"),
        monitor="val_loss",
        save_best_only=True,
        verbose=0,
    ),
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks,
    shuffle=True,
    verbose=1,
)

# -------------------- TEST EVALUATION --------------------
probabilities = model.predict(X_test, verbose=0)
predictions = probabilities.argmax(axis=1)

accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(
    y_test, predictions, labels=classes, average="macro", zero_division=0
)
report_text = classification_report(
    y_test,
    predictions,
    labels=classes,
    target_names=encoder.classes_,
    zero_division=0,
)
report_dict = classification_report(
    y_test,
    predictions,
    labels=classes,
    target_names=encoder.classes_,
    output_dict=True,
    zero_division=0,
)
print("\nTest classification report:\n", report_text)

metrics = {
    "model": "LSTM_regularized_group_validation_class_weighted",
    "sequence_length": SEQUENCE_LENGTH,
    "features": FEATURES,
    "target": TARGET,
    "test_size": TEST_SIZE,
    "validation_size_within_trainval": VALIDATION_SIZE,
    "train_sequences": int(len(train_idx)),
    "validation_sequences": int(len(val_idx)),
    "test_sequences": int(len(test_idx)),
    "train_participants": sorted(map(str, train_people)),
    "validation_participants": sorted(map(str, val_people)),
    "test_participants": sorted(map(str, test_people)),
    "test_accuracy": float(accuracy),
    "test_macro_f1": float(macro_f1),
    "classes": encoder.classes_.tolist(),
    "validation_contains_all_classes": bool(val_has_all_classes),
    "test_contains_all_classes": bool(test_has_all_classes),
    "data_is_synthetic": bool(
        df["is_simulated"].astype(str).str.lower().isin(["true", "1", "yes"]).all()
    ) if "is_simulated" in df.columns else None,
}

with open(os.path.join(RESULTS, "lstm_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

pd.DataFrame(report_dict).transpose().to_csv(
    os.path.join(RESULTS, "lstm_classification_report.csv"),
    index_label="class",
)

cm = confusion_matrix(y_test, predictions, labels=classes)
pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_).to_csv(
    os.path.join(RESULTS, "lstm_confusion_matrix.csv")
)

prediction_df = pd.DataFrame({
    "person_id": groups[test_idx],
    "day": target_days[test_idx],
    "actual_label": encoder.inverse_transform(y_test),
    "predicted_label": encoder.inverse_transform(predictions),
    "correct": (y_test == predictions),
})
for i, label in enumerate(encoder.classes_):
    prediction_df["prob_" + str(label)] = probabilities[:, i]
prediction_df.to_csv(
    os.path.join(RESULTS, "lstm_test_predictions.csv"), index=False
)

# Save final model and preprocessing objects. The saved model uses the best
# weights restored by EarlyStopping.
model.save(os.path.join(MODEL_DIR, "smartfall_lstm_final.keras"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "lstm_scaler.joblib"))
joblib.dump(encoder, os.path.join(MODEL_DIR, "lstm_label_encoder.joblib"))

# -------------------- GRAPHS --------------------
plt.figure()
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("SmartFall LSTM Training Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS, "lstm_training_loss.png"), dpi=150)
plt.close()

plt.figure(figsize=(8, 6))
plt.imshow(cm)
plt.title("LSTM Test Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("Actual label")
plt.xticks(classes, encoder.classes_, rotation=35, ha="right")
plt.yticks(classes, encoder.classes_)
for row in range(cm.shape[0]):
    for col in range(cm.shape[1]):
        plt.text(col, row, str(cm[row, col]), ha="center", va="center")
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS, "lstm_confusion_matrix.png"), dpi=150)
plt.close()

print("\n========== LSTM TRAINING COMPLETE ==========")
print(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Macro F1:      {macro_f1:.4f}")
print("Metrics:       results/lstm_metrics.json")
print("Predictions:   results/lstm_test_predictions.csv")
print("Report:        results/lstm_classification_report.csv")
print("Confusion:     results/lstm_confusion_matrix.csv")
print("Model:         model/smartfall_lstm_final.keras")
print("Graphs:        graphs/")
print("Note: Results are experimental and depend on this participant-level split.")
'''
