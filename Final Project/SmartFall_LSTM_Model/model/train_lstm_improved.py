
import os
import json
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
import joblib

# -------------------- FOLDERS --------------------
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = Path(r"C:\Users\PRAGATI\Desktop\SmartFall_LSTM_Model\data\smartfall_rnn_clean_ready_to_use.csv")
)
RESULTS = os.path.join(BASE, "results")
MODEL_DIR = os.path.join(BASE, "model")
GRAPHS = os.path.join(BASE, "graphs")

for folder in [RESULTS, MODEL_DIR, GRAPHS]:
    os.makedirs(folder, exist_ok=True)

# -------------------- SETTINGS --------------------
SEQUENCE_LENGTH = 5
TEST_SIZE = 0.25
EPOCHS = 50
BATCH_SIZE = 16
SEED = 42

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# Use original movement measurements only.
# Derived rolling/lag features are deliberately excluded.
FEATURES = [
    "gait_speed",
    "step_time_variability",
    "turn_duration",
    "sway_rms",
    "accel_rms"
]

TARGET = "scenario_label"
GROUP = "person_id"
DAY = "day"

# -------------------- LOAD DATA --------------------
if not os.path.isfile(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, sep=None, engine="python")
df.columns = df.columns.str.strip()

required = FEATURES + [TARGET, GROUP, DAY]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=required).copy()
df[DAY] = pd.to_numeric(df[DAY], errors="raise")
df = df.sort_values([GROUP, DAY]).reset_index(drop=True)

if df.duplicated([GROUP, DAY]).any():
    raise ValueError(
        "Duplicate person_id/day rows found. Resolve duplicates before training."
    )

if not np.isfinite(df[FEATURES].to_numpy(dtype=float)).all():
    raise ValueError("Movement features contain non-finite values.")

print("Dataset shape:", df.shape)
print("Participants:", df[GROUP].nunique())
print("Labels:\n", df[TARGET].value_counts())

# -------------------- BUILD SEQUENCES --------------------
# Each sequence contains five consecutive days for one participant.
# The label belongs to the final day in each sequence.
X_list, y_list, person_list, day_list = [], [], [], []

for person, part in df.groupby(GROUP, sort=False):
    part = part.sort_values(DAY).reset_index(drop=True)
    values = part[FEATURES].to_numpy(dtype=np.float32)
    labels = part[TARGET].astype(str).to_numpy()
    days = part[DAY].to_numpy()

    for end in range(SEQUENCE_LENGTH - 1, len(part)):
        start = end - SEQUENCE_LENGTH + 1

        # Do not bridge gaps in a participant's timeline.
        if not np.all(np.diff(days[start:end + 1]) == 1):
            continue

        X_list.append(values[start:end + 1])
        y_list.append(labels[end])
        person_list.append(person)
        day_list.append(days[end])

if len(X_list) < 12:
    raise ValueError(
        f"Only {len(X_list)} valid sequences found. "
        "Check whether each participant has five consecutive days."
    )

X = np.asarray(X_list, dtype=np.float32)
y_text = np.asarray(y_list)
groups = np.asarray(person_list)
target_days = np.asarray(day_list)

print("Sequence data shape:", X.shape)
print("Sequence labels:\n", pd.Series(y_text).value_counts())

# -------------------- ENCODE LABELS --------------------
encoder = LabelEncoder()
y = encoder.fit_transform(y_text)
classes = np.arange(len(encoder.classes_))

if len(classes) < 2:
    raise ValueError("At least two scenario classes are required.")

# -------------------- PARTICIPANT-LEVEL SPLIT --------------------
# No participant may occur in both training and test sets.
splitter = GroupShuffleSplit(
    n_splits=100,
    test_size=TEST_SIZE,
    random_state=SEED
)

chosen = None
best_coverage = -1

for train_idx, test_idx in splitter.split(X, y, groups):
    train_classes = set(y[train_idx])
    test_classes = set(y[test_idx])
    coverage = len(train_classes) + len(test_classes)

    if (
        train_classes == set(classes)
        and test_classes == set(classes)
    ):
        chosen = (train_idx, test_idx)
        break

    if coverage > best_coverage:
        best_coverage = coverage
        chosen = (train_idx, test_idx)

train_idx, test_idx = chosen

train_classes = set(y[train_idx])
test_classes = set(y[test_idx])

if train_classes != set(classes):
    raise ValueError(
        "The training participants do not cover all classes. "
        "More suitable participant data is needed."
    )

if test_classes != set(classes):
    warnings.warn(
        "The held-out participants do not contain every class. "
        "The evaluation report will show unsupported classes."
    )

train_people = set(groups[train_idx])
test_people = set(groups[test_idx])

assert train_people.isdisjoint(test_people)

print("Training sequences:", len(train_idx))
print("Testing sequences:", len(test_idx))
print("Training participants:", sorted(train_people))
print("Testing participants:", sorted(test_people))

# -------------------- SCALE FEATURES --------------------
# Fit scaling only on training data to avoid test-data leakage.
scaler = StandardScaler()
n_features = X.shape[2]

scaler.fit(X[train_idx].reshape(-1, n_features))

X_scaled = scaler.transform(
    X.reshape(-1, n_features)
).reshape(X.shape).astype(np.float32)

X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# -------------------- BUILD LSTM --------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(SEQUENCE_LENGTH, n_features)
    ),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(len(encoder.classes_), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "smartfall_lstm_best.keras"),
        monitor="val_loss",
        save_best_only=True
    )
]

# Validation is taken only from the training participant set.
history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=callbacks,
    shuffle=True,
    verbose=1
)

# -------------------- TEST EVALUATION --------------------
probabilities = model.predict(X_test, verbose=0)
predictions = probabilities.argmax(axis=1)

accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(
    y_test, predictions, labels=classes,
    average="macro", zero_division=0
)

report = classification_report(
    y_test,
    predictions,
    labels=classes,
    target_names=encoder.classes_,
    output_dict=True,
    zero_division=0
)

metrics = {
    "model": "LSTM",
    "sequence_length": SEQUENCE_LENGTH,
    "features": FEATURES,
    "target": TARGET,
    "test_size": TEST_SIZE,
    "train_sequences": int(len(train_idx)),
    "test_sequences": int(len(test_idx)),
    "train_participants": sorted(map(str, train_people)),
    "test_participants": sorted(map(str, test_people)),
    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "classes": encoder.classes_.tolist(),
    "data_is_synthetic": bool(
        df["is_simulated"].astype(str).str.lower()
        .isin(["true", "1", "yes"]).all()
    ) if "is_simulated" in df.columns else None
}

with open(os.path.join(RESULTS, "lstm_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

pd.DataFrame(report).transpose().to_csv(
    os.path.join(RESULTS, "lstm_classification_report.csv"),
    index_label="class"
)

cm = confusion_matrix(y_test, predictions, labels=classes)
pd.DataFrame(
    cm,
    index=encoder.classes_,
    columns=encoder.classes_
).to_csv(os.path.join(RESULTS, "lstm_confusion_matrix.csv"))

prediction_df = pd.DataFrame({
    "person_id": groups[test_idx],
    "day": target_days[test_idx],
    "actual_label": encoder.inverse_transform(y_test),
    "predicted_label": encoder.inverse_transform(predictions),
    "correct": (y_test == predictions)
})

for i, label in enumerate(encoder.classes_):
    prediction_df["prob_" + str(label)] = probabilities[:, i]

prediction_df.to_csv(
    os.path.join(RESULTS, "lstm_test_predictions.csv"),
    index=False
)

# -------------------- SAVE MODEL AND PREPROCESSORS --------------------
model.save(os.path.join(MODEL_DIR, "smartfall_lstm_final.keras"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "lstm_scaler.joblib"))
joblib.dump(encoder, os.path.join(MODEL_DIR, "lstm_label_encoder.joblib"))

# -------------------- SAVE GRAPHS --------------------
plt.figure()
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("SmartFall LSTM Training")
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
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS, "lstm_confusion_matrix.png"), dpi=150)
plt.close()

print("\n========== LSTM TRAINING COMPLETE ==========")
print(f"Test accuracy: {accuracy:.4f}")
print(f"Macro F1:      {macro_f1:.4f}")
print("Metrics:       results/lstm_metrics.json")
print("Predictions:   results/lstm_test_predictions.csv")
print("Report:        results/lstm_classification_report.csv")
print("Confusion:     results/lstm_confusion_matrix.csv")
print("Model:         model/smartfall_lstm_final.keras")
print("Graphs:        graphs/")
