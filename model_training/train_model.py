"""
train_model.py
==============
Script training CNN - Klasifikasi 151 Jenis Hewan
Dataset: Animal Image Dataset (151 classes) - Kaggle
Link: https://www.kaggle.com/datasets/sharansmenon/animals141

Cara penggunaan:
1. Download dataset dari Kaggle
2. Ekstrak ke folder: model_training/animals/
3. Jalankan: python train_model.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import json

# ─────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────
IMG_SIZE    = (128, 128)
BATCH_SIZE  = 32
EPOCHS      = 25
DATA_DIR    = "animals"          # Folder dataset hasil ekstrak Kaggle
MODEL_PATH  = "../model/animal_classifier.h5"
LABELS_PATH = "../model/class_labels.json"

# ─────────────────────────────────────────
# 1. PERSIAPAN DATA
# ─────────────────────────────────────────
print("=" * 60)
print("  KLASIFIKASI 151 JENIS HEWAN - TRAINING CNN")
print("=" * 60)

# Data augmentation untuk training
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.25,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    validation_split=0.2,
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2,
)

print("\n[1/5] Loading dataset...")
train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
)

val_gen = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
)

NUM_CLASSES   = len(train_gen.class_indices)
idx_to_class  = {v: k for k, v in train_gen.class_indices.items()}

# Simpan labels (nama folder = nama hewan, sudah bahasa Inggris)
os.makedirs("../model", exist_ok=True)
with open(LABELS_PATH, "w", encoding="utf-8") as f:
    json.dump(idx_to_class, f, ensure_ascii=False, indent=2)

print(f"\nTotal kelas  : {NUM_CLASSES}")
print(f"Data training: {train_gen.samples}")
print(f"Data validasi: {val_gen.samples}")

# ─────────────────────────────────────────
# 2. MEMBANGUN MODEL CNN
# ─────────────────────────────────────────
print("\n[2/5] Membangun model CNN (MobileNetV2 Transfer Learning)...")

base_model = keras.applications.MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # Freeze dulu

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(512, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ─────────────────────────────────────────
# 3. TRAINING - Phase 1: Feature Extraction
# ─────────────────────────────────────────
print("\n[3/5] Training Phase 1 - Feature Extraction (10 epoch)...")

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
    keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True),
]

history1 = model.fit(
    train_gen, epochs=10,
    validation_data=val_gen,
    callbacks=callbacks,
)

# Phase 2: Fine-tuning
print("\nPhase 2 - Fine-tuning 40 layer terakhir MobileNetV2...")
base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history2 = model.fit(
    train_gen, epochs=EPOCHS,
    initial_epoch=10,
    validation_data=val_gen,
    callbacks=callbacks,
)

# Gabungkan history
all_acc      = history1.history["accuracy"]     + history2.history["accuracy"]
all_val_acc  = history1.history["val_accuracy"] + history2.history["val_accuracy"]
all_loss     = history1.history["loss"]         + history2.history["loss"]
all_val_loss = history1.history["val_loss"]     + history2.history["val_loss"]

# ─────────────────────────────────────────
# 4. EVALUASI
# ─────────────────────────────────────────
print("\n[4/5] Evaluasi model...")
loss, accuracy = model.evaluate(val_gen)
print(f"\nAkurasi Validasi : {accuracy * 100:.2f}%")
print(f"Loss Validasi    : {loss:.4f}")

val_gen.reset()
y_pred = np.argmax(model.predict(val_gen), axis=1)
y_true = val_gen.classes
label_names = [idx_to_class[i] for i in range(NUM_CLASSES)]

print("\nClassification Report (per kelas):")
print(classification_report(y_true, y_pred, target_names=label_names))

# ─────────────────────────────────────────
# 5. VISUALISASI
# ─────────────────────────────────────────
print("\n[5/5] Menyimpan grafik...")
os.makedirs("../static/img", exist_ok=True)

# Grafik Akurasi & Loss
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Training CNN - Klasifikasi 151 Jenis Hewan", fontsize=14, fontweight="bold")

axes[0].plot(all_acc,     label="Train Accuracy", color="#2196F3", linewidth=2)
axes[0].plot(all_val_acc, label="Val Accuracy",   color="#4CAF50", linewidth=2)
axes[0].axvline(x=9, color="gray", linestyle="--", alpha=0.5, label="Fine-tune start")
axes[0].set_title("Akurasi Training vs Validasi")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Akurasi")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(all_loss,     label="Train Loss", color="#F44336", linewidth=2)
axes[1].plot(all_val_loss, label="Val Loss",   color="#FF9800", linewidth=2)
axes[1].axvline(x=9, color="gray", linestyle="--", alpha=0.5, label="Fine-tune start")
axes[1].set_title("Loss Training vs Validasi")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("../static/img/training_history.png", dpi=150, bbox_inches="tight")
plt.close()

# Confusion Matrix - karena 151 kelas, buat ukuran besar
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(40, 36))
sns.heatmap(cm, annot=False, cmap="Blues",
            xticklabels=label_names, yticklabels=label_names,
            linewidths=0.3)
plt.xlabel("Prediksi", fontsize=14)
plt.ylabel("Label Asli", fontsize=14)
plt.title("Confusion Matrix - 151 Kelas Hewan", fontsize=16, fontweight="bold")
plt.xticks(fontsize=5, rotation=90)
plt.yticks(fontsize=5)
plt.tight_layout()
plt.savefig("../static/img/confusion_matrix.png", dpi=100, bbox_inches="tight")
plt.close()

print("\n✅ Training selesai!")
print(f"   Model    : {MODEL_PATH}")
print(f"   Labels   : {LABELS_PATH}")
print(f"   Akurasi  : {accuracy * 100:.2f}%")
