# 🐾 AnimalAI — Klasifikasi Jenis Hewan dengan CNN

> **Tugas 10 — Kecerdasan Buatan**  
> Muhammad Fadillah Ramadhan · 301240051 · S1 Teknik Informatika

Aplikasi web berbasis **Flask** yang menggunakan **Convolutional Neural Network (CNN)** dengan arsitektur **MobileNetV2** untuk mengklasifikasikan jenis hewan dari gambar yang diunggah.

---

## 🐾 Kelas Hewan

| # | Indonesia | Latin | Folder |
|---|-----------|-------|--------|
| 1 | Anjing | *Canis lupus familiaris* | `cane` |
| 2 | Kucing | *Felis catus* | `gatto` |
| 3 | Kuda | *Equus ferus caballus* | `cavallo` |
| 4 | Laba-laba | *Araneae* | `ragno` |
| 5 | Kupu-kupu | *Lepidoptera* | `farfalla` |
| 6 | Ayam | *Gallus gallus domesticus* | `gallina` |
| 7 | Domba | *Ovis aries* | `pecora` |
| 8 | Sapi | *Bos taurus* | `mucca` |
| 9 | Tupai | *Sciuridae* | `scoiattolo` |
| 10 | Gajah | *Loxodonta / Elephas* | `elefante` |

---

## 🚀 Cara Menjalankan

### 1. Clone & Setup Environment

```bash
git clone https://github.com/liamfadillah/animal-classifier.git
cd animal-classifier

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Download Dataset

Download dari Kaggle: https://www.kaggle.com/datasets/sharansmenon/animals141

Ekstrak ke folder: `model_training/animals/`

Struktur folder:
```
model_training/
└── animals/
    ├── cane/       (Anjing)
    ├── gatto/      (Kucing)
    ├── cavallo/    (Kuda)
    ├── ragno/      (Laba-laba)
    ├── farfalla/   (Kupu-kupu)
    ├── gallina/    (Ayam)
    ├── pecora/     (Domba)
    ├── mucca/      (Sapi)
    ├── scoiattolo/ (Tupai)
    └── elefante/   (Gajah)
```

### 3. Train Model

```bash
cd model_training
python train_model.py
```

Model akan tersimpan di `model/animal_classifier.h5`  
Labels tersimpan di `model/class_labels.json`

### 4. Jalankan Aplikasi

```bash
python app.py
```

Buka browser: http://localhost:5000

---

## 📁 Struktur Project

```
animal-classifier/
├── app.py                    # Flask application
├── requirements.txt          # Dependencies
├── Procfile                  # Untuk deployment
├── railway.json              # Railway config
├── model/
│   ├── animal_classifier.h5  # Trained CNN model
│   └── class_labels.json     # Label mapping
├── model_training/
│   ├── train_model.py        # Training script
│   └── animals/              # Dataset (tidak di-commit)
├── templates/
│   ├── index.html            # Halaman utama
│   └── about.html            # Halaman tentang
└── static/
    ├── uploads/              # Gambar yang diupload user
    └── img/                  # Grafik hasil training
```

---

## 🏗️ Arsitektur Model

```
Input (128x128x3)
       ↓
MobileNetV2 (Transfer Learning, ImageNet)
       ↓
GlobalAveragePooling2D
       ↓
BatchNormalization
       ↓
Dense(256, ReLU) + Dropout(0.5)
       ↓
Dense(128, ReLU) + Dropout(0.3)
       ↓
Dense(10, Softmax)  ← Output: 151 kelas hewan
```

**Training Strategy:**
- Phase 1 (10 epochs): Freeze MobileNetV2, train custom head saja (lr=1e-3)
- Phase 2 (10 epochs): Unfreeze 30 layer terakhir, fine-tuning (lr=1e-5)

---

## 🌐 Deployment ke Railway

1. Push ke GitHub
2. Buat project baru di [railway.app](https://railway.app)
3. Connect GitHub repository
4. Railway otomatis mendeteksi Python dan deploy

---

## 📊 Dataset

- **Nama**: Animal Image Dataset (151 classes)
- **Sumber**: [Kaggle — alessiocorrado99](https://www.kaggle.com/datasets/sharansmenon/animals141)
- **Jumlah Gambar**: lebih dari 30.000
- **Kelas**: 10 jenis hewan
- **Split**: 80% train / 20% validation

---

## 🛠️ Tech Stack

| Layer | Teknologi |
|-------|-----------|
| Backend | Flask 3.x |
| Deep Learning | TensorFlow 2.x / Keras |
| Base Model | MobileNetV2 (ImageNet) |
| Frontend | Bootstrap 5 + Vanilla JS |
| Evaluasi | Scikit-learn |
| Deployment | Railway |
# animal_cnn_klasifikasi_301240051
# animalklasifikasi301230051
