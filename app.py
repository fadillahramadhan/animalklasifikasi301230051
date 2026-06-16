"""
app.py
======
Aplikasi Web Flask - Klasifikasi 151 Jenis Hewan dengan CNN
Dataset : Animal Image Dataset (151 classes) - Kaggle
Nama    : Muhammad Fadillah Ramadhan
NIM     : 301240051
"""

import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import uuid

# ─────────────────────────────────────────
app = Flask(__name__)
app.config["UPLOAD_FOLDER"]       = "static/uploads"
app.config["MAX_CONTENT_LENGTH"]  = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ─────────────────────────────────────────
# LOAD MODEL & LABELS
# ─────────────────────────────────────────
print("Loading model CNN (151 kelas)...")
model = load_model("model/animal_classifier.h5")
print("✅ Model berhasil dimuat!")

with open("model/class_labels.json", "r", encoding="utf-8") as f:
    raw = json.load(f)
    CLASS_LABELS = {int(k): v for k, v in raw.items()}

NUM_CLASSES = len(CLASS_LABELS)
IMG_SIZE    = (128, 128)

# Emoji map untuk beberapa hewan umum
EMOJI_MAP = {
    "antelope":"🦌","badger":"🦡","bat":"🦇","bear":"🐻","bee":"🐝",
    "beetle":"🪲","butterfly":"🦋","cat":"🐈","caterpillar":"🐛",
    "chimpanzee":"🐒","cockroach":"🪳","cow":"🐄","coyote":"🐺",
    "crab":"🦀","crow":"🐦","deer":"🦌","dog":"🐕","dolphin":"🐬",
    "donkey":"🫏","dragonfly":"🪷","duck":"🦆","eagle":"🦅",
    "elephant":"🐘","flamingo":"🦩","fly":"🪰","fox":"🦊",
    "goat":"🐐","goldfish":"🐟","goose":"🪿","gorilla":"🦍",
    "grasshopper":"🦗","hamster":"🐹","hare":"🐇","hedgehog":"🦔",
    "hippopotamus":"🦛","hornbill":"🐦","horse":"🐎","hummingbird":"🐦",
    "hyena":"🐾","jellyfish":"🪼","kangaroo":"🦘","koala":"🐨",
    "ladybugs":"🐞","leopard":"🐆","lion":"🦁","lizard":"🦎",
    "lobster":"🦞","mosquito":"🦟","moth":"🦋","mouse":"🐭",
    "octopus":"🐙","okapi":"🦌","orangutan":"🦧","otter":"🦦",
    "owl":"🦉","ox":"🐂","oyster":"🦪","panda":"🐼","parrot":"🦜",
    "pelecaniformes":"🐦","penguin":"🐧","pig":"🐷","pigeon":"🕊️",
    "porcupine":"🦔","possum":"🐾","raccoon":"🦝","rat":"🐀",
    "reindeer":"🦌","rhinoceros":"🦏","sandpiper":"🐦","seahorse":"🐴",
    "seal":"🦭","shark":"🦈","sheep":"🐑","snake":"🐍","sparrow":"🐦",
    "squid":"🦑","squirrel":"🐿️","starfish":"⭐","swan":"🦢",
    "tiger":"🐯","turkey":"🦃","turtle":"🐢","whale":"🐳",
    "wolf":"🐺","wombat":"🐾","woodpecker":"🐦","zebra":"🦓",
}

def get_emoji(label):
    key = label.lower().split()[0]
    return EMOJI_MAP.get(key, "🐾")

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_animal(img_path):
    img       = image.load_img(img_path, target_size=IMG_SIZE)
    arr       = image.img_to_array(img) / 255.0
    arr       = np.expand_dims(arr, axis=0)
    preds     = model.predict(arr, verbose=0)[0]
    top5_idx  = np.argsort(preds)[::-1][:5]
    results   = []
    for idx in top5_idx:
        label = CLASS_LABELS.get(idx, f"class_{idx}")
        # Capitalize tiap kata, ganti underscore dengan spasi
        label_display = label.replace("_", " ").title()
        results.append({
            "rank":       len(results) + 1,
            "label":      label_display,
            "confidence": float(preds[idx]) * 100,
            "emoji":      get_emoji(label),
        })
    return results

# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────
@app.route("/")
def index():
    # Kirim 20 sampel nama hewan untuk ditampilkan di homepage
    sample_animals = [
        {"name": CLASS_LABELS[i].replace("_"," ").title(), "emoji": get_emoji(CLASS_LABELS[i])}
        for i in range(min(20, NUM_CLASSES))
    ]
    return render_template("index.html",
                           sample_animals=sample_animals,
                           num_classes=NUM_CLASSES)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file."}), 400
    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "File tidak valid atau format tidak didukung."}), 400

    ext      = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    results   = predict_animal(filepath)
    image_url = url_for("static", filename=f"uploads/{filename}")
    return jsonify({"success": True, "image_url": image_url, "results": results})

@app.route("/about")
def about():
    return render_template("about.html", num_classes=NUM_CLASSES)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
