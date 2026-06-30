import os
import io
import json
import logging

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", static_url_path="")

# ─── CONFIGURATION (env-driven, no hardcoded secrets/paths) ───
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5000").split(",")
CORS(app, origins=ALLOWED_ORIGINS)

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "10"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# Fixed: flat, explicit paths matching the actual shipped artifacts
# (previously: os.path.join("skin_disease_v2.keras", "model.weights.h5") -> never existed)
CONFIG_PATH    = os.environ.get("MODEL_CONFIG_PATH", "config.json")
WEIGHTS_PATH   = os.environ.get("MODEL_WEIGHTS_PATH", "model_weights.h5")
LEGACY_H5_PATH = os.environ.get("MODEL_LEGACY_PATH", "skin_disease_mobilenet.h5")
IMAGE_SIZE     = (224, 224)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

# ─── LOAD CLASS NAMES ───
try:
    with open("class_names.json", "r") as f:
        CLASS_NAMES = json.load(f)
    logger.info("Successfully loaded %d classes", len(CLASS_NAMES))
except Exception:
    logger.exception("Failed to load class_names.json — falling back to hardcoded list")
    CLASS_NAMES = [
        "Actinic keratosis", "Atopic Dermatitis", "Benign keratosis",
        "Dermatofibroma", "Melanocytic nevus", "Melanoma",
        "Squamous cell carcinoma", "Tinea Ringworm Candidiasis", "Vascular lesion"
    ]

# ─── MEDICAL DATA ───
# NOTE: ideally moved to data/class_info.json and loaded here, kept inline for parity with original.
CLASS_INFO = {
    "Actinic keratosis": {
        "description": "Rough, scaly patches caused by years of sun exposure. This condition is considered precancerous.",
        "recommendation": "Consult a dermatologist for evaluation. Regularly apply SPF 50+ sunscreen and wear protective clothing."
    },
    "Atopic Dermatitis": {
        "description": "Also known as Eczema. A chronic, inflammatory skin condition causing intensely itchy, red, dry patches.",
        "recommendation": "Keep skin deeply moisturized with thick emollients, avoid harsh soaps, and talk to a doctor regarding topical creams."
    },
    "Benign keratosis": {
        "description": "Non-cancerous skin growths that look waxy, scaly, or crusty. Very common with aging skin.",
        "recommendation": "Generally completely harmless. No treatment required unless it gets irritated or removed for cosmetics."
    },
    "Dermatofibroma": {
        "description": "Common, benign, firm nodules that usually form on the lower legs.",
        "recommendation": "Harmless growths. Monitor for drastic changes in size or color, but otherwise require no action."
    },
    "Melanocytic nevus": {
        "description": "A standard benign skin mole made up of pigment-producing cells.",
        "recommendation": "Perform regular skin checks using the ABCDE self-exam. Visit a doctor if a mole grows, changes shape, or bleeds."
    },
    "Melanoma": {
        "description": "A highly serious form of skin cancer that begins in pigment cells (melanocytes).",
        "recommendation": "CRITICAL: Schedule an urgent medical review with a professional dermatologist for a physical biopsy."
    },
    "Squamous cell carcinoma": {
        "description": "A common type of skin cancer originating in the thin, flat squamous cells on the skin surface.",
        "recommendation": "Requires professional medical excision, minor surgery, or targeted treatments by clinical experts."
    },
    "Tinea Ringworm Candidiasis": {
        "description": "Superficial fungal infections producing circular itchy scales or patches.",
        "recommendation": "Keep the skin area clean and completely dry. Apply over-the-counter or prescribed antifungal treatments."
    },
    "Vascular lesion": {
        "description": "Benign clusters of abnormal blood vessels beneath the skin surface, like cherry angiomas.",
        "recommendation": "Usually harmless. Consult a professional specialist if bleeding occurs easily or growth spikes suddenly."
    }
}

# ─── LOAD MODEL ───
def load_model():
    logger.info("Loading model — trying config + weights first")

    # Strategy 1: Load config.json + model_weights.h5 (preferred, Keras 3 multi-file format)
    try:
        with open(CONFIG_PATH, "r") as f:
            config_data = json.load(f)
        m = tf.keras.models.model_from_json(json.dumps(config_data))
        m.load_weights(WEIGHTS_PATH)
        logger.info("Model loaded successfully (config + weights)")
        return m
    except Exception:
        logger.warning("Config+weights load failed", exc_info=True)

    # Strategy 2: Legacy single-file .h5 (kept default-safe: no safe_mode=False)
    try:
        if os.path.exists(LEGACY_H5_PATH):
            logger.info("Trying legacy .h5 load from %s", LEGACY_H5_PATH)
            m = tf.keras.models.load_model(LEGACY_H5_PATH, compile=False)
            logger.info("Model loaded successfully (.h5 file)")
            return m
    except Exception:
        logger.warning(".h5 load failed", exc_info=True)

    logger.error("All loading strategies failed.")
    return None


model = load_model()
if model is None:
    logger.error("Model could not be loaded. /predict will return 500 until this is fixed.")


# ─── ROUTE WEB PAGES ───
@app.route("/")
def serve_home():
    return app.send_static_file("index.html")


# ─── MAIN PREDICTION PIPELINE ───
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"detail": "AI model is offline."}), 500
    if "image" not in request.files:
        return jsonify({"detail": "No image found in request payload."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"detail": "Empty file uploaded."}), 400

    # Server-side content-type allow-list (client-side checks are not a security boundary)
    if file.mimetype not in ALLOWED_CONTENT_TYPES:
        return jsonify({"detail": "Unsupported file type."}), 400

    try:
        img_data = file.read()
        img = Image.open(io.BytesIO(img_data))
        img.verify()  # raises if the file isn't a valid image
        img = Image.open(io.BytesIO(img_data)).convert("RGB")  # re-open: verify() invalidates the handle
        img = img.resize(IMAGE_SIZE)

        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = np.expand_dims(img_array, axis=0)

        raw_predictions = model.predict(img_tensor)[0]
        max_idx = int(np.argmax(raw_predictions))

        predicted_class = CLASS_NAMES[max_idx]
        confidence_score = float(raw_predictions[max_idx]) * 100

        probabilities = {
            name: round(float(raw_predictions[idx]) * 100, 2)
            for idx, name in enumerate(CLASS_NAMES)
        }

        info = CLASS_INFO.get(predicted_class, {
            "description": "No data profile matches this condition.",
            "recommendation": "Seek normal medical consultation advice."
        })

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": round(confidence_score, 2),
            "probabilities": probabilities,
            "description": info["description"],
            "recommendation": info["recommendation"]
        })

    except Exception:
        # Fixed: log full details server-side, never leak them to the client
        logger.exception("Inference engine crash")
        return jsonify({"detail": "Server processing error."}), 500


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    port = int(os.environ.get("PORT", 5000))
    # Fixed: debug defaults OFF. Never run with debug=True on a 0.0.0.0 bind.
    app.run(host=os.environ.get("HOST", "0.0.0.0"), port=port, debug=debug_mode)
