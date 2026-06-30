# 🩺 Skin Disease Classification System

<img width="1536" height="1024" alt="WhatsApp Image 2026-06-30 at 12 29 14 PM" src="https://github.com/user-attachments/assets/ce77bdee-af2e-440c-b811-bdf111d42315" />


## 📌 About the Project

The **Skin Disease Classification System** is an AI-powered web application that classifies skin lesion images into **nine dermatological conditions** using a **MobileNetV2 Transfer Learning** model.

Users can upload an image and instantly receive:

- 🔍 Predicted skin disease
- 📊 Confidence score
- 📈 Probability distribution
- 📖 Disease description
- 💡 General recommendation

> ⚠ **Disclaimer:** This application is developed for educational and research purposes only and is **not intended for medical diagnosis**.

---
# 📊 Dataset

This project was trained using the publicly available **Skin Disease Classification Image Dataset** published on Kaggle.

**Dataset Information**
- Dataset Name: Skin Disease Classification [Image Dataset]
- Total Images: 900
- Number of Classes: 9
- Train/Validation Split: 80/20
- Source: Kaggle

### Dataset Classes

- Actinic keratosis
- Atopic Dermatitis
- Benign keratosis
- Dermatofibroma
- Melanocytic nevus
- Melanoma
- Squamous cell carcinoma
- Tinea Ringworm Candidiasis
- Vascular lesion

### Dataset Source

🔗 Dataset URL:

https://www.kaggle.com/datasets/riyaelizashaju/skin-disease-classification-image-dataset

### Citation


> Riya Eliza Shaju, *Skin Disease Classification Image Dataset*, Kaggle.
# 🚀 Demo

### Home Page
<img width="1600" height="734" alt="WhatsApp Image 2026-06-30 at 11 28 45 AM" src="https://github.com/user-attachments/assets/1dd6a80b-4310-433b-a099-93f869a8a510" />

### Upload Screen
<img width="1537" height="882" alt="WhatsApp Image 2026-06-30 at 11 29 53 AM" src="https://github.com/user-attachments/assets/eb04f4d7-038d-4f0c-af13-f4b1fdbd9972" />


### Prediction Result
<img width="1121" height="708" alt="WhatsApp Image 2026-06-30 at 11 30 45 AM" src="https://github.com/user-attachments/assets/0b23dbd1-0ae3-4c58-8ee4-9038c39ebcd0" />


# ✨ Features

- 🧠 Deep Learning Image Classification
- ⚡ MobileNetV2 Transfer Learning
- 🌐 Flask REST API
- 🎨 Responsive User Interface
- 🌙 Light & Dark Mode
- 📈 Confidence Score
- 📊 Probability Breakdown
- 📂 Image Upload Validation
- ♿ Accessible Design

---

# 🧬 Supported Skin Diseases

| Disease | Category |
|----------|----------|
| Actinic Keratosis | Pre-cancerous |
| Atopic Dermatitis | Inflammatory |
| Benign Keratosis | Benign |
| Dermatofibroma | Benign |
| Melanocytic Nevus | Benign |
| Melanoma | Malignant |
| Squamous Cell Carcinoma | Malignant |
| Tinea Ringworm Candidiasis | Fungal |
| Vascular Lesion | Vascular |

---

# 🧠 Model Architecture

```
Input Image (224×224×3)
        │
        ▼
MobileNetV2
(ImageNet Pretrained)
        │
        ▼
GlobalAveragePooling2D
        │
        ▼
Dropout (0.3)
        │
        ▼
Dense (128, ReLU)
        │
        ▼
Dense (9, Softmax)
```

---

# 🛠 Tech Stack

| Category | Technology |
|----------|------------|
| Programming Language | Python 3.12 |
| Framework | Flask |
| Deep Learning | TensorFlow / Keras |
| Model | MobileNetV2 |
| Frontend | HTML, CSS, JavaScript |
| Image Processing | Pillow |
| Numerical Computing | NumPy |

---

# 📂 Project Structure

```text
Skin-Disease-Classification-System/

│── app.py
│── requirements.txt
│── README.md
│── LICENSE
│── .gitignore

├── model/
│   ├── config.json
│   ├── metadata.json
│   ├── model.weights.h5
│   └── class_names.json

├── static/
│   ├── index.html
│   ├── style.css
│   └── script.js

├── docs/
│   └── AUDIT_REPORT.md

└── screenshots/
```

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/skin-disease-classification-system.git
```

Move into the project directory

```bash
cd skin-disease-classification-system
```

Create a virtual environment

```bash
python -m venv venv
```

Activate it

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

---

# 🌐 API Endpoint

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Home Page |
| POST | `/predict` | Predict Skin Disease |

---

# 📊 Example Response

```json
{
  "predicted_class": "Melanocytic Nevus",
  "confidence": 91.42,
  "description": "A benign skin mole.",
  "recommendation": "Regular skin monitoring is advised."
}
```

---

# 🎯 Future Improvements

- Fine-tune MobileNetV2
- Deploy on Render/AWS
- Add Grad-CAM Visualization
- User Authentication
- Medical Report Generation
- Mobile Application

---

# 📜 License

This project is licensed under the **MIT License**.

---

# 👩‍💻 Author

**Fatima Hussain**

BS Artificial Intelligence Student

⭐ If you found this project helpful, don't forget to **Star** the repository!
