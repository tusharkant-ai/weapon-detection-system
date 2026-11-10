# 🔴 YOLOv8 Weapon Detection System

A real-time weapon detection system built using **YOLOv8 + Streamlit**, capable of detecting multiple weapon classes from images, videos, and live camera feed with alert mechanisms.

---

## 🚀 Features

- 🔍 Detects weapons in **images, videos, and live camera**
- ⚡ Real-time detection using YOLOv8
- 🔔 Alert system:
  - Visual popup
  - Audio alert
- 🎯 Displays **class + confidence score**
- 📊 Includes evaluation metrics & confusion matrix
- 🖥️ Interactive Streamlit dashboard

---

## 🧠 Classes Detected

- 🔫 Pistol  
- 🔫 Rifle  
- 🔪 Knife  
- 🗡️ Sword  
- 🧨 Grenade  
- 🚀 Missile  
- 🪓 Bat  
- 🧍 Person  

---

## 📊 Dataset

- 📦 Custom dataset used for training
- 🖼️ ~2000 images per class
- 🧠 Total classes: 8
- 📁 Annotated in YOLO format

👉 **Download Dataset:**  
https://drive.google.com/drive/folders/18UI94BF2PfJQHCrRmCj5T3fxCmrYFZuK?usp=drive_link

---

## 📊 Model Training Info

- Model: **YOLOv8**
- Training environment: Google Colab
- Total epochs attempted: **195**
- Effective training: **30–35 epochs** (limited by Colab credits)

---

## 📂 Project Structure

```bash
weapon-detection-system/
├── app.py                  # Streamlit app
├── runs/best.pt            # Trained YOLOv8 model
├── Prediction/             # Sample predictions
├── results/                # Metrics & confusion matrix
├── notebooks/              # Training notebook
├── docs/                   # Flowchart
├── alert_sound.mp3         # Alert audio
└── requirements.txt        # Dependencies
````

---

## ⚡ Installation

```bash
git clone https://github.com/sandip234-ui/weapon-detection-system.git
cd weapon-detection-system
pip install -r requirements.txt
```

---

## ▶️ Run App

```bash
streamlit run app.py
```

👉 Make sure model exists at:

```bash
runs/weights/best.pt
```

---

## 📊 Results

* Confusion Matrix
* Precision / Recall Curves
* F1 Score Analysis

📁 Check `/results` folder

---

## 🔔 Alert System

* 🚨 Real-time visual alert
* 🔊 Audio alert (alert_sound.mp3)
* 🕒 Timestamped detection info

---

## 🎥 Demo

Sample outputs available in:

```
Prediction/images/
```

---

## 🔮 Future Improvements

* 📧 Email alerts
* 🧾 Logging system
* ☁️ Cloud deployment
* 📡 Multi-camera support

---

## 👨‍💻 Author

**Sandip Biswal**

* GitHub: [https://github.com/sandip234-ui](https://github.com/sandip234-ui)
* LinkedIn: [https://www.linkedin.com/in/sandip-biswal-728a7a291/](https://www.linkedin.com/in/sandip-biswal-728a7a291/)

---

## 📜 License

For educational and research purposes.

---
