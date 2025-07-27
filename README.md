# 🔥 Real-Time Fire Detection & Alert System

A real-time fire and smoke detection system using a custom YOLOv5 model. Built with OpenCV and PyTorch, it monitors live camera feeds to detect fire or smoke and instantly sends email alerts and automated phone calls using Twilio. Designed for early fire detection and emergency response.

---

## 🚀 Features

- 🔍 Real-time fire & smoke detection via webcam or mobile camera (DroidCam)
- 📷 Captures and saves the frame when fire is detected
- 📧 Sends an **email alert** with the detected image
- 📞 Makes an **automated phone call** using **Twilio API**
- ⚡ Continues running even after alerts (no crash)
- 🎯 Custom confidence threshold to reduce false positives
- 💻 Built to run smoothly on **Windows with GPU support**

---

## 🧠 Tech Stack

- **YOLOv5 + Custom Modules (YOLO-HF)**
- **PyTorch**
- **OpenCV**
- **Python**
- **Twilio API (for calls)**
- **smtplib (for email alerts)**

---

## 🗂 Dataset

- Fire and Smoke dataset with **6,500+ JPG images**
- Split into `train`, `val`, and `test` sets
- Augmented and preprocessed for better generalization

---

## ⚙️ Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yaswanth0068/FireWatchAI.git
   cd FireWatchAI
````

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Your Settings**

   * **Email Setup:** Edit your email, password, and receiver email in `alert_email.py`
   * **Twilio Setup:** Add your Twilio SID, Auth Token, and phone numbers in `alert_call.py`

4. **Run the Detection System**

   ```bash
   python detect_fire.py
   ```

---

## 📦 File Structure

```bash
├── detect_fire.py          # Main script for live detection
├── alert_email.py          # Sends email with image attachment
├── alert_call.py           # Makes phone call using Twilio
├── firesmoke_model.pt      # Trained YOLOv5 model
├── requirements.txt        # Python dependencies
├── /images                 # Sample fire detection results
└── /dataset                # (Optional) Training/validation images
```

---

## 📸 Sample Output

<img src="images/sample_detection.jpg" width="500"/>

---

## 🔐 Security Notice

Make sure to:

* Use a dummy email or app password for alerts
* Don’t push your Twilio credentials or passwords to GitHub

---

## 👨‍💻 Developed By

**Gunda Yaswanth**
B.Tech – Computer Science & Engineering
Narasaraopet Engineering College

[Portfolio](#) | [LinkedIn](https://www.linkedin.com/in/yaswanthgunda0068/) | [Email](mailto:yaswanthgunda12345@gmail.com)

---

## 🏁 Future Improvements

* Add voice assistant for spoken alerts
* Integrate SMS alerts
* Deploy on Raspberry Pi for edge detection
* Build a simple GUI for easier control

---

## ⭐ Give it a Star!

If you like this project, consider giving it a ⭐️ to support and inspire more awesome projects like this!

```
