import cv2
import torch
import numpy as np
import sys
import time
from threading import Thread
from twilio.rest import Client
from email.message import EmailMessage
import smtplib

# YOLOv5 path
sys.path.append('./yolov5')
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords

# 🔥 Model setup
device = torch.device('cpu')
model = DetectMultiBackend('firesmoke_model_cleaned.pt', device=device)
stride, names = model.stride, model.names
model.eval()

# 📧 Email setup
EMAIL_ADDRESS = "{From Mail Id Goes Here}"
EMAIL_PASSWORD = "{From Mail Password Goes Here}"
TO_EMAIL = "{To Mail Id Goes Here}"

# 📞 Twilio setup
account_sid = '{Twilio Account SID Goes Here}'
auth_token = '{Twilio Auth Token Goes Here}'
twilio_to = "{To Mobile Number Goes Here}"
twilio_from = "{From Mobile Number Goes Here}"
flow_sid = "{Twilio Account flow SID Goes Here}"

def send_email_alert(image_path):
    try:
        msg = EmailMessage()
        msg['Subject'] = "🔥 FIRE DETECTED!"
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = TO_EMAIL
        msg.set_content('🚨 Fire Alert\n\nFire detected. Image attached.')
        with open(image_path, "rb") as f:
            msg.add_attachment(f.read(), maintype='image', subtype='jpeg', filename="fire.jpg")
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
            print("✅ Email sent!")
    except Exception as e:
        print("❌ Email Error:", e)

def trigger_twilio_call():
    try:
        client = Client(account_sid, auth_token)
        execution = client.studio.v2.flows(flow_sid).executions.create(
            to=twilio_to,
            from_=twilio_from
        )
        print(f"✅ Call triggered — SID: {execution.sid}")
    except Exception as e:
        print("❌ Twilio Error:", e)

def send_alerts(image):
    cv2.imwrite("output.jpg", image)
    Thread(target=send_email_alert, args=("output.jpg",)).start()
    Thread(target=trigger_twilio_call).start()

# 📷 Camera settings
USE_DROIDCAM = False

if USE_DROIDCAM:
    cap = cv2.VideoCapture("http://10.213.56.83:4747/video")
else:
    cap = cv2.VideoCapture(0)


# 🔁 Fire tracking and settings
fire_frames = 0
fire_threshold = 5 # No.Of Frames
alert_cooldown = 10
last_alert_time = 0
fire_detection_count = 0
CONFIDENCE_THRESHOLD = 0.6
alert_triggered = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera read failed")
        break

    original = frame.copy()
    image = cv2.resize(original, (640, 640))
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).to(device).permute(2, 0, 1).float().div(255.0).unsqueeze(0)

    pred = model(img_tensor)
    pred = non_max_suppression(pred, conf_thres=CONFIDENCE_THRESHOLD, iou_thres=0.45)

    fire_detected = False
    det = pred[0]
    if det is not None and len(det):
        det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], original.shape).round()
        for *xyxy, conf, cls in det:
            cls_id = int(cls.item())
            if cls_id == 0:
                fire_detected = True
                xyxy = [int(x.item()) for x in xyxy]
                label = f'{names[cls_id]} {conf:.2f}'
                cv2.rectangle(original, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 2)
                cv2.putText(original, label, (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if fire_detected:
        fire_frames += 1
        if fire_frames >= fire_threshold and not alert_triggered:
            now = time.time()
            if now - last_alert_time >= alert_cooldown:
                fire_detection_count += 1
                send_alerts(original)
                last_alert_time = now
                alert_triggered = True
    else:
        fire_frames = 0
        alert_triggered = False

    # 🖼️ UI panel
    panel = np.zeros((original.shape[0], 300, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)
    cv2.putText(panel, "🔥 Fire Detection Panel", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(panel, f"Detected: {fire_detected}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
    cv2.putText(panel, f"Seen for: {fire_frames / 12:.1f}s", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 100), 2)
    cv2.putText(panel, f"Alerts Sent: {fire_detection_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 180), 2)

    # 📺 Final Output Frame
    resized_frame = cv2.resize(original, (640, 480))
    resized_panel = cv2.resize(panel, (300, 480))
    combined = np.hstack((resized_frame, resized_panel))

    cv2.imshow("🚨 Fire Monitoring Dashboard", combined)

    key = cv2.waitKey(80) & 0xFF
    if key == ord('q') or key == 27:
        print("🛑 Exit requested")
        break

cap.release()
cv2.destroyAllWindows()
