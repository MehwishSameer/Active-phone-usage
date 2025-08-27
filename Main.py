import cv2
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from ultralytics import YOLO
from moviepy import VideoFileClip
from PIL import Image
import mediapipe as mp

# ==========================
# 1. Load YOLOv8 model
# ==========================
yolo_model = YOLO("yolov8x6.pt")

# ==========================
# 2. Load MobileNet Ensemble
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

num_classes = 2  # phone, pos_machine

# List of trained fold models
model_paths = [
    "mobilenet_pos_vs_phone_fold1.pth",
    "mobilenet_pos_vs_phone_fold2.pth",
    "mobilenet_pos_vs_phone_fold3.pth",
    "mobilenet_pos_vs_phone_fold4.pth",
    "mobilenet_pos_vs_phone_fold5.pth",
]

models_ensemble = []
for path in model_paths:
    model_m = models.mobilenet_v2(weights=None)
    model_m.classifier[1] = nn.Linear(model_m.last_channel, num_classes)
    model_m.load_state_dict(torch.load(path, map_location=device))
    model_m = model_m.to(device)
    model_m.eval()
    models_ensemble.append(model_m)

# ==========================
# 3. Parameters
# ==========================
input_dir = "video_dataset"
output_dir = "output_videos"
os.makedirs(output_dir, exist_ok=True)

CONF_THRESH = 0.34

# ==========================
# 4. MediaPipe Hands
# ==========================
mp_hands = mp.solutions.hands
hand_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=4,
    min_detection_confidence=0.10,
    min_tracking_confidence=0.10
)

def get_hand_boxes(frame):
    """Return bounding boxes of hands from MediaPipe"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand_detector.process(rgb)
    boxes = []

    if results.multi_hand_landmarks:
        h, w, _ = frame.shape
        for hand_landmarks in results.multi_hand_landmarks:
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)
            boxes.append((x_min, y_min, x_max, y_max))
    return boxes

def overlaps(boxA, boxB):
    """Check if two boxes overlap"""
    x1A, y1A, x2A, y2A = boxA
    x1B, y1B, x2B, y2B = boxB

    inter_x1 = max(x1A, x1B)
    inter_y1 = max(y1A, y1B)
    inter_x2 = min(x2A, x2B)
    inter_y2 = min(y2A, y2B)

    return inter_x1 < inter_x2 and inter_y1 < inter_y2

def adjust_conf_with_hands(phone_box, conf, hand_boxes):
    """Boost or penalize confidence based on overlap with hands"""
    if any(overlaps(phone_box, hb) for hb in hand_boxes):
        conf = min(conf * 1.5, 1.0)  # boost
    else:
        conf *= 0.5                  # penalize
    return conf

def classify_phone_or_pos(frame, box):
    """Crop YOLO detection and classify with MobileNet ensemble"""
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return "unknown"

    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    img_t = transform(crop_pil).unsqueeze(0).to(device)

    # Ensemble prediction (average softmax probs)
    probs_sum = torch.zeros((1, num_classes), device=device)
    with torch.no_grad():
        for model_m in models_ensemble:
            outputs = model_m(img_t)
            probs = torch.softmax(outputs, dim=1)
            probs_sum += probs
    probs_avg = probs_sum / len(models_ensemble)

    pred = torch.argmax(probs_avg, dim=1).item()
    return "phone" if pred == 0 else "pos_machine"

# ==========================
# 5. Main loop
# ==========================
for file in os.listdir(input_dir):
    if not file.lower().endswith((".mp4", ".avi", ".mov")):
        continue

    input_path = os.path.join(input_dir, file)
    output_path = os.path.join(output_dir, f"det_{file}")

    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_out = output_path.replace(".mp4", "_temp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_out, fourcc, fps, (width, height))

    phone_timestamps = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame, conf=CONF_THRESH, iou=0.3, imgsz=960, verbose=False)

        phone_boxes = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                name = yolo_model.names[cls]

                if name == "cell phone":
                    phone_boxes.append((list(map(int, box.xyxy[0].tolist())), conf))

        # MediaPipe hand detection
        hand_boxes = get_hand_boxes(frame)

        phone_candidates = []
        for (phone_box, conf) in phone_boxes:
            # Adjust confidence using hand overlap
            conf = adjust_conf_with_hands(phone_box, conf, hand_boxes)

            if conf >= CONF_THRESH:
                cls_label = classify_phone_or_pos(frame, phone_box)
                if cls_label == "phone":
                    phone_candidates.append((conf, phone_box))

        # Draw highest-confidence phone
        if phone_candidates:
            best_conf, best_box = max(phone_candidates, key=lambda x: x[0])
            x1, y1, x2, y2 = best_box
            label = f"PHONE {best_conf:.2f}"
            color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            timestamp = frame_count / fps
            if not phone_timestamps or (timestamp - phone_timestamps[-1]) >= 1.0:
                phone_timestamps.append(timestamp)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    # Preserve audio
    original = VideoFileClip(input_path)
    processed = VideoFileClip(temp_out)
    final = processed.with_audio(original.audio)
    final.write_videofile(output_path, codec="libx264", audio_codec="aac")

    os.remove(temp_out)

    # Log detections
    log_path = os.path.join(output_dir, f"{file}_log.txt")
    with open(log_path, "w") as f:
        f.write("Device detected (confidence adjusted with hands) at timestamps (s):\n")
        for t in phone_timestamps:
            f.write(f"{t:.2f}\n")
        f.write(f"\nTotal detections: {len(phone_timestamps)}\n")

    print(f"Processed {file} → {output_path}")
