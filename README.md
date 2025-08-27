# Active Phone Usage Detection

This project detects **active phone usage vs POS device usage** using deep learning and computer vision techniques.  
It combines **YOLOv8 object detection** with a **K-Fold trained MobileNet classifier** and **MediaPipe hand tracking** to achieve robust classification.

---

## 📂 Dataset
- **POS samples**: 58 images  
- **Cell Phone samples**: 50 images  
- Dataset is **balanced** with slight class-weight penalty adjustments during training.  

---

## 🏋️ Training
- **Model**: MobileNet (pretrained on ImageNet, fine-tuned for binary classification)  
- **Cross-validation**: 5-fold (ensures robust evaluation on small dataset)  
- **Loss penalty**:  
  - Extra penalty when misclassifying **POS** (higher priority, must always classify correctly).  
  - Penalty for "no hand overlap" with phone/pos regions (ensures object is actively held).  
- **Outputs**:  
  - Each fold saves its own model separately (`model_fold1.pth`, `model_fold2.pth`, …).  
  - Best validation accuracy per fold is stored.  

---

## ✋ MediaPipe Hand Adjustment
MediaPipe Hands is integrated to check:  
- **Overlap between detected hands and YOLOv8 bounding box (phone/POS)**.  
- **Both hands threshold**:
  - At least one hand must overlap the detected phone/POS.  
  - If both overlap → higher confidence.  
- **No overlap penalty**: reduces confidence score significantly.  

---

## 🧠 Inference (Main Code)
The **main detection pipeline** works as follows:  
1. **YOLOv8** detects candidate objects (phone or POS).  
2. Each bounding box is passed to **all 5 trained MobileNet models**.  
3. Predictions are averaged → **final classification decision**.  
4. MediaPipe hands check → if hands overlap detected object.  
   - If no hands overlap → penalize probability.  
5. Apply threshold:  
   - **Both-hands confidence threshold** for stricter validation.  
   - **Overall phone threshold** (YOLOv8 detection + classifier output).  

---

🛠️ Tech Stack

- PyTorch (MobileNet training)
- Ultralytics YOLOv8 (object detection)
- MediaPipe (hand tracking)
- OpenCV (video processing)
