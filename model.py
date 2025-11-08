# ============================================================
# 🧠 CrowdTrackAI - Model Training & Testing Script (Final)
# ============================================================

# ✅ Run once before training:
# pip install ultralytics opencv-python

from ultralytics import YOLO
import os
import cv2

# ============================================================
# 1️⃣ DATASET PATH
# ============================================================
dataset_path = r"C:\Users\userp\OneDrive\Desktop\CrowdTrackAI\dataset"

if os.path.exists(dataset_path):
    print("✅ Dataset folder found at:", dataset_path)
else:
    print("⚠️ Dataset folder not found! Please check the path.")
    exit()

# ============================================================
# 2️⃣ CREATE data.yaml FILE (with absolute paths)
# ============================================================
yaml_content = f"""train: {dataset_path}\\images\\train
val: {dataset_path}\\images\\val

nc: 1
names: ['person']
"""

yaml_path = os.path.join(dataset_path, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(yaml_content)

print("✅ data.yaml file created at:", yaml_path)

# ============================================================
# 3️⃣ TRAIN MODEL
# ============================================================
print("\n🚀 Starting YOLOv8 training...")
model = YOLO("yolov8n.pt")  # lightweight model (fast training)

model.train(
    data=yaml_path,
    epochs=50,        # number of training rounds
    imgsz=640,        # image size
    batch=8,          # adjust for your CPU/GPU
    name="crowdtrack_model"
)

print("✅ Training complete!")
print("📁 Model saved in: runs/detect/crowdtrack_model/weights/best.pt")

# ============================================================
# 4️⃣ VALIDATE MODEL (Precision, Recall, Accuracy)
# ============================================================
print("\n🧪 Validating model...")
metrics = model.val()

# Extract main metrics
precision = metrics.results_dict.get('metrics/precision(B)', None)
recall = metrics.results_dict.get('metrics/recall(B)', None)
mAP50 = metrics.results_dict.get('metrics/mAP50(B)', None)
mAP50_95 = metrics.results_dict.get('metrics/mAP50-95(B)', None)

# Calculate simple accuracy (approx)
if precision is not None and recall is not None:
    accuracy = (2 * precision * recall) / (precision + recall)  # F1-based approximation
else:
    accuracy = None

print("\n✅ Validation Metrics:")
print(f"📏 Precision: {precision:.4f}" if precision else "📏 Precision: N/A")
print(f"🎯 Recall: {recall:.4f}" if recall else "🎯 Recall: N/A")
print(f"📊 mAP@50: {mAP50:.4f}" if mAP50 else "📊 mAP@50: N/A")
print(f"📈 mAP@50-95: {mAP50_95:.4f}" if mAP50_95 else "📈 mAP@50-95: N/A")
print(f"✅ Approx. Accuracy (F1-based): {accuracy:.4f}" if accuracy else "✅ Accuracy: N/A")

# ============================================================
# 5️⃣ TEST MODEL ON ONE IMAGE
# ============================================================
test_image = rf"{dataset_path}\images\val\sample.jpg"

if os.path.exists(test_image):
    print("\n🔍 Testing model on:", test_image)
    results = model(test_image)
    results.show()
else:
    print("\n⚠️ No test image found. Add one in your val folder to test manually.")

print("\n✅ CrowdTrackAI model training, validation & testing complete.")
# ============================================================
