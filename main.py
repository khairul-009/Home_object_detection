import shutil
import random
from pathlib import Path
from ultralytics import YOLO

# ===================== CONFIG =====================
RAW_DATASET = "row_data"
OUTPUT_DATASET = "dataset_yolo"
TRAIN_RATIO = 0.8
IMG_EXTS = [".jpg", ".jpeg", ".png"]

MODEL_NAME = "yolov8n.pt"   # yolov8n/s/m/l/x
EPOCHS = 120
IMGSZ = 640
BATCH = 16
DEVICE = "cpu"                # 0 = GPU, "cpu" = CPU
# =================================================

# ---------- PATHS ----------
images_dir = Path(RAW_DATASET) / "images"
labels_dir = Path(RAW_DATASET) / "labels"
classes_file = Path(RAW_DATASET) / "classes.txt"

out_img_train = Path(OUTPUT_DATASET) / "images/train"
out_img_val = Path(OUTPUT_DATASET) / "images/val"
out_lbl_train = Path(OUTPUT_DATASET) / "labels/train"
out_lbl_val = Path(OUTPUT_DATASET) / "labels/val"

# ---------- CREATE FOLDERS ----------
for p in [out_img_train, out_img_val, out_lbl_train, out_lbl_val]:
    p.mkdir(parents=True, exist_ok=True)

# ---------- LOAD CLASSES ----------
with open(classes_file, "r") as f:
    class_names = [c.strip() for c in f.readlines() if c.strip()]

print(f"✔ Classes: {class_names}")

# ---------- COLLECT IMAGES ----------
images = [img for img in images_dir.iterdir() if img.suffix.lower() in IMG_EXTS]
random.shuffle(images)

split_idx = int(len(images) * TRAIN_RATIO)
train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

# ---------- COPY FILES ----------
def copy_data(img_list, img_out, lbl_out):
    for img in img_list:
        lbl = labels_dir / f"{img.stem}.txt"
        if not lbl.exists():
            print(f"⚠ Skipping (no label): {img.name}")
            continue
        shutil.copy(img, img_out / img.name)
        shutil.copy(lbl, lbl_out / lbl.name)

copy_data(train_imgs, out_img_train, out_lbl_train)
copy_data(val_imgs, out_img_val, out_lbl_val)

# ---------- CREATE data.yaml ----------
data_yaml_content = f"""
path: {Path(OUTPUT_DATASET).absolute()}
train: images/train
val: images/val

nc: {len(class_names)}
names: {class_names}
"""

data_yaml_path = Path(OUTPUT_DATASET) / "data.yaml"
with open(data_yaml_path, "w") as f:
    f.write(data_yaml_content.strip())

print("✅ Dataset preparation complete")

# ===================== TRAIN YOLO =====================
print("🚀 Starting training...")

model = YOLO(MODEL_NAME)

model.train(
    data=str(data_yaml_path),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    device=DEVICE,
    project="runs/train",
    name="custom_yolo"
)

print("🎉 Training finished successfully!")
