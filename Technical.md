# 🔧 TECHNICAL DOCUMENTATION
## Traffic Violation Detection & E-Challan Generation System

> Complete technical reference — architecture, datasets, training, all features, code explanation, API, troubleshooting, and full project history with every decision explained.

---

## 📋 Table of Contents

1. [Project History & Evolution](#1-project-history--evolution)
2. [System Architecture](#2-system-architecture)
3. [Environment Setup](#3-environment-setup)
4. [Dataset Download & Preparation](#4-dataset-download--preparation)
5. [Model Training](#5-model-training)
6. [Detection Modules — All 8](#6-detection-modules--all-8)
7. [Tracking Modules](#7-tracking-modules)
8. [Violation Engine — All Rules](#8-violation-engine--all-rules)
9. [Challan Generation](#9-challan-generation)
10. [Database Schema](#10-database-schema)
11. [Flask API Reference](#11-flask-api-reference)
12. [Dashboard Features — All](#12-dashboard-features--all)
13. [Web UI Architecture](#13-web-ui-architecture)
14. [Key Design Decisions](#14-key-design-decisions)
15. [Known Issues & Fixes](#15-known-issues--fixes)
16. [Performance Benchmarks](#16-performance-benchmarks)
17. [Future Scope](#17-future-scope)

---

## 1. Project History & Evolution

### 1.1 Project Start

The project began with the goal of building an AI-powered traffic violation detection system similar to real-world CCTV-based enforcement systems used in Indian cities. The original plan included live CCTV camera integration, but this was scoped down to image and video file upload for practical development on a laptop.

**Key insight:** `cv2.VideoCapture()` treats both live cameras and video files identically — making the transition to live cameras trivial in the future.

---

### 1.2 Phase 1 — Planning & Tech Selection

**Violations planned:**
- Helmet detection
- Seatbelt detection
- Number plate recognition
- Speed detection
- Red light violation
- Wrong way detection
- No parking
- Triple riding

**Tech selected:**
- YOLOv8 — state-of-art object detection, free, fast
- EasyOCR — best OCR for Indian number plates
- Flask — lightweight web framework
- SQLite — zero-config database
- ReportLab — PDF generation

---

### 1.3 Phase 2 — Dataset Collection from Roboflow

All datasets downloaded from [Roboflow Universe](https://universe.roboflow.com) in YOLOv8 format.

**Helmet Dataset:**
- Link: https://universe.roboflow.com/helmet-detection-nmzik/helmet-detection-nsbwm/dataset/18/download/yolov8
- 4,607 training images
- Classes: Helmet, NHelmet, Motorbike, PNumber
- Problem: No `valid/` folder in download
- Fix: Created valid split using Python (20% of train)

**Seatbelt Dataset:**
- Link: https://universe.roboflow.com/noriel-vlmtq/seatbelt-axfll/dataset/1
- 2,080 images, class: seatbelt
- Had train/valid/test splits

**Number Plate Dataset:**
- Link: https://universe.roboflow.com/testing-for-pothole-detection/license-plate-detection-agqth/dataset/3/download
- 2,130 images, classes: plate, licence
- Had train/valid/test splits

**data.yaml fix applied to all:**
```yaml
# Before (broken — relative paths)
train: ../train/images

# After (fixed — absolute paths)
train: D:/traffic_violation_system/datasets/helmet/train/images
val: D:/traffic_violation_system/datasets/helmet/valid/images
```

**Valid folder creation script (helmet dataset):**
```python
import os, shutil, random
train_img = r'D:\traffic_violation_system\datasets\helmet\train\images'
train_lbl = r'D:\traffic_violation_system\datasets\helmet\train\labels'
valid_img = r'D:\traffic_violation_system\datasets\helmet\valid\images'
valid_lbl = r'D:\traffic_violation_system\datasets\helmet\valid\labels'
os.makedirs(valid_img, exist_ok=True)
os.makedirs(valid_lbl, exist_ok=True)
files = os.listdir(train_img)
random.shuffle(files)
split = int(len(files) * 0.2)
for f in files[:split]:
    shutil.copy(os.path.join(train_img, f), os.path.join(valid_img, f))
    label = f.replace('.jpg','.txt').replace('.png','.txt')
    if os.path.exists(os.path.join(train_lbl, label)):
        shutil.copy(os.path.join(train_lbl, label), os.path.join(valid_lbl, label))
```

---

### 1.4 Phase 3 — Model Training

**Critical Windows Bug Encountered:**

```
RuntimeError: An attempt has been made to start a new process before
the current process has finished its bootstrapping phase.
```

**Cause:** Windows requires `if __name__ == '__main__':` guard for multiprocessing. YOLOv8 training uses multiprocessing internally.

**Fix:**
```python
# WRONG — crashes on Windows
model.train(data='data.yaml', epochs=50)

# CORRECT — Windows compatible
if __name__ == '__main__':
    model.train(data='data.yaml', epochs=50, workers=0)
```

`workers=0` disables all multiprocessing — most stable on Windows.

**Training Results:**
```
Helmet Model:
  Epochs: 50 (early stopped at 33)
  mAP50: 91.9%
  Best model: models/helmet_model3/weights/best.pt

Seatbelt Model:
  Epochs: 50 (completed)
  Best model: models/seatbelt_model/weights/best.pt

Number Plate Model:
  Epochs: 50 (early stopped at 23)
  mAP50: 89.6% overall, 97.3% for plate class
  Best model: models/numberplate_model/weights/best.pt
```

---

### 1.5 Phase 4 — First Detection Script

`test_yolo.py` — basic test using pre-trained YOLOv8n on downloaded traffic video.

**Result:** Vehicles detected successfully with bounding boxes and confidence scores.

**Observation:** Pre-trained model already detects: car, motorcycle, bus, truck, person, traffic light — no training needed for these.

---

### 1.6 Phase 5 — UI Framework Selection (3 attempts)

#### Attempt 1: Streamlit
**Problem:** Basic UI, limited video support, tab switching broken with newer versions.
**Decision:** Abandoned.

#### Attempt 2: Gradio
**Installed version:** 6.13.0
**Problem:** `gr.Tab` tabs not clickable — version compatibility issue with HuggingFace Hub.
**Error:** `ImportError: cannot import name 'HfFolder' from 'huggingface_hub'`
**Fix attempted:** Downgrade to gradio==4.44.0
**New problem:** Further dependency conflicts with pillow version.
**Decision:** Abandoned Gradio entirely.

#### Attempt 3: Flask + HTML/CSS/JS ✅
**Why Flask won:**
- Zero dependency conflicts
- Full control over tab behavior (pure HTML)
- Native video streaming support
- REST API built-in
- Professional web app appearance
- Works perfectly on all browsers

---

### 1.7 Phase 6 — Frame-Level to Vehicle-Centric Logic

**Problem discovered:** One image with 2 bikes (both no helmet) + 1 car (no seatbelt) was generating only 1 challan instead of 3.

**Root cause:** System was checking violations for the entire frame, not per vehicle.

**Old (wrong) approach:**
```python
# All detections in frame merged
vehicles = detect_vehicles(frame)
helmets  = detect_helmets(frame)
violations = check_helmet(helmets)      # ALL helmets in frame
violations += check_seatbelt(seatbelts) # ALL seatbelts in frame
make_challan(plate, violations)         # Only 1 challan
```

**New (correct) approach:**
```python
# Per vehicle loop
for i, vehicle in enumerate(vehicles):
    # Only nearby detections assigned to THIS vehicle
    near_helmets   = get_helmets_near(helmets, vehicle['bbox'])
    near_seatbelts = get_seatbelts_near(seatbelts, vehicle['bbox'])

    violations = check_vehicle_violations(vehicle, near_helmets, near_seatbelts)
    plate = get_plate_near(plates, vehicle['bbox'])

    # Unique key per vehicle
    key = plate if plate else f"UNKNOWN-{vehicle['class']}-{i}"
    make_challan(key, violations, challan_done)
```

**Proximity matching function:**
```python
def is_inside_or_near(obj_bbox, vehicle_bbox, margin=40):
    ox1,oy1,ox2,oy2 = obj_bbox
    vx1,vy1,vx2,vy2 = vehicle_bbox
    ocx = (ox1+ox2)//2
    ocy = (oy1+oy2)//2
    return (vx1-margin < ocx < vx2+margin and
            vy1-margin < ocy < vy2+margin)
```

---

### 1.8 Phase 7 — Rule Logic Fixes

**Fix 1: Triple Riding Threshold**
```python
# Wrong: 2 people on bike = violation (legal)
if person_count >= 2:

# Correct: 3+ people = violation
if person_count >= 3:
```

**Fix 2: Vehicle Type Filtering**
```python
# Helmet only for motorcycles
if vehicle['class'] == 'motorcycle':
    check_helmet_violation(near_helmets)

# Seatbelt only for cars/trucks/buses
elif vehicle['class'] in ['car', 'truck', 'bus']:
    check_seatbelt_violation(near_seatbelts)
```

**Fix 3: Challan Without Plate**
```python
# Old: skip if no plate
if violations and plate_text:
    make_challan(plate_text, ...)

# New: generate with UNKNOWN key
key = plate_text if plate_text else \
      f"UNKNOWN-{vehicle['class'].upper()}-{i}-{datetime.now().strftime('%H%M%S')}"
make_challan(key, violations, ...)
```

**Fix 4: Duplicate Vehicle Detection**
```python
# Same vehicle detected twice → skip duplicate
processed_veh = set()
vkey = f"{v['bbox'][0]//50}_{v['bbox'][1]//50}"
if vkey in processed_veh:
    continue
processed_veh.add(vkey)
```

---

### 1.9 Phase 8 — Dashboard Evolution

| Version | Change | Reason |
|---|---|---|
| V1 | Basic Streamlit | Starting point |
| V2 | Streamlit with charts | Tabs broken |
| V3 | Gradio 4 tabs | Import errors |
| V4 | Flask basic | Working but plain |
| V5 | Flask + glassmorphism dark | Too dark, unreadable text |
| V6 | Flask + light theme | Clean and readable |
| V7 (Final) | Flask + light + collapsible charts | Professional, compact |

**Dashboard features added iteratively:**
1. Basic challan table
2. Stats cards (total/paid/unpaid)
3. Search + filter
4. Update status modal (paid/waived)
5. Payment modes (Cash/UPI/Online/Cheque)
6. Waived status with remarks
7. Delete challan
8. Export CSV
9. Print button
10. Auto-refresh (30 seconds)
11. Charts (pie + bar) as collapsible FAB panel
12. Notification badge (unpaid count)

---

## 2. System Architecture

### 2.1 Complete Pipeline

```
User (Browser)
    │
    ▼
Flask Web App (flask_app.py)
    │
    ├─── GET /           → Serve index.html
    │
    ├─── POST /detect_image
    │        │
    │        ▼
    │    detect_all(frame)
    │        ├── VehicleDetector   → vehicles[], persons[]
    │        ├── HelmetDetector    → helmets[]
    │        ├── SeatbeltDetector  → seatbelts[]
    │        └── PlateDetector     → plates[]
    │        │
    │        ▼
    │    Per Vehicle Loop
    │        ├── is_inside_or_near() — assign detections to vehicle
    │        ├── check_vehicle_violations()
    │        │       motorcycle  → helmet check + triple riding
    │        │       car/truck   → seatbelt check
    │        ├── get_plate_near()
    │        └── make_challan() → PDF + SQLite
    │        │
    │        ▼
    │    JSON Response → Browser
    │
    ├─── POST /detect_video
    │        │
    │        ▼
    │    Same as image + per frame:
    │        ├── TrafficLightDetector
    │        ├── SpeedTracker (line crossing, FPS adaptive)
    │        ├── WrongWayDetector (position history)
    │        └── NoParkingDetector (zone + time)
    │        │
    │        ▼
    │    Annotated video + JSON response
    │
    └─── Dashboard routes
             ├── GET /dashboard_data
             ├── POST /update_challan
             ├── DELETE /delete_challan/<id>
             ├── GET /search_challans
             ├── GET /export_csv
             └── GET /download_pdf/<id>
```

---

## 3. Environment Setup

### 3.1 System Requirements

```
Python:  3.12.10
GPU:     NVIDIA RTX 3050 (4GB VRAM) — tested
CUDA:    13.2
OS:      Windows 11 (64-bit)
RAM:     16 GB minimum
Storage: ~5 GB free (datasets + models)
```

### 3.2 Step-by-Step Installation

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/Mac

# 2. Fix PowerShell execution policy (Windows only)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 3. Install PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Verify GPU detection
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
# Expected: CUDA: True | GPU: NVIDIA GeForce RTX 3050 Laptop GPU

# 5. Install all other packages
pip install ultralytics==8.4.33
pip install opencv-python==4.13.0.92
pip install easyocr==1.7.2
pip install flask==3.1.3
pip install reportlab==4.4.10
pip install pillow==10.4.0
pip install numpy pandas
```

---

## 4. Dataset Download & Preparation

### 4.1 Helmet Dataset

**Download:** https://universe.roboflow.com/helmet-detection-nmzik/helmet-detection-nsbwm/dataset/18/download/yolov8

```bash
# Extract to:
datasets/helmet/
    ├── train/images/   (4607 images)
    ├── train/labels/   (4607 .txt files)
    ├── valid/images/   (create if missing — see script above)
    ├── valid/labels/
    └── data.yaml
```

**data.yaml (final):**
```yaml
train: D:/traffic_violation_system/datasets/helmet/train/images
val: D:/traffic_violation_system/datasets/helmet/valid/images
test: D:/traffic_violation_system/datasets/helmet/train/images
nc: 4
names: ['Helmet', 'Motorbike', 'NHelmet', 'PNumber']
```

---

### 4.2 Seatbelt Dataset

**Download:** https://universe.roboflow.com/noriel-vlmtq/seatbelt-axfll/dataset/1

```bash
# Extract to:
datasets/seatbelt/
    ├── train/images/   (2080 images)
    ├── valid/images/
    ├── test/images/
    └── data.yaml
```

**data.yaml (final):**
```yaml
train: D:/traffic_violation_system/datasets/seatbelt/train/images
val: D:/traffic_violation_system/datasets/seatbelt/valid/images
test: D:/traffic_violation_system/datasets/seatbelt/test/images
nc: 1
names: ['seatbelt']
```

---

### 4.3 Number Plate Dataset

**Download:** https://universe.roboflow.com/testing-for-pothole-detection/license-plate-detection-agqth/dataset/3/download

```bash
# Extract to:
datasets/number_plate/
    ├── train/images/   (2130 images)
    ├── valid/images/
    ├── test/images/
    └── data.yaml
```

**data.yaml (final):**
```yaml
train: D:/traffic_violation_system/datasets/number_plate/train/images
val: D:/traffic_violation_system/datasets/number_plate/valid/images
test: D:/traffic_violation_system/datasets/number_plate/test/images
nc: 2
names: ['plate', 'licence']
```

---

## 5. Model Training

### 5.1 Helmet Model Training

```python
# train_helmet.py
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    model.train(
        data=r'D:\traffic_violation_system\datasets\helmet\data.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        project=r'D:\traffic_violation_system\models',
        name='helmet_model',
        patience=10,
        workers=0
    )
    print("Helmet training complete!")
```

```bash
python train_helmet.py
# Duration: ~30 minutes on RTX 3050
# Output: models/helmet_model3/weights/best.pt
# Result: mAP50 = 91.9%
```

---

### 5.2 Seatbelt Model Training

```python
# train_seatbelt.py
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    model.train(
        data=r'D:\traffic_violation_system\datasets\seatbelt\data.yaml',
        epochs=50, imgsz=640, batch=8, device=0,
        project=r'D:\traffic_violation_system\models',
        name='seatbelt_model', patience=10, workers=0
    )
```

```bash
python train_seatbelt.py
# Output: models/seatbelt_model/weights/best.pt
```

---

### 5.3 Number Plate Model Training

```python
# train_numberplate.py
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    model.train(
        data=r'D:\traffic_violation_system\datasets\number_plate\data.yaml',
        epochs=50, imgsz=640, batch=8, device=0,
        project=r'D:\traffic_violation_system\models',
        name='numberplate_model', patience=10, workers=0
    )
```

```bash
python train_numberplate.py
# Early stopped at epoch 23
# Output: models/numberplate_model/weights/best.pt
# Result: mAP50 = 89.6% (plate class: 97.3%)
```

---

### 5.4 Training Parameters Explained

| Parameter | Value | Reason |
|---|---|---|
| `model` | yolov8n.pt | Nano — fits 4GB VRAM with 3 models running |
| `epochs` | 50 | Enough with early stopping |
| `imgsz` | 640 | Standard YOLOv8 input size |
| `batch` | 8 | Fits in 4GB VRAM |
| `device` | 0 | CUDA:0 = RTX 3050 |
| `patience` | 10 | Stop if no improvement for 10 epochs |
| `workers` | 0 | **Critical** — fixes Windows crash |
| `pretrained` | True | Transfer learning from COCO |

---

## 6. Detection Modules — All 8

### 6.1 VehicleDetector (`detection/vehicle_detector.py`)

Uses pre-trained YOLOv8n (COCO weights).

```python
# COCO class IDs used:
vehicle_classes = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}
person_class = 0  # person

# Tuned parameters:
conf = 0.35   # Lower = more detections, tested value
iou  = 0.60   # Higher = fewer duplicates
```

**Returns:** `vehicles[], persons[]`

---

### 6.2 HelmetDetector (`detection/helmet_detector.py`)

Custom trained on 4,607 images.

**Classes:**
- `Helmet` → wearing helmet ✅
- `NHelmet` → NOT wearing helmet ❌ → violation
- `Motorbike` → motorcycle region
- `PNumber` → plate region (bonus)

**Violation logic:**
```python
near_helmets = get_helmets_near(helmets, vehicle_bbox, margin=60)
no_helmet_found = any(
    h['class'] == 'NHelmet' and h['confidence'] >= 0.35
    for h in near_helmets
)
if no_helmet_found or len(near_helmets) == 0:
    # violation
```

---

### 6.3 SeatbeltDetector (`detection/seatbelt_detector.py`)

Custom trained on 2,080 images.

**Violation logic:**
```python
near_seatbelts = get_seatbelts_near(seatbelts, vehicle_bbox, margin=60)
if len(near_seatbelts) == 0:
    # No seatbelt detected near car = violation
```

**Note:** Only applied to car/truck/bus, never motorcycle.

---

### 6.4 PlateDetector (`detection/plate_detector.py`)

Two-stage: YOLOv8 detect location → EasyOCR read text.

```python
# Stage 1: Detect plate region
results = self.model(frame, verbose=False)[0]

# Stage 2: Crop and read
plate_img = frame[y1:y2, x1:x2]
ocr_result = self.reader.readtext(plate_img)

# Low confidence threshold for better recall
for (_, text, conf) in ocr_result:
    if conf > 0.2:
        plate_text += text + ' '
```

**EasyOCR initialized with GPU:**
```python
self.reader = easyocr.Reader(['en'], gpu=True)
```

---

### 6.5 TrafficLightDetector (`detection/traffic_light.py`)

COCO class 9 (traffic light) + HSV color analysis.

```python
def detect_color(self, light_img):
    hsv = cv2.cvtColor(light_img, cv2.COLOR_BGR2HSV)
    h, w = light_img.shape[:2]

    # Split into 3 vertical zones
    top = hsv[:h//3, :]   # Red zone
    mid = hsv[h//3:2*h//3, :]  # Yellow zone
    bot = hsv[2*h//3:, :]  # Green zone

    # Red detection (2 HSV ranges — red wraps around 180)
    red1 = cv2.inRange(top, [0,100,100], [10,255,255])
    red2 = cv2.inRange(top, [160,100,100], [180,255,255])

    # Count pixels to determine color
    # Highest pixel count wins
```

---

### 6.6 WrongWayDetector (`detection/wrong_way.py`)

Tracks Y-position history to detect downward movement.

```python
HISTORY_LENGTH = 5

def is_wrong_way(self, vehicle_id):
    positions = self.vehicle_history.get(vehicle_id, [])
    if len(positions) < self.HISTORY_LENGTH:
        return False

    # Count frames where Y increased (moving down = wrong way)
    increasing = sum(
        1 for i in range(1, len(positions))
        if positions[i] > positions[i-1]
    )
    return increasing >= 4  # 4 out of 5 frames
```

**Limitation:** Assumes top-down camera angle with expected flow from top to bottom.

---

### 6.7 NoParkingDetector (`detection/no_parking.py`)

Zone-based with configurable time threshold.

```python
PARKING_TIME_LIMIT = 5.0  # seconds

def update(self, vehicle_id, bbox, current_speed=0):
    in_zone, zone_name = self.is_in_zone(bbox)
    if not in_zone:
        return False, 0

    if vehicle_id not in self.stationary_vehicles:
        self.stationary_vehicles[vehicle_id] = {
            'first_seen': time.time(),
            'zone': zone_name
        }

    elapsed = time.time() - self.stationary_vehicles[vehicle_id]['first_seen']
    return elapsed > self.PARKING_TIME_LIMIT, elapsed
```

**Zone definition in flask_app.py:**
```python
no_parking_detector.add_zone(50, 300, 300, 450, "No Parking Zone 1")
no_parking_detector.add_zone(700, 300, 950, 450, "No Parking Zone 2")
```

---

### 6.8 TripleRidingDetector (`detection/triple_riding.py`)

Person count on motorcycle.

```python
# In check_vehicle_violations():
person_count = get_persons_on_vehicle(persons, vehicle_bbox)
if person_count >= 3:
    # Triple riding violation
    # Fine: Rs. 1000
```

**Note:** 2 persons on bike = legal (double riding allowed in India).

---

## 7. Tracking Modules

### 7.1 SpeedTracker (`tracking/speed_tracker.py`)

**Method:** Virtual line crossing

```python
# Two horizontal lines at configurable heights
LINE1_Y = int(frame_height * 0.30)  # Upper line
LINE2_Y = int(frame_height * 0.65)  # Lower line
REAL_DISTANCE = 10                   # Meters between lines
SPEED_LIMIT = 60                     # km/h

# Algorithm:
# 1. Vehicle crosses LINE1 → record timestamp
# 2. Vehicle crosses LINE2 → calculate speed
elapsed = time_at_line2 - time_at_line1
speed_ms = REAL_DISTANCE / elapsed
speed_kmh = speed_ms * 3.6
```

**FPS-adaptive frame skip:**
```python
# Process every Nth frame based on video FPS
if fps <= 15:    SKIP = 5
elif fps <= 24:  SKIP = 8
else:            SKIP = 12
```

**Why line crossing:**
- No camera calibration needed
- Works with any video
- Mathematically simple
- Reliable and deterministic

**Limitation:** Vehicle must travel across both lines for speed to be calculated. Short clips or sideways-moving vehicles won't trigger speed.

---

## 8. Violation Engine — All Rules

### 8.1 Fine Table (`violation_engine/rules.py`)

```python
FINE_RULES = {
    'NHelmet':      {'description': 'No Helmet',           'fine': 500},
    'no_seatbelt':  {'description': 'No Seatbelt',         'fine': 500},
    'triple_riding':{'description': 'Triple Riding',        'fine': 1000},
    'red_light':    {'description': 'Red Light Violation',  'fine': 1000},
    'speeding':     {'description': 'Over Speeding',        'fine': 1500},
    'wrong_way':    {'description': 'Wrong Way Driving',    'fine': 2000},
    'no_parking':   {'description': 'No Parking Violation', 'fine': 300},
}
```

### 8.2 Complete Violation Check Logic

```python
def check_vehicle_violations(vehicle, persons, helmets, seatbelts):
    vclass = vehicle['class']
    vbbox  = vehicle['bbox']
    violations = []

    if vclass == 'motorcycle':
        # Rule 1: Helmet check
        near_helmets = get_helmets_near(helmets, vbbox)
        no_helmet = any(h['class']=='NHelmet' and h['confidence']>=0.35
                        for h in near_helmets)
        if no_helmet or len(near_helmets) == 0:
            violations.append({'description':'No Helmet', 'fine':500})

        # Rule 2: Triple riding (3+ persons)
        if get_persons_on_vehicle(persons, vbbox) >= 3:
            violations.append({'description':'Triple Riding', 'fine':1000})

    elif vclass in ['car', 'truck', 'bus']:
        # Rule 3: Seatbelt check
        if len(get_seatbelts_near(seatbelts, vbbox)) == 0:
            violations.append({'description':'No Seatbelt', 'fine':500})

    # Video-only rules (added in detect_video route):
    # Rule 4: Red light      → check_red_light_violation()
    # Rule 5: Over speeding  → check_speed_violation()
    # Rule 6: Wrong way      → check_wrong_way_violation()
    # Rule 7: No parking     → check_no_parking_violation()

    return deduplicate(violations)
```

### 8.3 Deduplication

```python
# Prevent same violation type appearing twice
seen = set()
unique = [v for v in violations
          if v['description'] not in seen
          and not seen.add(v['description'])]
```

---

## 9. Challan Generation

### 9.1 PDF Structure

```
─────────────────────────────────────────
  🚔 TRAFFIC POLICE DEPARTMENT
  E-Challan — Digital Traffic Violation Notice
─────────────────────────────────────────
  Challan ID   │ CH20260404141518502
  Date & Time  │ 04-04-2026 14:15:18
  Location     │ Camera ID: CAM-01
  Vehicle No.  │ WB20S6340
  Status       │ UNPAID
─────────────────────────────────────────
  Violations Detected:

  #  │ Violation       │ Fine Amount
  1  │ No Helmet       │ Rs. 500
  2  │ Triple Riding   │ Rs. 1,000
     │ TOTAL FINE      │ Rs. 1,500
─────────────────────────────────────────
  Pay fine within 30 days to avoid legal action.
  Generated by AI-powered Traffic Violation Detection System
─────────────────────────────────────────
```

### 9.2 Challan ID Format

```python
def generate_challan_id():
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')  # 14 digits
    random_suffix = random.randint(100, 999)              # 3 digits
    return f"CH{timestamp}{random_suffix}"
# Example: CH20260404141518502
```

### 9.3 make_challan() Function

```python
def make_challan(key, violations, challan_done):
    if not violations: return None
    if key in challan_done: return None  # Prevent duplicate

    challan_done.add(key)
    challan_data = {
        'challan_id':     generate_challan_id(),
        'timestamp':      datetime.now().strftime('%d-%m-%Y %H:%M:%S'),
        'vehicle_number': key,
        'location':       'Camera ID: CAM-01',
        'violations':     violations,
        'total_fine':     get_total_fine(violations)
    }
    pdf_path = generate_challan_pdf(challan_data)
    save_challan(challan_data, pdf_path)
    return challan_data, pdf_path
```

---

## 10. Database Schema

### 10.1 SQLite Table: challans

```sql
CREATE TABLE IF NOT EXISTS challans (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    challan_id    TEXT UNIQUE,          -- CH20260404141518502
    vehicle_number TEXT,                -- WB20S6340 or UNKNOWN-CAR-0
    violation_type TEXT,                -- No Helmet, No Seatbelt
    fine_amount   INTEGER,              -- 500
    location      TEXT,                 -- Camera ID: CAM-01
    timestamp     TEXT,                 -- 04-04-2026 14:15:18
    pdf_path      TEXT,                 -- D:/output/challan_CH...pdf
    status        TEXT DEFAULT 'UNPAID',-- UNPAID / PAID / WAIVED
    payment_mode  TEXT DEFAULT NULL,    -- Cash / UPI / Online / Cheque
    payment_date  TEXT DEFAULT NULL,    -- 04-04-2026 15:30
    remarks       TEXT DEFAULT NULL     -- Free text notes
);
```

### 10.2 Status Values

| Status | Meaning | Payment Mode Required |
|---|---|---|
| UNPAID | Default — fine not collected | No |
| PAID | Fine collected | Yes (Cash/UPI/Online/Cheque) |
| WAIVED | Challan forgiven/cancelled | No |

### 10.3 Key DB Operations

```python
# Save new challan
def save_challan(challan_data, pdf_path):
    cursor.execute('INSERT OR IGNORE INTO challans ...')

# Update status
def update_challan_status(challan_id, status, payment_mode, remarks):
    cursor.execute('UPDATE challans SET status=?, payment_mode=?, ...')

# Delete
def delete_challan(challan_id):
    cursor.execute('DELETE FROM challans WHERE challan_id=?')

# Search with filter
def search_challans(query, status_filter):
    cursor.execute('SELECT ... WHERE vehicle_number LIKE ? AND status=?')
```

---

## 11. Flask API Reference

| Endpoint | Method | Purpose | Body/Params |
|---|---|---|---|
| `/` | GET | Serve web UI | — |
| `/detect_image` | POST | Process image | `form-data: image` |
| `/detect_video` | POST | Process video | `form-data: video` |
| `/output_video` | GET | Stream annotated video | — |
| `/dashboard_data` | GET | All stats + challans | — |
| `/update_challan` | POST | Update status/payment | JSON |
| `/delete_challan/<id>` | DELETE | Delete record | — |
| `/search_challans` | GET | Search + filter | `?q=&status=` |
| `/export_csv` | GET | Download all as CSV | — |
| `/download_pdf/<id>` | GET | Download PDF | — |

### detect_image Response:
```json
{
  "image": "base64_jpg_string",
  "vehicles": 2,
  "persons": 3,
  "plate": "WB20S6340",
  "vehicle_types": ["motorcycle", "car"],
  "violations": [
    {"desc": "No Helmet", "fine": 500},
    {"desc": "No Seatbelt", "fine": 500}
  ],
  "total_fine": 1000,
  "challan_id": "CH20260404141518502",
  "total_challans": 2
}
```

### update_challan Request:
```json
{
  "challan_id": "CH20260404141518502",
  "status": "PAID",
  "payment_mode": "UPI",
  "remarks": "Paid via GPay"
}
```

---

## 12. Dashboard Features — All

### 12.1 Stats Cards (8 cards)
| Card | Color | Data |
|---|---|---|
| Total Challans | Blue | COUNT(*) |
| Unpaid | Red | COUNT WHERE status='UNPAID' |
| Paid | Green | COUNT WHERE status='PAID' |
| Waived | Cyan | COUNT WHERE status='WAIVED' |
| Total Fine | Orange | SUM(fine_amount) |
| Pending | Red | SUM WHERE UNPAID |
| Collected | Green | SUM WHERE PAID |
| Actions | — | Refresh / CSV / Print |

### 12.2 Search & Filter
```javascript
// Real-time search as user types
async function searchChallans() {
    const q = document.getElementById('search-input').value;
    const status = document.getElementById('filter-select').value;
    const res = await fetch(`/search_challans?q=${q}&status=${status}`);
}
```

### 12.3 Update Modal
Fields:
- Status: UNPAID / PAID / WAIVED
- Payment Mode (visible only when PAID): Cash / Online / UPI / Cheque
- Remarks: free text

### 12.4 Charts (Collapsible FAB)
- Floating 📊 button bottom-right
- Click → slides up chart panel
- Click ✕ → closes panel
- Pie chart: violations by type (doughnut)
- Bar chart: paid vs unpaid vs waived

### 12.5 Auto Refresh
```javascript
setInterval(() => {
    autoRefreshTimer--;
    if (autoRefreshTimer <= 0) {
        autoRefreshTimer = 30;
        loadDashboard();  // Reload every 30 seconds
    }
}, 1000);
```

### 12.6 CSV Export
Columns exported: ID, Challan ID, Vehicle, Violation, Fine, Location, Time, PDF Path, Status, Payment Mode, Payment Date, Remarks

---

## 13. Web UI Architecture

### 13.1 Tab Structure

```
index.html
├── Header
│   ├── Auto-refresh timer (pulse dot)
│   ├── Title (TRAFFIC POLICE DEPARTMENT)
│   └── Notification bell (unpaid count)
│
├── Tab Bar
│   ├── 📷 Image Detection
│   ├── 🎥 Video Detection
│   └── 📊 Dashboard
│
├── Tab 1: Image Detection
│   ├── Left: Upload area + preview + detect button
│   └── Right: Result image + report + fine total + challan download
│
├── Tab 2: Video Detection
│   ├── Left: Upload + preview + detect button + features list
│   └── Right: Annotated video + summary report
│
├── Tab 3: Dashboard
│   ├── Stats grid (8 cards)
│   ├── Search + filter bar
│   ├── Challans table (scrollable, max-height 360px)
│   │   └── Per row: Edit ✏️ / PDF 📄 / Delete 🗑️
│   └── Update modal (overlay)
│
└── FAB Button (bottom-right, dashboard only)
    └── Collapsible chart panel
        ├── Pie chart (violations by type)
        └── Bar chart (payment status)
```

---

## 14. Key Design Decisions

### 14.1 YOLOv8n vs Larger Models

| Model | Size | VRAM | Speed | mAP |
|---|---|---|---|---|
| YOLOv8n | 6MB | ~1.2GB | Fastest | Good |
| YOLOv8s | 22MB | ~2GB | Fast | Better |
| YOLOv8m | 52MB | ~4GB | Medium | Best |

**Decision: YOLOv8n** — Running 3 custom models + COCO simultaneously on 4GB VRAM requires the smallest variant. All 3 models + inference stays under 3GB VRAM.

### 14.2 EasyOCR vs Tesseract vs CRNN

| OCR | Accuracy | Indian Plates | Complexity |
|---|---|---|---|
| Tesseract | Low | Poor | Medium |
| EasyOCR | Good | Good | Simple |
| CRNN (custom) | Best | Best | High |

**Decision: EasyOCR** — Best balance of accuracy and simplicity. `pip install easyocr` + GPU support out of box.

### 14.3 SQLite vs MongoDB

| DB | Setup | Scale | Complexity |
|---|---|---|---|
| SQLite | Zero | Small | Simple |
| MongoDB | Server needed | Large | Complex |

**Decision: SQLite** — Project is single-machine. No server needed. Simple backup (one file). Built into Python.

### 14.4 Flask vs FastAPI

| Framework | Template | Simplicity | Async |
|---|---|---|---|
| Flask | Jinja2 built-in | Simpler | No |
| FastAPI | Manual | More complex | Yes |

**Decision: Flask** — HTML rendering + REST API in one. No async needed. Well-documented. Less boilerplate.

### 14.5 Confidence Threshold = 0.35

Testing on multiple images:
- `conf=0.5` → motorcycle often missed
- `conf=0.25` → too many false positives
- `conf=0.35` → best for traffic scene diversity

### 14.6 Line Crossing vs Optical Flow for Speed

| Method | Accuracy | Camera Dependency | Complexity |
|---|---|---|---|
| Line crossing | Medium | None | Simple |
| Optical flow | High | None | Complex |
| Camera calibration | High | High | Very complex |

**Decision: Line crossing** — Simple, reliable, no calibration needed. Sufficient for demonstration purposes.

---

## 15. Known Issues & Fixes

### Issue 1: Duplicate Vehicle Detection
**Symptom:** Same vehicle detected twice.
**Fix:** Grid-based deduplication
```python
vkey = f"{v['bbox'][0]//50}_{v['bbox'][1]//50}"
if vkey in processed_veh: continue
```

### Issue 2: NHelmet Detected But No Violation
**Cause:** Old logic required plate for challan.
**Fix:** UNKNOWN key if no plate found.

### Issue 3: Wrong Vehicle Type Rule Applied
**Symptom:** Seatbelt violation on motorcycle.
**Fix:** Explicit vehicle type check before rule application.

### Issue 4: Speed Always 0
**Cause:** Vehicle didn't cross both speed lines.
**Status:** Known limitation. Works on longer videos with top-down angle.

### Issue 5: Windows Training Crash
**Fix:** `if __name__ == '__main__':` + `workers=0`

### Issue 6: Gradio Tab Not Switching
**Fix:** Migrated to Flask.

### Issue 7: Double Riding Flagged
**Fix:** Changed threshold from `>= 2` to `>= 3`.

### Issue 8: Large Video Upload Timeout
**Status:** Flask default request size is 16MB. For larger videos, increase:
```python
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
```

---

## 16. Performance Benchmarks

### 16.1 Model Accuracy

| Model | mAP50 | mAP50-95 | Training Epochs |
|---|---|---|---|
| Helmet | 91.9% | 56.4% | 33 (early stop) |
| Number Plate | 89.6% | 63.6% | 23 (early stop) |
| Plate class | 97.3% | 84.5% | — |
| Seatbelt | Trained | — | 50 |

### 16.2 Inference Speed (RTX 3050)

| Stage | Time |
|---|---|
| Vehicle Detection | ~3ms |
| Helmet Detection | ~5ms |
| Seatbelt Detection | ~5ms |
| Plate Detection | ~3ms |
| EasyOCR | ~50ms |
| **Total per frame** | **~66ms** |
| **Effective FPS** | **~15 FPS** |

### 16.3 Training Time (RTX 3050 4GB)

| Model | Epochs | Time |
|---|---|---|
| Helmet | 33 | ~35 minutes |
| Seatbelt | 50 | ~25 minutes |
| Number Plate | 23 | ~15 minutes |

### 16.4 Video Processing

For a 30-second MP4 at 30 FPS:
- Total frames: 900
- Processed (every 12th): 75 frames
- Processing time: ~5-8 seconds
- Output: annotated video + challans

---

## 17. Future Scope

### 17.1 Live CCTV Integration
```python
# Simple change in detect_video():
cap = cv2.VideoCapture("rtsp://camera_ip:554/stream")
# Or IP webcam app
cap = cv2.VideoCapture("http://192.168.1.5:8080/video")
```

### 17.2 RTO Database Lookup
```python
import requests
response = requests.get(f"https://rto-api.in/v1/{plate_number}")
owner = response.json()['owner_name']
owner_phone = response.json()['mobile']
```

### 17.3 SMS Notification
```python
from twilio.rest import Client
client.messages.create(
    to=owner_phone,
    body=f"Traffic Challan: {challan_id}. Fine: Rs.{fine}. Pay within 30 days."
)
```

### 17.4 Night Vision
```python
# Preprocess dark frames
frame = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
```

### 17.5 Mobile Phone Detection
Requires custom dataset of drivers using phones while driving.

### 17.6 Online Payment Integration
Add payment gateway (Razorpay/PayU) to allow online fine payment directly from dashboard.

---

## 📎 Quick Reference

```bash
# Activate environment
venv\Scripts\activate

# Train models (one time)
python train_helmet.py
python train_seatbelt.py
python train_numberplate.py

# Start application
python flask_app.py

# Access at
http://127.0.0.1:5000

# Push to GitHub
git add .
git commit -m "Update"
git push
```

---

*Documentation: Yash Gupta | HP Victus 15 | RTX 3050 4GB | May 2026*