from ultralytics import YOLO

class HelmetDetector:
    def __init__(self):
        self.model = YOLO(r'D:\traffic_violation_system\models\helmet_model3\weights\best.pt')

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls = int(box.cls[0])
            name = self.model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append({
                'class': name,
                'confidence': conf,
                'bbox': (x1, y1, x2, y2)
            })

        return detections