import cv2
from ultralytics import YOLO

class VehicleDetector:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        self.person_class = 0

    def detect(self, frame):
        results = self.model(
            frame,
            verbose=False,
            conf=0.35,      # Higher confidence
            iou=0.5       # NMS threshold
        )[0]

        vehicles = []
        persons  = []

        for box in results.boxes:
            cls  = int(box.cls[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            if cls in self.vehicle_classes:
                vehicles.append({
                    'class':      self.vehicle_classes[cls],
                    'confidence': conf,
                    'bbox':       (x1, y1, x2, y2)
                })
            elif cls == self.person_class:
                persons.append({
                    'class':      'person',
                    'confidence': conf,
                    'bbox':       (x1, y1, x2, y2)
                })

        return vehicles, persons