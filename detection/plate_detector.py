from ultralytics import YOLO
import easyocr
import cv2

class PlateDetector:
    def __init__(self):
        self.model = YOLO(r'D:\traffic_violation_system\models\numberplate_model\weights\best.pt')
        self.reader = easyocr.Reader(['en'], gpu=True)

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        plates = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Plate crop karo
            plate_img = frame[y1:y2, x1:x2]
            if plate_img.size == 0:
                continue
            # OCR se text nikalo
            ocr_result = self.reader.readtext(plate_img)
            plate_text = ''
            for (_, text, conf) in ocr_result:
                if conf > 0.3:
                    plate_text += text + ' '

            plates.append({
                'bbox': (x1, y1, x2, y2),
                'text': plate_text.strip()
            })

        return plates