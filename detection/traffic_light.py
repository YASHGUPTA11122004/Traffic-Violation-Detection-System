import cv2
import numpy as np
from ultralytics import YOLO

class TrafficLightDetector:
    def __init__(self):
        # YOLO pre-trained model use karo
        self.model = YOLO('yolov8n.pt')
        self.traffic_light_class = 9  # COCO mein traffic light = 9

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        lights = []

        for box in results.boxes:
            cls = int(box.cls[0])
            if cls == self.traffic_light_class:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # Light crop karo
                light_img = frame[y1:y2, x1:x2]
                if light_img.size == 0:
                    continue

                # Color detect karo
                color = self.detect_color(light_img)
                lights.append({
                    'bbox': (x1, y1, x2, y2),
                    'color': color,
                    'confidence': conf
                })

        return lights

    def detect_color(self, light_img):
        hsv = cv2.cvtColor(light_img, cv2.COLOR_BGR2HSV)
        h, w = light_img.shape[:2]

        # 3 regions — top=red, mid=yellow, bottom=green
        top = hsv[:h//3, :]
        mid = hsv[h//3:2*h//3, :]
        bot = hsv[2*h//3:, :]

        # Red range
        red1 = cv2.inRange(top, np.array([0,100,100]), np.array([10,255,255]))
        red2 = cv2.inRange(top, np.array([160,100,100]), np.array([180,255,255]))
        red_pixels = cv2.countNonZero(red1) + cv2.countNonZero(red2)

        # Yellow range
        yellow = cv2.inRange(mid, np.array([20,100,100]), np.array([30,255,255]))
        yellow_pixels = cv2.countNonZero(yellow)

        # Green range
        green = cv2.inRange(bot, np.array([40,100,100]), np.array([80,255,255]))
        green_pixels = cv2.countNonZero(green)

        # Sabse zyada pixels wala color
        max_pixels = max(red_pixels, yellow_pixels, green_pixels)
        if max_pixels < 10:
            return 'unknown'
        if max_pixels == red_pixels:
            return 'red'
        if max_pixels == yellow_pixels:
            return 'yellow'
        return 'green'

    def is_red(self, lights):
        for l in lights:
            if l['color'] == 'red':
                return True
        return False