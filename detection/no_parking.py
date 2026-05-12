import cv2
import time

class NoParkingDetector:
    def __init__(self):
        # No parking zones — (x1, y1, x2, y2)
        # Video ke hisab se adjust karna
        self.no_parking_zones = []
        self.stationary_vehicles = {}  # id → {first_seen, bbox}
        self.PARKING_TIME_LIMIT = 5.0  # 5 seconds se zyada = violation

    def add_zone(self, x1, y1, x2, y2, name="No Parking Zone"):
        self.no_parking_zones.append({
            'bbox': (x1, y1, x2, y2),
            'name': name
        })

    def is_in_zone(self, vehicle_bbox):
        vx1, vy1, vx2, vy2 = vehicle_bbox
        vcx = (vx1 + vx2) // 2
        vcy = (vy1 + vy2) // 2

        for zone in self.no_parking_zones:
            zx1, zy1, zx2, zy2 = zone['bbox']
            if zx1 < vcx < zx2 and zy1 < vcy < zy2:
                return True, zone['name']
        return False, None

    def update(self, vehicle_id, bbox, current_speed=0):
        in_zone, zone_name = self.is_in_zone(bbox)

        if not in_zone:
            if vehicle_id in self.stationary_vehicles:
                del self.stationary_vehicles[vehicle_id]
            return False, 0

        # Vehicle zone mein hai
        if vehicle_id not in self.stationary_vehicles:
            self.stationary_vehicles[vehicle_id] = {
                'first_seen': time.time(),
                'bbox': bbox,
                'zone': zone_name
            }

        # Kitna time ho gaya
        elapsed = time.time() - self.stationary_vehicles[vehicle_id]['first_seen']

        if elapsed > self.PARKING_TIME_LIMIT:
            return True, elapsed

        return False, elapsed

    def draw_zones(self, frame):
        for zone in self.no_parking_zones:
            x1, y1, x2, y2 = zone['bbox']
            # Red transparent zone
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,0,255), -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            # Border
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
            cv2.putText(frame, zone['name'],
                       (x1+5, y1+25),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0,0,255), 2)
        return frame