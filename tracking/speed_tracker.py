import cv2
import numpy as np
from datetime import datetime

class SpeedTracker:
    def __init__(self):
        # Speed lines - will be set based on video frame height
        self.LINE1_Y = None
        self.LINE2_Y = None
        self.REAL_DISTANCE = 15  # meters between lines
        self.SPEED_LIMIT = 60    # km/h

        # Simple position-based tracking
        self.last_positions = {}  # position_key → (timestamp, y_position)
        self.vehicle_speeds = {}  # position_key → speed

    def set_lines(self, frame_height):
        """Set speed lines based on frame height for any video resolution."""
        self.LINE1_Y = int(frame_height * 0.4)  # 40% from top
        self.LINE2_Y = int(frame_height * 0.6)  # 60% from bottom

    def get_position_key(self, bbox):
        """Create a key based on horizontal position only for more stable tracking."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        return f"{cx // 80}"

    def update(self, bbox):
        """Estimate speed when the same vehicle crosses both speed lines."""
        key = self.get_position_key(bbox)
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        current_time = datetime.now()

        speed = 0

        if key in self.last_positions:
            start_time, start_y = self.last_positions[key]

            # Travel from above to below
            if start_y < self.LINE1_Y and cy > self.LINE2_Y:
                elapsed = (current_time - start_time).total_seconds()
                if elapsed > 0.1:
                    speed_ms = self.REAL_DISTANCE / elapsed
                    speed = round(speed_ms * 3.6, 1)
                    self.vehicle_speeds[key] = speed
                    print(f"Vehicle at {key} crossed LINE1→LINE2: speed={speed} km/h, elapsed={elapsed:.2f}s")
                    del self.last_positions[key]

            # Travel from below to above
            elif start_y > self.LINE2_Y and cy < self.LINE1_Y:
                elapsed = (current_time - start_time).total_seconds()
                if elapsed > 0.1:
                    speed_ms = self.REAL_DISTANCE / elapsed
                    speed = round(speed_ms * 3.6, 1)
                    self.vehicle_speeds[key] = speed
                    print(f"Vehicle at {key} crossed LINE2→LINE1: speed={speed} km/h, elapsed={elapsed:.2f}s")
                    del self.last_positions[key]

        else:
            if cy < self.LINE1_Y or cy > self.LINE2_Y:
                self.last_positions[key] = (current_time, cy)
                print(f"Vehicle at {key} started tracking at y={cy}")

        return self.vehicle_speeds.get(key, 0)

    def is_speeding(self, bbox):
        key = self.get_position_key(bbox)
        speed = self.vehicle_speeds.get(key, 0)
        return speed > self.SPEED_LIMIT, speed

    def draw_lines(self, frame):
        h, w = frame.shape[:2]
        # Line 1 — yellow
        cv2.line(frame, (0, self.LINE1_Y), (w, self.LINE1_Y),
                (0, 255, 255), 2)
        cv2.putText(frame, "Speed Line 1",
                   (10, self.LINE1_Y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        # Line 2 — orange
        cv2.line(frame, (0, self.LINE2_Y), (w, self.LINE2_Y),
                (0, 165, 255), 2)
        cv2.putText(frame, "Speed Line 2",
                   (10, self.LINE2_Y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
        return frame