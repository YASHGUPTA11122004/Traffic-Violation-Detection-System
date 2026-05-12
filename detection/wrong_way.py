import cv2

class WrongWayDetector:
    def __init__(self):
        # Expected direction — neeche se upar (y decrease hona chahiye)
        # Agar y increase ho raha hai = wrong way
        self.vehicle_history = {}  # id → [positions]
        self.HISTORY_LENGTH = 5

    def update(self, vehicle_id, bbox):
        x1, y1, x2, y2 = bbox
        cy = (y1 + y2) // 2

        if vehicle_id not in self.vehicle_history:
            self.vehicle_history[vehicle_id] = []

        self.vehicle_history[vehicle_id].append(cy)

        # Sirf last 5 positions rakho
        if len(self.vehicle_history[vehicle_id]) > self.HISTORY_LENGTH:
            self.vehicle_history[vehicle_id].pop(0)

        return self.is_wrong_way(vehicle_id)

    def is_wrong_way(self, vehicle_id):
        positions = self.vehicle_history.get(vehicle_id, [])

        if len(positions) < self.HISTORY_LENGTH:
            return False

        # Agar y consistently badh raha hai = wrong way
        increasing = sum(1 for i in range(1, len(positions))
                        if positions[i] > positions[i-1])

        # 4 out of 5 frames mein badh raha hai = wrong way
        return increasing >= 4

    def draw_direction(self, frame, vehicle_id, bbox, is_wrong):
        if not is_wrong:
            return frame
        x1, y1, x2, y2 = bbox
        # Wrong way alert
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 3)
        cv2.putText(frame, "WRONG WAY!",
                   (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0,0,255), 2)
        return frame