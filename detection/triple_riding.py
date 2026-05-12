import cv2

class TripleRidingDetector:
    def __init__(self):
        self.PERSON_LIMIT = 2  # Bike pe max 2 log allowed

    def detect(self, vehicles, persons, frame):
        violations = []

        for v in vehicles:
            if v['class'] != 'motorcycle':
                continue

            vx1, vy1, vx2, vy2 = v['bbox']

            # Bike ke upar kitne persons hain
            persons_on_bike = []
            for p in persons:
                px1, py1, px2, py2 = p['bbox']
                pcx = (px1 + px2) // 2
                pcy = (py1 + py2) // 2

                # Person bike ke andar hai?
                if vx1 < pcx < vx2 and vy1 < pcy < vy2:
                    persons_on_bike.append(p)

            # 3 ya zyada log = violation
            if len(persons_on_bike) >= 3:
                violations.append({
                    'bbox': v['bbox'],
                    'count': len(persons_on_bike)
                })

                # Draw karo
                cv2.rectangle(frame,
                             (vx1, vy1), (vx2, vy2),
                             (0, 0, 255), 3)
                cv2.putText(frame,
                           f"TRIPLE RIDING! ({len(persons_on_bike)} persons)",
                           (vx1, vy1-15),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0,0,255), 2)

        return violations, frame