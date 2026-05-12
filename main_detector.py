import cv2
import sys
sys.path.append(r'D:\traffic_violation_system')

from detection.vehicle_detector import VehicleDetector
from detection.helmet_detector import HelmetDetector
from detection.seatbelt_detector import SeatbeltDetector
from detection.plate_detector import PlateDetector

def main():
    # Sab models load karo
    print("Models load ho rahe hain...")
    vehicle_detector = VehicleDetector()
    helmet_detector = HelmetDetector()
    seatbelt_detector = SeatbeltDetector()
    plate_detector = PlateDetector()
    print("Sab models ready!")

    # Window fullscreen
    cv2.namedWindow('Traffic Violation Detection', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Traffic Violation Detection',
                          cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    # Video load karo
    cap = cv2.VideoCapture(r'D:\traffic_violation_system\videos\traffic.mp4')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video khatam!")
            break

        # 1. Vehicles detect karo
        vehicles = vehicle_detector.detect(frame)

        # 2. Helmet detect karo
        helmets = helmet_detector.detect(frame)

        # 3. Seatbelt detect karo
        seatbelts = seatbelt_detector.detect(frame)

        # 4. Number plate detect karo
        plates = plate_detector.detect(frame)

        # 5. Vehicles — blue box
        for v in vehicles:
            x1, y1, x2, y2 = v['bbox']
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.putText(frame, f"{v['class']} {v['confidence']:.2f}",
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (255,0,0), 2)

        # 6. Helmet — green=helmet, red=no helmet
        for h in helmets:
            x1, y1, x2, y2 = h['bbox']
            color = (0,255,0) if h['class'] == 'Helmet' else (0,0,255)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"{h['class']} {h['confidence']:.2f}",
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, color, 2)

        # 7. Seatbelt — yellow box
        for s in seatbelts:
            x1, y1, x2, y2 = s['bbox']
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 2)
            cv2.putText(frame, f"Seatbelt {s['confidence']:.2f}",
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0,255,255), 2)

        # 8. Number plate — white box + OCR text
        for p in plates:
            x1, y1, x2, y2 = p['bbox']
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,255), 2)
            cv2.putText(frame, p['text'],
                       (x1, y2+25), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (255,255,255), 2)

        # 9. Info panel — upar left corner
        y_pos = 30
        cv2.putText(frame, f"Vehicles: {len(vehicles)}",
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (255,255,255), 2)
        y_pos += 30
        cv2.putText(frame, f"Helmets: {len(helmets)}",
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0,255,0), 2)
        y_pos += 30
        cv2.putText(frame, f"Seatbelts: {len(seatbelts)}",
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0,255,255), 2)
        y_pos += 30
        cv2.putText(frame, f"Plates: {len(plates)}",
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (255,255,255), 2)

        # Show karo
        cv2.imshow('Traffic Violation Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()