import cv2
import sys
from datetime import datetime
sys.path.append(r'D:\traffic_violation_system')

from detection.vehicle_detector import VehicleDetector
from detection.helmet_detector import HelmetDetector
from detection.seatbelt_detector import SeatbeltDetector
from detection.plate_detector import PlateDetector
from detection.traffic_light import TrafficLightDetector
from detection.wrong_way import WrongWayDetector
from detection.no_parking import NoParkingDetector
from detection.triple_riding import TripleRidingDetector
from tracking.speed_tracker import SpeedTracker
from violation_engine.rules import (check_helmet_violation,
                                     check_seatbelt_violation,
                                     check_speed_violation,
                                     check_red_light_violation,
                                     check_wrong_way_violation,
                                     check_no_parking_violation,
                                     check_triple_riding_violation,
                                     generate_challan_id,
                                     get_total_fine)
from challan.generator import generate_challan_pdf
from database.db import save_challan
from database.models import create_tables

challan_registry = {}
CHALLAN_COOLDOWN = 300
MIN_CONFIDENCE = 0.45

def process_violation(violations, plate_text, frame_count):
    if not violations:
        return
    if plate_text in challan_registry:
        if frame_count - challan_registry[plate_text] < CHALLAN_COOLDOWN:
            return
    challan_registry[plate_text] = frame_count
    challan_data = {
        'challan_id': generate_challan_id(),
        'timestamp': datetime.now().strftime('%d-%m-%Y %H:%M:%S'),
        'vehicle_number': plate_text,
        'location': 'Camera ID: CAM-01',
        'violations': violations,
        'total_fine': get_total_fine(violations)
    }
    pdf_path = generate_challan_pdf(challan_data)
    save_challan(challan_data, pdf_path)
    print(f"\n🚨 VIOLATION DETECTED!")
    print(f"Vehicle:    {plate_text}")
    print(f"Violations: {[v['description'] for v in violations]}")
    print(f"Fine:       Rs. {challan_data['total_fine']}")
    print(f"Challan:    {pdf_path}\n")

def get_vehicle_type(vehicles):
    if not vehicles:
        return 'unknown'
    types = [v['class'] for v in vehicles]
    if 'motorcycle' in types:
        return 'motorcycle'
    if 'car' in types:
        return 'car'
    if 'truck' in types:
        return 'truck'
    if 'bus' in types:
        return 'bus'
    return 'unknown'

def main():
    create_tables()

    print("Models load ho rahe hain...")
    vehicle_detector = VehicleDetector()
    helmet_detector = HelmetDetector()
    seatbelt_detector = SeatbeltDetector()
    plate_detector = PlateDetector()
    traffic_light_detector = TrafficLightDetector()
    wrong_way_detector = WrongWayDetector()
    speed_tracker = SpeedTracker()
    triple_riding_detector = TripleRidingDetector()

    no_parking_detector = NoParkingDetector()
    no_parking_detector.add_zone(50, 400, 300, 550, "No Parking Zone 1")
    no_parking_detector.add_zone(700, 400, 950, 550, "No Parking Zone 2")

    print("Sab models ready!\n")

    cv2.namedWindow('Traffic Violation System', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Traffic Violation System',
                          cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    cap = cv2.VideoCapture(r'D:\traffic_violation_system\videos\traffic2.mp4')
    frame_count = 0
    total_challans = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video khatam!")
            break

        # Set speed lines based on video height (general for any resolution)
        if speed_tracker.LINE1_Y is None:
            speed_tracker.set_lines(frame.shape[0])

        frame_count += 1

        # Detection (every frame for accurate speed tracking)
        vehicles, persons = vehicle_detector.detect(frame)
        helmets = helmet_detector.detect(frame)
        seatbelts = seatbelt_detector.detect(frame)
        plates = plate_detector.detect(frame)
        lights = traffic_light_detector.detect(frame)

        # Vehicle type
        vehicle_type = get_vehicle_type(vehicles)

        # Red light
        is_red_light = traffic_light_detector.is_red(lights)

        # Violations
        all_violations = []

        # Helmet
        if vehicle_type == 'motorcycle':
            all_violations += check_helmet_violation(helmets)

        # Seatbelt
        if vehicle_type in ['car', 'truck', 'bus']:
            all_violations += check_seatbelt_violation(seatbelts)

        # Red light
        if is_red_light and len(vehicles) > 0:
            all_violations += check_red_light_violation(is_red_light, vehicles)

        # Triple Riding
        triple_violations, frame = triple_riding_detector.detect(
            vehicles, persons, frame)
        all_violations += check_triple_riding_violation(triple_violations)

        # Speed + Wrong Way + No Parking
        for i, v in enumerate(vehicles):
            speed = speed_tracker.update(v['bbox'])
            is_speeding, spd = speed_tracker.is_speeding(v['bbox'])
            if is_speeding:
                all_violations += check_speed_violation(spd, limit=60)

            is_wrong = wrong_way_detector.update(i, v['bbox'])
            if is_wrong:
                all_violations += check_wrong_way_violation(is_wrong)
                frame = wrong_way_detector.draw_direction(
                    frame, i, v['bbox'], is_wrong)

            is_parked, duration = no_parking_detector.update(i, v['bbox'])
            if is_parked:
                all_violations += check_no_parking_violation(
                    is_parked, duration)

        # Plate text
        plate_text = ''
        for p in plates:
            if len(p['text']) >= 4:
                plate_text = p['text'].strip()
                break

        # Challan
        if all_violations and plate_text:
            process_violation(all_violations, plate_text, frame_count)
            total_challans = len(challan_registry)

        # ---- Draw ----

        # No Parking Zones
        frame = no_parking_detector.draw_zones(frame)

        # Speed Lines
        frame = speed_tracker.draw_lines(frame)

        # Persons — light gray
        for p in persons:
            px1, py1, px2, py2 = p['bbox']
            cv2.rectangle(frame, (px1,py1), (px2,py2), (200,200,200), 1)

        # Vehicles
        for i, v in enumerate(vehicles):
            x1, y1, x2, y2 = v['bbox']
            speed = speed_tracker.vehicle_speeds.get(i, 0)
            is_wrong = wrong_way_detector.is_wrong_way(i)
            is_parked, dur = no_parking_detector.update(i, v['bbox'])

            if is_wrong:
                color = (0, 0, 255)
            elif is_parked:
                color = (0, 165, 255)
            elif speed > 60:
                color = (0, 100, 255)
            else:
                color = (255, 0, 0)

            label = f"{v['class']}"
            if speed > 0:
                label += f" {speed}km/h"
            if is_parked:
                label += f" PARKED {int(dur)}s"

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label,
                       (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, color, 2)

        # Helmet
        for h in helmets:
            if h['confidence'] < MIN_CONFIDENCE:
                continue
            x1, y1, x2, y2 = h['bbox']
            color = (0,255,0) if h['class'] == 'Helmet' else (0,0,255)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"{h['class']} {h['confidence']:.2f}",
                       (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, color, 2)

        # Seatbelt
        for s in seatbelts:
            if s['confidence'] < MIN_CONFIDENCE:
                continue
            x1, y1, x2, y2 = s['bbox']
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 2)
            cv2.putText(frame, f"Seatbelt {s['confidence']:.2f}",
                       (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0,255,255), 2)

        # Plate
        for p in plates:
            x1, y1, x2, y2 = p['bbox']
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,255), 2)
            if p['text']:
                cv2.putText(frame, p['text'],
                           (x1, y2+25),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (255,255,255), 2)

        # Traffic Light
        for l in lights:
            x1, y1, x2, y2 = l['bbox']
            color_map = {
                'red': (0,0,255),
                'yellow': (0,255,255),
                'green': (0,255,0),
                'unknown': (255,255,255)
            }
            color = color_map.get(l['color'], (255,255,255))
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 3)
            cv2.putText(frame, f"Light: {l['color']}",
                       (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, color, 2)

        # Red Light Indicator
        indicator_color = (0,0,255) if is_red_light else (0,255,0)
        indicator_text = "RED LIGHT!" if is_red_light else "CLEAR"
        cv2.circle(frame, (frame.shape[1]-60, 40), 20,
                  indicator_color, -1)
        cv2.putText(frame, indicator_text,
                   (frame.shape[1]-160, 48),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, indicator_color, 2)

        # Info Panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (320,240), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        cv2.putText(frame, f"Vehicles:   {len(vehicles)}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (255,255,255), 2)
        cv2.putText(frame, f"Persons:    {len(persons)}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (200,200,200), 2)
        cv2.putText(frame, f"Type:       {vehicle_type}",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (255,255,0), 2)
        cv2.putText(frame, f"Traffic:    {indicator_text}",
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, indicator_color, 2)
        cv2.putText(frame, f"Violations: {len(all_violations)}",
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0,0,255), 2)
        cv2.putText(frame, f"Challans:   {total_challans}",
                   (10, 180), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0,255,255), 2)
        cv2.putText(frame, f"Frame:      {frame_count}",
                   (10, 210), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (200,200,200), 2)

        cv2.imshow('Traffic Violation System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nTotal challans generated: {total_challans}")

if __name__ == '__main__':
    main()