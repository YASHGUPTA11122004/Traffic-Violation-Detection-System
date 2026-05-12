from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import sys
import os
import base64
import csv
from datetime import datetime
from PIL import Image
import sqlite3

sys.path.append(r'D:\traffic_violation_system')

from detection.vehicle_detector import VehicleDetector
from detection.helmet_detector import HelmetDetector
from detection.seatbelt_detector import SeatbeltDetector
from detection.plate_detector import PlateDetector
from detection.triple_riding import TripleRidingDetector
from detection.traffic_light import TrafficLightDetector
from detection.wrong_way import WrongWayDetector
from detection.no_parking import NoParkingDetector
from tracking.speed_tracker import SpeedTracker
from violation_engine.rules import (
    check_helmet_violation,
    check_seatbelt_violation,
    check_triple_riding_violation,
    check_speed_violation,
    check_red_light_violation,
    check_wrong_way_violation,
    check_no_parking_violation,
    generate_challan_id,
    get_total_fine
)
from challan.generator import generate_challan_pdf
from database.db import save_challan, update_challan_status, delete_challan
from database.models import create_tables, DB_PATH

app = Flask(__name__)

print("Models load ho rahe hain...")
create_tables()
vehicle_detector       = VehicleDetector()
helmet_detector        = HelmetDetector()
seatbelt_detector      = SeatbeltDetector()
plate_detector         = PlateDetector()
triple_detector        = TripleRidingDetector()
traffic_light_detector = TrafficLightDetector()
no_parking_detector    = NoParkingDetector()
no_parking_detector.add_zone(50, 300, 300, 450, "No Parking Zone 1")
no_parking_detector.add_zone(700, 300, 950, 450, "No Parking Zone 2")
print("Sab models ready!")

MIN_CONF   = 0.35
OUTPUT_DIR = r'D:\traffic_violation_system\output'

# ── Helpers ───────────────────────────────────────────────
def get_bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) // 2, (y1 + y2) // 2


def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_overlap(obj_bbox, veh_bbox):
    ox1, oy1, ox2, oy2 = obj_bbox
    vx1, vy1, vx2, vy2 = veh_bbox
    inter_x1 = max(ox1, vx1)
    inter_y1 = max(oy1, vy1)
    inter_x2 = min(ox2, vx2)
    inter_y2 = min(oy2, vy2)
    width = max(0, inter_x2 - inter_x1)
    height = max(0, inter_y2 - inter_y1)
    return width * height


def is_inside_or_near(obj_bbox, veh_bbox, margin=40):
    ox1, oy1, ox2, oy2 = obj_bbox
    vx1, vy1, vx2, vy2 = veh_bbox

    # Check if object center is inside expanded vehicle area
    ocx, ocy = get_bbox_center(obj_bbox)
    if (vx1 - margin <= ocx <= vx2 + margin and
        vy1 - margin <= ocy <= vy2 + margin):
        return True

    # Check if object overlaps enough with vehicle bbox
    overlap = bbox_overlap(obj_bbox, veh_bbox)
    if overlap == 0:
        return False

    obj_area = bbox_area(obj_bbox)
    veh_area = bbox_area(veh_bbox)
    if obj_area == 0 or veh_area == 0:
        return False

    if overlap / obj_area >= 0.05 or overlap / veh_area >= 0.05:
        return True

    return False


def get_persons_on_vehicle(persons, vbbox, margin=30):
    return sum(1 for p in persons
               if is_inside_or_near(p['bbox'], vbbox, margin))


def get_helmets_near(helmets, vbbox, margin=60):
    return [h for h in helmets
            if is_inside_or_near(h['bbox'], vbbox, margin)]


def get_seatbelts_near(seatbelts, vbbox, margin=60):
    return [s for s in seatbelts
            if is_inside_or_near(s['bbox'], vbbox, margin)]


def get_plate_near(plates, vbbox, margin=40):
    for p in plates:
        if is_inside_or_near(p['bbox'], vbbox, margin):
            if p['text'] and len(p['text']) >= 3:
                return p['text'].strip()
    return ''

def check_vehicle_violations(vehicle, persons, helmets, seatbelts):
    vclass = vehicle['class']
    vbbox  = vehicle['bbox']
    violations = []

    if vclass == 'motorcycle':
        # Helmet check — har NHelmet = violation
        near_helmets = get_helmets_near(helmets, vbbox)
        no_helmet_found = False
        for h in near_helmets:
            if h['class'] == 'NHelmet' and h['confidence'] >= MIN_CONF:
                no_helmet_found = True
                break
        # Agar koi helmet nahi mila ya NHelmet mila
        if no_helmet_found or len(near_helmets) == 0:
            violations.append({
                'type': 'no_helmet',
                'description': 'No Helmet',
                'fine': 500,
                'confidence': 1.0,
                'bbox': vbbox
            })

        # Triple riding — 3+ log
        person_count = get_persons_on_vehicle(persons, vbbox)
        if person_count >= 3:
            violations.append({
                'type': 'triple_riding',
                'description': f'Triple Riding ({person_count} persons)',
                'fine': 1000,
                'confidence': 1.0,
                'bbox': vbbox
            })

    elif vclass in ['car', 'truck', 'bus']:
        # Seatbelt check
        near_seatbelts = get_seatbelts_near(seatbelts, vbbox)
        if len(near_seatbelts) == 0:
            violations.append({
                'type': 'no_seatbelt',
                'description': 'No Seatbelt',
                'fine': 500,
                'confidence': 1.0,
                'bbox': vbbox
            })

    # Deduplicate
    seen = set()
    return [v for v in violations
            if v['description'] not in seen
            and not seen.add(v['description'])]

def make_challan(key, violations, challan_done):
    if not violations: return None
    if key in challan_done: return None
    challan_done.add(key)
    challan_data = {
        'challan_id':     generate_challan_id(),
        'timestamp':      datetime.now().strftime('%d-%m-%Y %H:%M:%S'),
        'vehicle_number': key,
        'location':       'Camera ID: CAM-01',
        'violations':     violations,
        'total_fine':     get_total_fine(violations)
    }
    pdf = generate_challan_pdf(challan_data)
    save_challan(challan_data, pdf)
    return challan_data, pdf

def detect_all(frame):
    vehicles, persons = vehicle_detector.detect(frame)
    helmets           = helmet_detector.detect(frame)
    seatbelts         = seatbelt_detector.detect(frame)
    plates            = plate_detector.detect(frame)
    return vehicles, persons, helmets, seatbelts, plates

def draw_frame(frame, vehicles, persons, helmets, seatbelts, plates):
    for v in vehicles:
        x1,y1,x2,y2 = v['bbox']
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
        cv2.putText(frame,f"{v['class']} {v['confidence']:.2f}",
                   (x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
    for p in persons:
        x1,y1,x2,y2 = p['bbox']
        cv2.rectangle(frame,(x1,y1),(x2,y2),(180,180,180),1)
    for h in helmets:
        if h['confidence'] < MIN_CONF: continue
        x1,y1,x2,y2 = h['bbox']
        color = (0,200,0) if h['class']=='Helmet' else (0,0,255)
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        cv2.putText(frame,f"{h['class']} {h['confidence']:.2f}",
                   (x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
    for s in seatbelts:
        if s['confidence'] < MIN_CONF: continue
        x1,y1,x2,y2 = s['bbox']
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)
        cv2.putText(frame,f"Seatbelt {s['confidence']:.2f}",
                   (x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)
    for p in plates:
        x1,y1,x2,y2 = p['bbox']
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,255),2)
        if p['text']:
            cv2.putText(frame,p['text'],(x1,y2+20),
                       cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    return frame

# ── Routes ────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_image', methods=['POST'])
def detect_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image'})

        file  = request.files['image']
        img   = Image.open(file.stream).convert('RGB')
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        vehicles, persons, helmets, seatbelts, plates = detect_all(frame)
        frame = draw_frame(frame, vehicles, persons, helmets, seatbelts, plates)

        challan_done   = set()
        all_violations = []
        challan_ids    = []
        challan_pdfs   = []
        processed_veh  = set()

        for i, v in enumerate(vehicles):
            # Duplicate vehicle check via bbox overlap
            vkey = f"{v['bbox'][0]//50}_{v['bbox'][1]//50}"
            if vkey in processed_veh:
                continue
            processed_veh.add(vkey)

            violations = check_vehicle_violations(v, persons, helmets, seatbelts)

            if violations:
                plate_text = get_plate_near(plates, v['bbox'])
                key = plate_text if plate_text else \
                      f"UNKNOWN-{v['class'].upper()}-{i}-{datetime.now().strftime('%H%M%S')}"

                all_violations += violations

                result = make_challan(key, violations, challan_done)
                if result:
                    challan_data, pdf_path = result
                    challan_ids.append(challan_data['challan_id'])
                    challan_pdfs.append(pdf_path)

                # Mark violation on frame
                x1,y1,x2,y2 = v['bbox']
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
                cv2.putText(frame,"VIOLATION!",(x1,y1-25),
                           cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        _, buffer = cv2.imencode('.jpg', frame)
        img_b64   = base64.b64encode(buffer).decode('utf-8')

        seen     = set()
        unique_v = [v for v in all_violations
                    if v['description'] not in seen
                    and not seen.add(v['description'])]

        all_plates = list(set([
            get_plate_near(plates, v['bbox'])
            for v in vehicles
            if get_plate_near(plates, v['bbox'])
        ]))

        return jsonify({
            'image':         img_b64,
            'vehicles':      len(processed_veh),
            'persons':       len(persons),
            'plate':         ', '.join(all_plates) if all_plates else '',
            'vehicle_types': list(set([v['class'] for v in vehicles])),
            'pdf_path':      challan_pdfs[0] if challan_pdfs else None
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/detect_video', methods=['POST'])
def detect_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video'})

        file      = request.files['video']
        temp_path = os.path.join(OUTPUT_DIR, 'temp_input.mp4')
        file.save(temp_path)

        cap  = cv2.VideoCapture(temp_path)
        fps  = cap.get(cv2.CAP_PROP_FPS) or 30
        w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Process every Nth frame based on FPS
        # Low FPS video → process more frames
        # High FPS video → skip more
        if fps <= 15:
            SKIP = 5
        elif fps <= 24:
            SKIP = 8
        else:
            SKIP = 12

        out_path = os.path.join(OUTPUT_DIR, 'result_video.mp4')
        out = cv2.VideoWriter(out_path,
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (w, h))

        # Speed tracker — calibrated per video
        speed_tracker      = SpeedTracker()
        # Set speed lines at 20% and 80% of frame height for better coverage
        speed_tracker.LINE1_Y = int(h * 0.20)
        speed_tracker.LINE2_Y = int(h * 0.80)
        # Real distance between lines ~10m
        speed_tracker.REAL_DISTANCE = 10

        wrong_way_detector = WrongWayDetector()
        all_violations     = []
        frame_count        = 0
        challan_done       = set()
        challan_pdfs       = []

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1

            if frame_count % SKIP != 0:
                out.write(frame)
                continue

            vehicles, persons, helmets, seatbelts, plates = detect_all(frame)
            frame = draw_frame(frame, vehicles, persons, helmets, seatbelts, plates)

            # Traffic light
            lights       = traffic_light_detector.detect(frame)
            is_red_light = traffic_light_detector.is_red(lights)

            processed_veh = set()

            for i, v in enumerate(vehicles):
                vkey = f"{v['bbox'][0]//50}_{v['bbox'][1]//50}"
                if vkey in processed_veh:
                    continue
                processed_veh.add(vkey)

                violations = check_vehicle_violations(v, persons, helmets, seatbelts)

                # Red light
                if is_red_light:
                    rl = check_red_light_violation(is_red_light, [v])
                    if rl: violations.append(rl[0])

                # Speed — position-based tracking
                speed = speed_tracker.update(v['bbox'])
                speeding, spd = speed_tracker.is_speeding(v['bbox'])
                if speeding:
                    sv = check_speed_violation(spd, limit=60)
                    if sv: violations.append(sv[0])
                    print(f"SPEED VIOLATION DETECTED: {spd} km/h")

                # Debug speed tracking
                print(f"Vehicle {i}: bbox={v['bbox']}, speed={speed}, stored_speeds={list(speed_tracker.vehicle_speeds.values())}")

                # Wrong way
                is_wrong = wrong_way_detector.update(i, v['bbox'])
                if is_wrong:
                    wv = check_wrong_way_violation(is_wrong)
                    if wv: violations.append(wv[0])
                    frame = wrong_way_detector.draw_direction(
                        frame, i, v['bbox'], is_wrong)

                # No parking
                is_parked, duration = no_parking_detector.update(i, v['bbox'])
                if is_parked:
                    pv = check_no_parking_violation(is_parked, duration)
                    if pv: violations.append(pv[0])

                # Speed display - show speed for this position
                key = speed_tracker.get_position_key(v['bbox'])
                current_speed = speed_tracker.vehicle_speeds.get(key, 0)
                if current_speed > 0:
                    x1,y1,x2,y2 = v['bbox']
                    spd_color = (0,0,255) if current_speed > 60 else (0,200,0)
                    cv2.putText(frame,f"{current_speed:.0f}km/h",
                               (x1,y2+20),cv2.FONT_HERSHEY_SIMPLEX,
                               0.6,spd_color,2)

                if violations:
                    plate_text = get_plate_near(plates, v['bbox'])
                    key = plate_text if plate_text else \
                          f"UNKNOWN-{v['class'].upper()}-{i}"
                    all_violations += violations

                    result = make_challan(key, violations, challan_done)
                    if result:
                        _, pdf_path = result
                        challan_pdfs.append(pdf_path)

                    x1,y1,x2,y2 = v['bbox']
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
                    cv2.putText(frame,"VIOLATION!",
                               (x1,y1-25),cv2.FONT_HERSHEY_SIMPLEX,
                               0.7,(0,0,255),2)

            # Speed lines
            frame = speed_tracker.draw_lines(frame)

            # No parking zones
            frame = no_parking_detector.draw_zones(frame)

            # Red light indicator
            rl_color = (0,0,255) if is_red_light else (0,200,0)
            cv2.circle(frame,(50,50),18,rl_color,-1)
            cv2.putText(frame,"RED!" if is_red_light else "OK",
                       (76,58),cv2.FONT_HERSHEY_SIMPLEX,0.65,rl_color,2)

            # FPS info on frame
            cv2.putText(frame,f"FPS:{fps:.0f} Skip:{SKIP}",
                       (w-150,30),cv2.FONT_HERSHEY_SIMPLEX,
                       0.5,(200,200,200),1)

            out.write(frame)

        cap.release()
        out.release()

        seen   = set()
        unique = []
        for v in all_violations:
            if v['description'] not in seen:
                seen.add(v['description'])
                unique.append({'desc':v['description'],'fine':v['fine']})

        return jsonify({
            'frames':           frame_count,
            'plates':           list(challan_done),
            'total_violations': len(all_violations),
            'challans':         len(challan_done),
            'violations':       unique,
            'video_path':       '/output_video',
            'pdf_paths':        challan_pdfs
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/output_video')
def output_video():
    return send_file(os.path.join(OUTPUT_DIR,'result_video.mp4'),
                    mimetype='video/mp4')

@app.route('/dashboard_data')
def dashboard_data():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("SELECT COUNT(*) FROM challans")
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM challans WHERE status='UNPAID'")
    unpaid = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM challans WHERE status='PAID'")
    paid = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM challans WHERE status='WAIVED'")
    waived = c.fetchone()[0]
    c.execute("SELECT SUM(fine_amount) FROM challans")
    total_fine = c.fetchone()[0] or 0
    c.execute("SELECT SUM(fine_amount) FROM challans WHERE status='UNPAID'")
    pending = c.fetchone()[0] or 0
    c.execute("SELECT SUM(fine_amount) FROM challans WHERE status='PAID'")
    collected = c.fetchone()[0] or 0
    c.execute('''SELECT challan_id,vehicle_number,violation_type,
               fine_amount,timestamp,status,payment_mode,
               payment_date,remarks FROM challans
               ORDER BY timestamp DESC''')
    rows = c.fetchall()
    conn.close()
    return jsonify({
        'total':total,'unpaid':unpaid,'paid':paid,'waived':waived,
        'total_fine':total_fine,'pending':pending,'collected':collected,
        'challans':[{
            'id':r[0],'vehicle':r[1],'violation':r[2],
            'fine':r[3],'time':r[4],'status':r[5],
            'payment_mode':r[6] or '','payment_date':r[7] or '',
            'remarks':r[8] or ''
        } for r in rows]
    })

@app.route('/update_challan', methods=['POST'])
def update_challan():
    try:
        data = request.json
        update_challan_status(
            data['challan_id'], data['status'],
            data.get('payment_mode'), data.get('remarks')
        )
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/delete_challan/<challan_id>', methods=['DELETE'])
def delete_challan_route(challan_id):
    try:
        delete_challan(challan_id)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/search_challans')
def search_challans():
    query  = request.args.get('q','')
    status = request.args.get('status','ALL')
    conn   = sqlite3.connect(DB_PATH)
    c      = conn.cursor()
    if status == 'ALL':
        c.execute('''SELECT challan_id,vehicle_number,violation_type,
                   fine_amount,timestamp,status,payment_mode,
                   payment_date,remarks FROM challans
                   WHERE vehicle_number LIKE ? OR challan_id LIKE ?
                   ORDER BY timestamp DESC''',
                  (f'%{query}%',f'%{query}%'))
    else:
        c.execute('''SELECT challan_id,vehicle_number,violation_type,
                   fine_amount,timestamp,status,payment_mode,
                   payment_date,remarks FROM challans
                   WHERE (vehicle_number LIKE ? OR challan_id LIKE ?)
                   AND status=? ORDER BY timestamp DESC''',
                  (f'%{query}%',f'%{query}%',status))
    rows = c.fetchall()
    conn.close()
    return jsonify({'challans':[{
        'id':r[0],'vehicle':r[1],'violation':r[2],
        'fine':r[3],'time':r[4],'status':r[5],
        'payment_mode':r[6] or '','payment_date':r[7] or '',
        'remarks':r[8] or ''
    } for r in rows]})

@app.route('/export_csv')
def export_csv():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute('SELECT * FROM challans ORDER BY timestamp DESC')
    rows = c.fetchall()
    conn.close()
    csv_path = os.path.join(OUTPUT_DIR,'challans_export.csv')
    with open(csv_path,'w',newline='',encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ID','Challan ID','Vehicle','Violation',
                        'Fine','Location','Time','PDF',
                        'Status','Payment Mode','Payment Date','Remarks'])
        writer.writerows(rows)
    return send_file(csv_path,as_attachment=True,
                    download_name='challans_export.csv')

@app.route('/download_pdf/<challan_id>')
def download_pdf(challan_id):
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("SELECT pdf_path FROM challans WHERE challan_id=?",
              (challan_id,))
    row  = c.fetchone()
    conn.close()
    if row and os.path.exists(row[0]):
        return send_file(row[0],as_attachment=True,
                        download_name=f'challan_{challan_id}.pdf')
    return jsonify({'error':'PDF not found'})

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5000)