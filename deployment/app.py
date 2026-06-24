import os
import io
import sys
from huggingface_hub import hf_hub_download

# Download models from HuggingFace
print("Downloading models...")
os.makedirs("models", exist_ok=True)

hf_hub_download(
    repo_id="yashgupta11122004/traffic-violation-models",
    filename="helmet_best.pt",
    local_dir="models"
)
hf_hub_download(
    repo_id="yashgupta11122004/traffic-violation-models",
    filename="seatbelt_best.pt",
    local_dir="models"
)
hf_hub_download(
    repo_id="yashgupta11122004/traffic-violation-models",
    filename="numberplate_best.pt",
    local_dir="models"
)
print("Models downloaded!")

os.environ['HELMET_MODEL']   = "models/helmet_best.pt"
os.environ['SEATBELT_MODEL'] = "models/seatbelt_best.pt"
os.environ['PLATE_MODEL']    = "models/numberplate_best.pt"
os.environ['OUTPUT_DIR']     = "/app/output"

from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import base64
import csv
from datetime import datetime
from PIL import Image
import sqlite3

from ultralytics import YOLO
import easyocr

app = Flask(__name__)

DB_PATH    = "/app/output/violations.db"
OUTPUT_DIR = "/app/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_tables():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS challans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        challan_id TEXT UNIQUE,
        vehicle_number TEXT,
        violation_type TEXT,
        fine_amount INTEGER,
        location TEXT,
        timestamp TEXT,
        pdf_path TEXT,
        status TEXT DEFAULT 'UNPAID',
        payment_mode TEXT DEFAULT NULL,
        payment_date TEXT DEFAULT NULL,
        remarks TEXT DEFAULT NULL
    )''')
    for col in ['payment_mode','payment_date','remarks']:
        try:
            c.execute(f"ALTER TABLE challans ADD COLUMN {col} TEXT DEFAULT NULL")
        except: pass
    conn.commit()
    conn.close()

create_tables()

print("Loading models...")
vehicle_model   = YOLO('yolov8n.pt')
helmet_model    = YOLO(os.environ['HELMET_MODEL'])
seatbelt_model  = YOLO(os.environ['SEATBELT_MODEL'])
plate_model     = YOLO(os.environ['PLATE_MODEL'])
ocr_reader      = easyocr.Reader(['en'], gpu=False)
print("All models loaded!")

MIN_CONF = 0.35

def is_near(obj_bbox, veh_bbox, margin=40):
    ox1,oy1,ox2,oy2 = obj_bbox
    vx1,vy1,vx2,vy2 = veh_bbox
    ocx,ocy = (ox1+ox2)//2,(oy1+oy2)//2
    return (vx1-margin < ocx < vx2+margin and
            vy1-margin < ocy < vy2+margin)

def detect_all(frame):
    veh_res  = vehicle_model(frame, verbose=False, conf=0.35, iou=0.6)[0]
    helm_res = helmet_model(frame, verbose=False, conf=MIN_CONF)[0]
    seat_res = seatbelt_model(frame, verbose=False, conf=MIN_CONF)[0]
    plat_res = plate_model(frame, verbose=False, conf=MIN_CONF)[0]

    vehicles, persons = [], []
    for box in veh_res.boxes:
        cls  = int(box.cls[0])
        x1,y1,x2,y2 = map(int,box.xyxy[0])
        conf = float(box.conf[0])
        vc = {2:'car',3:'motorcycle',5:'bus',7:'truck'}
        if cls in vc:
            vehicles.append({'class':vc[cls],'confidence':conf,'bbox':(x1,y1,x2,y2)})
        elif cls == 0:
            persons.append({'class':'person','confidence':conf,'bbox':(x1,y1,x2,y2)})

    helmets = []
    for box in helm_res.boxes:
        cls = int(box.cls[0])
        x1,y1,x2,y2 = map(int,box.xyxy[0])
        helmets.append({'class':helmet_model.names[cls],
                        'confidence':float(box.conf[0]),'bbox':(x1,y1,x2,y2)})

    seatbelts = []
    for box in seat_res.boxes:
        x1,y1,x2,y2 = map(int,box.xyxy[0])
        seatbelts.append({'class':'seatbelt',
                          'confidence':float(box.conf[0]),'bbox':(x1,y1,x2,y2)})

    plates = []
    for box in plat_res.boxes:
        x1,y1,x2,y2 = map(int,box.xyxy[0])
        plate_img = frame[y1:y2,x1:x2]
        text = ''
        if plate_img.size > 0:
            ocr_res = ocr_reader.readtext(plate_img)
            for (_,t,c) in ocr_res:
                if c > 0.2: text += t + ' '
        plates.append({'bbox':(x1,y1,x2,y2),'text':text.strip()})

    return vehicles, persons, helmets, seatbelts, plates

def get_near(items, vbbox, margin=60):
    return [i for i in items if is_near(i['bbox'], vbbox, margin)]

def get_plate_near(plates, vbbox):
    for p in plates:
        if is_near(p['bbox'], vbbox, 40) and len(p['text']) >= 3:
            return p['text'].strip()
    return ''

def check_violations(vehicle, persons, helmets, seatbelts):
    vclass = vehicle['class']
    vbbox  = vehicle['bbox']
    violations = []

    if vclass == 'motorcycle':
        near_h = get_near(helmets, vbbox)
        no_helmet = any(h['class']=='NHelmet' and h['confidence']>=MIN_CONF
                        for h in near_h)
        if no_helmet or len(near_h) == 0:
            violations.append({'description':'No Helmet','fine':500})
        persons_on = sum(1 for p in persons if is_near(p['bbox'],vbbox,30))
        if persons_on >= 3:
            violations.append({'description':'Triple Riding','fine':1000})

    elif vclass in ['car','truck','bus']:
        near_s = get_near(seatbelts, vbbox)
        if len(near_s) == 0:
            violations.append({'description':'No Seatbelt','fine':500})

    seen = set()
    return [v for v in violations
            if v['description'] not in seen and not seen.add(v['description'])]

def draw_frame(frame, vehicles, persons, helmets, seatbelts, plates):
    for v in vehicles:
        x1,y1,x2,y2 = v['bbox']
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
        cv2.putText(frame,v['class'],(x1,y1-8),
                   cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
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

def generate_challan_id():
    import random
    return f"CH{datetime.now().strftime('%Y%m%d%H%M%S')}{random.randint(100,999)}"

def get_total_fine(violations):
    return sum(v['fine'] for v in violations)

def make_challan(key, violations, challan_done):
    if not violations or key in challan_done: return None
    challan_done.add(key)
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch

    challan_id = generate_challan_id()
    timestamp  = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
    total_fine = get_total_fine(violations)

    pdf_path = os.path.join(OUTPUT_DIR, f'challan_{challan_id}.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    header_style = ParagraphStyle('h', parent=styles['Title'],
                                  fontSize=20, textColor=colors.darkblue)
    elements.append(Paragraph("TRAFFIC POLICE DEPARTMENT", header_style))
    elements.append(Paragraph("E-Challan - Digital Traffic Violation Notice", styles['Normal']))
    elements.append(Spacer(1, 0.2*inch))

    info = [['Challan ID', challan_id],['Date & Time', timestamp],
            ['Location', 'Camera ID: CAM-01'],['Vehicle', key],['Status','UNPAID']]
    t = Table(info, colWidths=[2.5*inch, 4*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(0,-1),colors.lightblue),
        ('FONTNAME',(0,0),(-1,-1),'Helvetica-Bold'),
        ('GRID',(0,0),(-1,-1),0.5,colors.grey),
        ('PADDING',(0,0),(-1,-1),8),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.2*inch))

    vdata = [['#','Violation','Fine']]
    for i,v in enumerate(violations,1):
        vdata.append([str(i), v['description'], f"Rs. {v['fine']}"])
    vdata.append(['','TOTAL FINE', f"Rs. {total_fine}"])
    vt = Table(vdata, colWidths=[0.5*inch, 4*inch, 2*inch])
    vt.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.darkblue),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('BACKGROUND',(0,-1),(-1,-1),colors.lightcoral),
        ('FONTNAME',(0,0),(-1,-1),'Helvetica-Bold'),
        ('GRID',(0,0),(-1,-1),0.5,colors.grey),
        ('PADDING',(0,0),(-1,-1),8),
    ]))
    elements.append(vt)
    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph("Pay fine within 30 days.",styles['Normal']))

    doc.build(elements)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    vstr = ', '.join([v['description'] for v in violations])
    c.execute('''INSERT OR IGNORE INTO challans
        (challan_id,vehicle_number,violation_type,fine_amount,
         location,timestamp,pdf_path,status)
        VALUES (?,?,?,?,?,?,?,?)''',
        (challan_id,key,vstr,total_fine,'CAM-01',timestamp,pdf_path,'UNPAID'))
    conn.commit()
    conn.close()
    return challan_id, pdf_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_image', methods=['POST'])
def detect_image():
    try:
        file = request.files['image']
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'Image decode failed — invalid image file'})

        vehicles,persons,helmets,seatbelts,plates = detect_all(frame)
        frame = draw_frame(frame,vehicles,persons,helmets,seatbelts,plates)

        challan_done = set()
        all_violations = []
        challan_ids = []
        processed = set()

        for i,v in enumerate(vehicles):
            vkey = f"{v['bbox'][0]//50}_{v['bbox'][1]//50}"
            if vkey in processed: continue
            processed.add(vkey)

            violations = check_violations(v,persons,helmets,seatbelts)
            if violations:
                plate = get_plate_near(plates,v['bbox'])
                key = plate if plate else f"UNKNOWN-{v['class'].upper()}-{i}"
                all_violations += violations
                result = make_challan(key,violations,challan_done)
                if result: challan_ids.append(result[0])
                x1,y1,x2,y2 = v['bbox']
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)

        _,buffer = cv2.imencode('.jpg',frame)
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        seen = set()
        unique_v = [v for v in all_violations
                    if v['description'] not in seen
                    and not seen.add(v['description'])]

        return jsonify({
            'image': img_b64,
            'vehicles': len(processed),
            'persons': len(persons),
            'plate': ', '.join([get_plate_near(plates,v['bbox'])
                                for v in vehicles
                                if get_plate_near(plates,v['bbox'])]),
            'vehicle_types': list(set([v['class'] for v in vehicles])),
            'violations': [{'desc':v['description'],'fine':v['fine']}
                           for v in unique_v],
            'total_fine': get_total_fine(all_violations),
            'challan_id': challan_ids[0] if challan_ids else None,
            'total_challans': len(challan_ids)
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/detect_video', methods=['POST'])
def detect_video():
    try:
        file = request.files['video']
        temp = os.path.join(OUTPUT_DIR,'temp_input.mp4')
        file.save(temp)

        cap = cv2.VideoCapture(temp)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        SKIP = 5 if fps<=15 else 8 if fps<=24 else 12
        out_path = os.path.join(OUTPUT_DIR,'result_video.mp4')
        out = cv2.VideoWriter(out_path,
                             cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))

        all_violations = []
        frame_count = 0
        challan_done = set()

        while True:
            ret,frame = cap.read()
            if not ret: break
            frame_count += 1
            if frame_count % SKIP != 0:
                out.write(frame)
                continue

            vehicles,persons,helmets,seatbelts,plates = detect_all(frame)
            frame = draw_frame(frame,vehicles,persons,helmets,seatbelts,plates)

            processed = set()
            for i,v in enumerate(vehicles):
                vkey = f"{v['bbox'][0]//50}_{v['bbox'][1]//50}"
                if vkey in processed: continue
                processed.add(vkey)

                violations = check_violations(v,persons,helmets,seatbelts)
                if violations:
                    plate = get_plate_near(plates,v['bbox'])
                    key = plate if plate else f"UNKNOWN-{v['class'].upper()}-{i}"
                    all_violations += violations
                    make_challan(key,violations,challan_done)
                    x1,y1,x2,y2 = v['bbox']
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
                    cv2.putText(frame,"VIOLATION!",(x1,y1-25),
                               cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            out.write(frame)

        cap.release()
        out.release()

        seen = set()
        unique = []
        for v in all_violations:
            if v['description'] not in seen:
                seen.add(v['description'])
                unique.append({'desc':v['description'],'fine':v['fine']})

        return jsonify({
            'frames': frame_count,
            'plates': list(challan_done),
            'total_violations': len(all_violations),
            'challans': len(challan_done),
            'violations': unique,
            'video_path': '/output_video'
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/output_video')
def output_video():
    return send_file(os.path.join(OUTPUT_DIR,'result_video.mp4'), mimetype='video/mp4')

@app.route('/dashboard_data')
def dashboard_data():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM challans"); total=c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM challans WHERE status='UNPAID'"); unpaid=c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM challans WHERE status='PAID'"); paid=c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM challans WHERE status='WAIVED'"); waived=c.fetchone()[0]
    c.execute("SELECT SUM(fine_amount) FROM challans"); total_fine=c.fetchone()[0] or 0
    c.execute("SELECT SUM(fine_amount) FROM challans WHERE status='UNPAID'"); pending=c.fetchone()[0] or 0
    c.execute("SELECT SUM(fine_amount) FROM challans WHERE status='PAID'"); collected=c.fetchone()[0] or 0
    c.execute('''SELECT challan_id,vehicle_number,violation_type,fine_amount,
               timestamp,status,payment_mode,payment_date,remarks
               FROM challans ORDER BY timestamp DESC''')
    rows = c.fetchall()
    conn.close()
    return jsonify({
        'total':total,'unpaid':unpaid,'paid':paid,'waived':waived,
        'total_fine':total_fine,'pending':pending,'collected':collected,
        'challans':[{'id':r[0],'vehicle':r[1],'violation':r[2],
                     'fine':r[3],'time':r[4],'status':r[5],
                     'payment_mode':r[6] or '','payment_date':r[7] or '',
                     'remarks':r[8] or ''} for r in rows]
    })

@app.route('/update_challan', methods=['POST'])
def update_challan():
    try:
        data = request.json
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''UPDATE challans SET status=?,payment_mode=?,
                   payment_date=?,remarks=? WHERE challan_id=?''',
                  (data['status'],data.get('payment_mode'),
                   datetime.now().strftime('%d-%m-%Y %H:%M'),
                   data.get('remarks'),data['challan_id']))
        conn.commit()
        conn.close()
        return jsonify({'success':True})
    except Exception as e:
        return jsonify({'error':str(e)})

@app.route('/delete_challan/<challan_id>', methods=['DELETE'])
def delete_challan(challan_id):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('DELETE FROM challans WHERE challan_id=?',(challan_id,))
        conn.commit()
        conn.close()
        return jsonify({'success':True})
    except Exception as e:
        return jsonify({'error':str(e)})

@app.route('/search_challans')
def search_challans():
    q = request.args.get('q','')
    s = request.args.get('status','ALL')
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if s == 'ALL':
        c.execute('''SELECT challan_id,vehicle_number,violation_type,fine_amount,
                   timestamp,status,payment_mode,payment_date,remarks
                   FROM challans WHERE vehicle_number LIKE ? OR challan_id LIKE ?
                   ORDER BY timestamp DESC''',(f'%{q}%',f'%{q}%'))
    else:
        c.execute('''SELECT challan_id,vehicle_number,violation_type,fine_amount,
                   timestamp,status,payment_mode,payment_date,remarks
                   FROM challans WHERE (vehicle_number LIKE ? OR challan_id LIKE ?)
                   AND status=? ORDER BY timestamp DESC''',
                  (f'%{q}%',f'%{q}%',s))
    rows = c.fetchall()
    conn.close()
    return jsonify({'challans':[{'id':r[0],'vehicle':r[1],'violation':r[2],
                                 'fine':r[3],'time':r[4],'status':r[5],
                                 'payment_mode':r[6] or '','payment_date':r[7] or '',
                                 'remarks':r[8] or ''} for r in rows]})

@app.route('/export_csv')
def export_csv():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT * FROM challans ORDER BY timestamp DESC')
    rows = c.fetchall()
    conn.close()
    csv_path = os.path.join(OUTPUT_DIR,'challans_export.csv')
    with open(csv_path,'w',newline='',encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ID','Challan ID','Vehicle','Violation','Fine',
                        'Location','Time','PDF','Status','Payment Mode',
                        'Payment Date','Remarks'])
        writer.writerows(rows)
    return send_file(csv_path,as_attachment=True,download_name='challans_export.csv')

@app.route('/download_pdf/<challan_id>')
def download_pdf(challan_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT pdf_path FROM challans WHERE challan_id=?",(challan_id,))
    row = c.fetchone()
    conn.close()
    if row and os.path.exists(row[0]):
        return send_file(row[0],as_attachment=True,download_name=f'challan_{challan_id}.pdf')
    return jsonify({'error':'PDF not found'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)