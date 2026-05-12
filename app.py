import gradio as gr
import cv2
import numpy as np
import sys
import os
from datetime import datetime
from PIL import Image
import sqlite3
import pandas as pd

sys.path.append(r'D:\traffic_violation_system')

from detection.vehicle_detector import VehicleDetector
from detection.helmet_detector import HelmetDetector
from detection.seatbelt_detector import SeatbeltDetector
from detection.plate_detector import PlateDetector
from detection.triple_riding import TripleRidingDetector
from violation_engine.rules import (
    check_helmet_violation,
    check_seatbelt_violation,
    check_triple_riding_violation,
    generate_challan_id,
    get_total_fine
)
from challan.generator import generate_challan_pdf
from database.db import save_challan
from database.models import create_tables, DB_PATH

# Models Load
print("Models load ho rahe hain...")
create_tables()
vehicle_detector  = VehicleDetector()
helmet_detector   = HelmetDetector()
seatbelt_detector = SeatbeltDetector()
plate_detector    = PlateDetector()
triple_detector   = TripleRidingDetector()
print("Sab models ready!")

MIN_CONF = 0.45

# ── Helper ────────────────────────────────────────────────
def detect_frame(frame):
    vehicles, persons = vehicle_detector.detect(frame)
    helmets           = helmet_detector.detect(frame)
    seatbelts         = seatbelt_detector.detect(frame)
    plates            = plate_detector.detect(frame)

    vehicle_type = 'unknown'
    types = [v['class'] for v in vehicles]
    if 'motorcycle' in types:     vehicle_type = 'motorcycle'
    elif 'car'      in types:     vehicle_type = 'car'
    elif 'truck'    in types:     vehicle_type = 'truck'
    elif 'bus'      in types:     vehicle_type = 'bus'

    violations = []
    if vehicle_type == 'motorcycle':
        violations += check_helmet_violation(helmets)
    if vehicle_type in ['car', 'truck', 'bus']:
        violations += check_seatbelt_violation(seatbelts)

    triple_v, frame = triple_detector.detect(vehicles, persons, frame)
    violations += check_triple_riding_violation(triple_v)

    plate_text = ''
    for p in plates:
        if len(p['text']) >= 4:
            plate_text = p['text'].strip()
            break

    for v in vehicles:
        x1,y1,x2,y2 = v['bbox']
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
        cv2.putText(frame,v['class'],(x1,y1-8),
                   cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,0,0),2)

    for p in persons:
        x1,y1,x2,y2 = p['bbox']
        cv2.rectangle(frame,(x1,y1),(x2,y2),(200,200,200),1)

    for h in helmets:
        if h['confidence'] < MIN_CONF: continue
        x1,y1,x2,y2 = h['bbox']
        color = (0,255,0) if h['class']=='Helmet' else (0,0,255)
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
            cv2.putText(frame,p['text'],(x1,y2+22),
                       cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

    return frame, vehicles, persons, violations, plate_text

# ── Tab 1: Image ──────────────────────────────────────────
def detect_image(image):
    if image is None:
        return None, "❌ Koi image upload nahi ki!", None

    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    frame, vehicles, persons, violations, plate_text = detect_frame(frame)

    lines = []
    lines.append(f"🚗 Vehicles Detected:  {len(vehicles)}")
    lines.append(f"👤 Persons Detected:   {len(persons)}")
    lines.append(f"🔢 Number Plate:       {plate_text if plate_text else 'Not detected'}")
    lines.append(f"🏍️  Vehicle Type:       {list(set([v['class'] for v in vehicles]))}")
    lines.append("")

    if violations:
        lines.append("🚨 VIOLATIONS FOUND:")
        total = 0
        for v in violations:
            lines.append(f"  ❌ {v['description']} → ₹{v['fine']}")
            total += v['fine']
        lines.append(f"\n💰 TOTAL FINE: ₹{total}")
    else:
        lines.append("✅ No Violations Detected!")

    pdf_path = None
    if violations and plate_text:
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
        lines.append(f"\n📄 Challan Generated!")

    out_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(out_image), "\n".join(lines), pdf_path

# ── Tab 2: Video ──────────────────────────────────────────
def detect_video(video_path):
    if video_path is None:
        return None, "❌ Koi video upload nahi ki!"

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = r'D:\traffic_violation_system\output\result_video.mp4'
    out = cv2.VideoWriter(out_path,
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         fps, (w, h))

    all_violations = []
    all_plates     = set()
    frame_count    = 0
    challan_done   = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 != 0:
            out.write(frame)
            continue

        frame, vehicles, persons, violations, plate_text = detect_frame(frame)

        if violations:
            all_violations += violations
        if plate_text:
            all_plates.add(plate_text)

        if violations and plate_text and plate_text not in challan_done:
            challan_done.add(plate_text)
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

        out.write(frame)

    cap.release()
    out.release()

    lines = []
    lines.append(f"📹 Total Frames:       {frame_count}")
    lines.append(f"🔢 Plates Detected:    {', '.join(all_plates) if all_plates else 'None'}")
    lines.append(f"🚨 Total Violations:   {len(all_violations)}")
    lines.append(f"📄 Challans Generated: {len(challan_done)}")
    lines.append("")

    if all_violations:
        seen = set()
        lines.append("Violations Found:")
        for v in all_violations:
            if v['description'] not in seen:
                seen.add(v['description'])
                lines.append(f"  ❌ {v['description']} → ₹{v['fine']}")

    return out_path, "\n".join(lines)

# ── Tab 3: Dashboard ──────────────────────────────────────
def get_dashboard():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM challans")
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM challans WHERE status='UNPAID'")
    unpaid = c.fetchone()[0]
    c.execute("SELECT SUM(fine_amount) FROM challans")
    total_fine = c.fetchone()[0] or 0
    c.execute("SELECT SUM(fine_amount) FROM challans WHERE status='UNPAID'")
    pending = c.fetchone()[0] or 0

    stats = f"""
## 🚔 Traffic Police Department — Dashboard

| Metric | Value |
|--------|-------|
| Total Challans | {total} |
| Unpaid | {unpaid} |
| Paid | {total - unpaid} |
| Total Fine | ₹{total_fine:,} |
| Pending Fine | ₹{pending:,} |
| Collected | ₹{total_fine - pending:,} |
"""
    try:
        df = pd.read_sql_query(
            """SELECT challan_id, vehicle_number,
               violation_type, fine_amount,
               timestamp, status
               FROM challans
               ORDER BY timestamp DESC""",
            conn
        )
    except:
        df = pd.DataFrame()

    conn.close()
    return stats, df

# ── Gradio UI ─────────────────────────────────────────────
with gr.Blocks(
    title="Traffic Violation Detection System",
    theme=gr.themes.Base(
        primary_hue="blue",
        secondary_hue="orange"
    )
) as app:

    gr.Markdown("""
    # 🚔 Traffic Violation Detection & E-Challan System
    ### AI-Powered | Real-time Detection | Auto Challan Generation
    ---
    """)

    with gr.Tabs():

        with gr.TabItem("📷 Image Detection"):
            gr.Markdown("### Upload karo koi bhi traffic image — violations detect honge!")
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(
                        label="📤 Image Upload karo",
                        type="pil"
                    )
                    img_btn = gr.Button(
                        "🔍 Detect Violations",
                        variant="primary",
                        size="lg"
                    )
                with gr.Column():
                    img_output = gr.Image(label="📊 Detection Result")
                    img_text   = gr.Textbox(
                        label="📋 Violation Report",
                        lines=12
                    )
                    img_pdf = gr.File(label="📄 Download Challan PDF")

            img_btn.click(
                fn=detect_image,
                inputs=[img_input],
                outputs=[img_output, img_text, img_pdf]
            )

        with gr.TabItem("🎥 Video Detection"):
            gr.Markdown("### Upload karo traffic video — frame by frame analyze hoga!")
            with gr.Row():
                with gr.Column():
                    vid_input = gr.Video(label="📤 Video Upload karo")
                    vid_btn   = gr.Button(
                        "🔍 Detect Violations",
                        variant="primary",
                        size="lg"
                    )
                    gr.Markdown("""
                    **Detect hoga:**
                    - 🚗 Vehicles
                    - ⛑️ Helmet / No Helmet
                    - 🪑 Seatbelt
                    - 👥 Triple Riding
                    - 🔢 Number Plate
                    """)
                with gr.Column():
                    vid_output = gr.Video(label="📊 Annotated Video")
                    vid_text   = gr.Textbox(
                        label="📋 Summary",
                        lines=12
                    )

            vid_btn.click(
                fn=detect_video,
                inputs=[vid_input],
                outputs=[vid_output, vid_text]
            )

        with gr.TabItem("📊 Dashboard"):
            refresh_btn = gr.Button("🔄 Refresh Data", variant="secondary")
            stats_md    = gr.Markdown()
            challan_df  = gr.Dataframe(
                label="📋 All Challans",
                interactive=False
            )
            refresh_btn.click(
                fn=get_dashboard,
                outputs=[stats_md, challan_df]
            )
            app.load(
                fn=get_dashboard,
                outputs=[stats_md, challan_df]
            )

if __name__ == '__main__':
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )