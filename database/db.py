import sqlite3
from database.models import DB_PATH
from datetime import datetime

def save_challan(challan_data, pdf_path):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    violations_str = ', '.join([v['description'] for v in challan_data['violations']])
    c.execute('''INSERT OR IGNORE INTO challans
        (challan_id, vehicle_number, violation_type,
         fine_amount, location, timestamp, pdf_path, status)
        VALUES (?,?,?,?,?,?,?,?)''', (
        challan_data['challan_id'],
        challan_data['vehicle_number'],
        violations_str,
        challan_data['total_fine'],
        challan_data.get('location', 'CAM-01'),
        challan_data['timestamp'],
        pdf_path, 'UNPAID'
    ))
    conn.commit()
    conn.close()

def update_challan_status(challan_id, status, payment_mode=None, remarks=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''UPDATE challans SET status=?, payment_mode=?,
               payment_date=?, remarks=? WHERE challan_id=?''',
              (status, payment_mode,
               datetime.now().strftime('%d-%m-%Y %H:%M'),
               remarks, challan_id))
    conn.commit()
    conn.close()

def delete_challan(challan_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM challans WHERE challan_id=?', (challan_id,))
    conn.commit()
    conn.close()

def get_all_challans():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT * FROM challans ORDER BY timestamp DESC')
    rows = c.fetchall()
    conn.close()
    return rows