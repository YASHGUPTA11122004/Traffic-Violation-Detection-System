import sqlite3

DB_PATH = r'D:\traffic_violation_system\database\violations.db'

def create_tables():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS challans (
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
        )
    ''')
    for col in ['payment_mode', 'payment_date', 'remarks']:
        try:
            c.execute(f"ALTER TABLE challans ADD COLUMN {col} TEXT DEFAULT NULL")
        except: pass
    conn.commit()
    conn.close()
    print("Database ready!")

if __name__ == '__main__':
    create_tables()