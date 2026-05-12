import sys
sys.path.append(r'D:\traffic_violation_system')

from violation_engine.rules import generate_challan_id
from challan.generator import generate_challan_pdf
from datetime import datetime

# Test challan data
challan_data = {
    'challan_id': generate_challan_id(),
    'timestamp': datetime.now().strftime('%d-%m-%Y %H:%M:%S'),
    'vehicle_number': 'UP80AB1234',
    'location': 'Camera ID: CAM-01, MG Road',
    'violations': [
        {
            'description': 'No Helmet',
            'fine': 500
        },
        {
            'description': 'No Seatbelt',
            'fine': 500
        }
    ],
    'total_fine': 1000
}

# Challan generate karo
pdf_path = generate_challan_pdf(challan_data)
print(f"Challan ready: {pdf_path}")