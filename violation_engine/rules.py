from datetime import datetime
import random

# Fine rules
FINE_RULES = {
    'NHelmet': {'description': 'No Helmet', 'fine': 500},
    'no_seatbelt': {'description': 'No Seatbelt', 'fine': 500},
    'red_light': {'description': 'Red Light Violation', 'fine': 1000},
    'speeding': {'description': 'Over Speeding', 'fine': 1500},
}

def generate_challan_id():
    return f"CH{datetime.now().strftime('%Y%m%d%H%M%S')}{random.randint(100,999)}"

def check_helmet_violation(helmet_detections):
    violations = []
    for h in helmet_detections:
        if h['class'] == 'NHelmet' and h['confidence'] > 0.5:
            violations.append({
                'type': 'NHelmet',
                'description': FINE_RULES['NHelmet']['description'],
                'fine': FINE_RULES['NHelmet']['fine'],
                'confidence': h['confidence'],
                'bbox': h['bbox']
            })
    return violations

def check_seatbelt_violation(seatbelt_detections):
    # Agar koi seatbelt detect nahi hua = violation
    violations = []
    if len(seatbelt_detections) == 0:
        violations.append({
            'type': 'no_seatbelt',
            'description': FINE_RULES['no_seatbelt']['description'],
            'fine': FINE_RULES['no_seatbelt']['fine'],
            'confidence': 1.0,
            'bbox': None
        })
    return violations

def get_total_fine(violations):
    return sum(v['fine'] for v in violations)

# Speed violation
FINE_RULES['speeding'] = {'description': 'Over Speeding', 'fine': 1500}
FINE_RULES['red_light'] = {'description': 'Red Light Violation', 'fine': 1000}

def check_speed_violation(speed_kmh, limit=60):
    violations = []
    if speed_kmh > limit:
        violations.append({
            'type': 'speeding',
            'description': f'Over Speeding ({speed_kmh} km/h)',
            'fine': FINE_RULES['speeding']['fine'],
            'confidence': 1.0,
            'bbox': None
        })
    return violations

def check_red_light_violation(is_red, vehicles_detected):
    violations = []
    if is_red and len(vehicles_detected) > 0:
        violations.append({
            'type': 'red_light',
            'description': 'Red Light Violation',
            'fine': FINE_RULES['red_light']['fine'],
            'confidence': 1.0,
            'bbox': None
        })
    return violations

FINE_RULES['wrong_way'] = {'description': 'Wrong Way Driving', 'fine': 2000}

def check_wrong_way_violation(is_wrong):
    violations = []
    if is_wrong:
        violations.append({
            'type': 'wrong_way',
            'description': FINE_RULES['wrong_way']['description'],
            'fine': FINE_RULES['wrong_way']['fine'],
            'confidence': 1.0,
            'bbox': None
        })
    return violations

FINE_RULES['no_parking'] = {'description': 'No Parking Violation', 'fine': 300}

def check_no_parking_violation(is_parked, duration):
    violations = []
    if is_parked:
        violations.append({
            'type': 'no_parking',
            'description': f'No Parking Violation ({int(duration)}s)',
            'fine': FINE_RULES['no_parking']['fine'],
            'confidence': 1.0,
            'bbox': None
        })
    return violations

FINE_RULES['triple_riding'] = {'description': 'Triple Riding', 'fine': 1000}

def check_triple_riding_violation(triple_violations):
    violations = []
    for t in triple_violations:
        violations.append({
            'type': 'triple_riding',
            'description': f"Triple Riding ({t['count']} persons)",
            'fine': FINE_RULES['triple_riding']['fine'],
            'confidence': 1.0,
            'bbox': t['bbox']
        })
    return violations