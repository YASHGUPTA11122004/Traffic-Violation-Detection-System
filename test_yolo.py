from ultralytics import YOLO
import cv2

# YOLOv8 pre-trained model load karo
model = YOLO('yolov8n.pt')

# Video file se input lo
cap = cv2.VideoCapture(r'D:\traffic_violation_system\videos\traffic.mp4')

print("Video chal raha hai... 'q' dabao band karne ke liye")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video khatam!")
        break

    # Detection karo
    results = model(frame, verbose=False)
    
    # Frame pe boxes draw karo
    annotated = results[0].plot()
    
    # Show karo
    cv2.imshow('Traffic Detection - Test', annotated)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done!")