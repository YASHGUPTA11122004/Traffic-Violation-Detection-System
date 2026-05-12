from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')

    model.train(
        data=r'D:\traffic_violation_system\datasets\seatbelt\data.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        project=r'D:\traffic_violation_system\models',
        name='seatbelt_model',
        patience=10,
        workers=0
    )

    print("Seatbelt model training complete!")