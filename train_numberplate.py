from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')

    model.train(
        data=r'D:\traffic_violation_system\datasets\number_plate\data.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        project=r'D:\traffic_violation_system\models',
        name='numberplate_model',
        patience=10,
        workers=0
    )

    print("Number plate model training complete!")