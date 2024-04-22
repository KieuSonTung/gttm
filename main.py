from src.object_detection.model import ObjectDetection

model = ObjectDetection(model_name='yolo', cfg_path='sample_cfg.yaml')
# model.train()
results = model.infer('runs/detect/train/weights/best.pt')

print(results)