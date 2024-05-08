from src.object_detection.model import ObjectDetection

model = ObjectDetection('yolo', 'sample_cfg.yaml')
model.train()