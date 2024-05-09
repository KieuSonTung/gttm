# from src.object_detection.model import ObjectDetection

# model = ObjectDetection('yolo', 'sample_cfg.yaml')
# model.train()

from src.main import run

results = run('/content/Untitled video - Made with Clipchamp.mp4')
print(results)