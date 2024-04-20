from utils.main_utils import yaml_to_dict
from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback


class YOLOObjectDetection():
    def __init__(self, cfg_path) -> None:
        self.cfg = yaml_to_dict(cfg_path)
    
    def train(self):
        model = YOLO(self.cfg['model'])

        # Step 3: Add W&B Callback for Ultralytics
        add_wandb_callback(model, enable_model_checkpointing=True)

        # Step 4: Train and Fine-Tune the Model
        results = model.train(**self.cfg)

        # Step 7: Finalize the W&B Run
        wandb.finish()
    
    def infer(self):
        pass