from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback
import wandb
import os


# Step 1: Initialize a Weights & Biases run
wandb.init(project="ultralytics", job_type="training")

# Step 2: Define the YOLOv8 Model and Dataset
model_name = "yolov8n"
home_dir = os.path.expanduser('~')

cfg_path = f'{home_dir}/gttm/sample_cfg.yaml'

model = YOLO('yolov8n.pt')
cfg = yaml_to_dict(cfg_path)

# Step 3: Add W&B Callback for Ultralytics
add_wandb_callback(model, enable_model_checkpointing=True)

# Step 4: Train and Fine-Tune the Model
results = model.train(**cfg)

# Step 7: Finalize the W&B Run
wandb.finish()
