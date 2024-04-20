from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback
import wandb
import os
from utils.main_utils import yaml_to_dict


# Step 1: Initialize a Weights & Biases run
wandb.init(project="ultralytics", job_type="training")

# Step 2: Define the Model and Config
home_dir = os.path.expanduser('~')

cfg_path = f'{home_dir}/gttm/sample_cfg.yaml'
cfg = yaml_to_dict(cfg_path)

model = YOLO(cfg['model'])

# Step 3: Add W&B Callback for Ultralytics
add_wandb_callback(model, enable_model_checkpointing=True)

# Step 4: Train and Fine-Tune the Model
results = model.train(**cfg)

# Step 7: Finalize the W&B Run
wandb.finish()