from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback
import wandb
import os

# Step 1: Initialize a Weights & Biases run
wandb.init(project="ultralytics", job_type="training")

# Step 2: Define the YOLOv8 Model and Dataset
model_name = "yolov8n"
data_path = "/home/tungks/shared_drive_cv/GTTMData/external/traffic_camera"
dataset_name = os.path.join(data_path, "data.yaml")
model = YOLO(f"{model_name}.pt")

# Step 3: Add W&B Callback for Ultralytics
add_wandb_callback(model, enable_model_checkpointing=True)

# Step 4: Train and Fine-Tune the Model
results = model.train(project="ultralytics", data=dataset_name, epochs=5, device=0)

# Step 7: Finalize the W&B Run
wandb.finish()