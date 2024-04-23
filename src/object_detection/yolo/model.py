from src.object_detection.yolo.utils.main_utils import read_yaml_file
from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback
import cv2
import torch
import os


class YOLOObjectDetection():
    def __init__(self, cfg_path: str) -> None:
        _, file_extension = os.path.splitext(cfg_path)

        if file_extension == ".pt":
            self.model = YOLO(cfg_path)
            self.cfg = {}
        elif file_extension in [".yaml", ".yml"]:
            self.cfg = read_yaml_file(cfg_path)
            self.model = YOLO('yolov8n.pt')
        
    
    def train(self):

        # add W&B Callback for Ultralytics
        add_wandb_callback(self.model, enable_model_checkpointing=True)

        # train and Fine-Tune the Model
        results = self.model.train(mode='train', **self.cfg)

        # finalize the W&B Run
        wandb.finish()
    
    def infer(self, frame, frame_count) -> torch.tensor:
        '''
        Infer by frame

        Params
        model_path (str): path to pretrained model (.pt)
        '''

        # load custom parameter
        pred_cfg = self.cfg.copy()
        pred_cfg.pop('source', None)
        pred_cfg.pop('conf', None)
        pred_cfg.pop('save', None)

        # load model
        results = self.model.predict(frame, mode='predict', save=False, **pred_cfg)
    
        tensor = results[0].boxes.data
        zeros_column = torch.zeros(tensor.shape[0], 1, device=tensor.device)
        zeros_column.fill_(frame_count)
        output = torch.cat((zeros_column, tensor), dim=1)
    
        return output