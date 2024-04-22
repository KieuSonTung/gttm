from src.object_detection.yolo.utils.main_utils import read_yaml_file, merge_dicts
from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback
import cv2
import torch


class YOLOObjectDetection():
    def __init__(self, cfg_path: str) -> None:
        self.cfg = read_yaml_file(cfg_path)
    
    def train(self):
        model = YOLO('yolov8n.pt')

        # add W&B Callback for Ultralytics
        add_wandb_callback(model, enable_model_checkpointing=True)

        # train and Fine-Tune the Model
        results = model.train(mode='train', **self.cfg)

        # finalize the W&B Run
        wandb.finish()
    
    def infer(self, model_path: str):
        '''
        model_path (str): path to pretrained model (.pt)
        source (str): source directory for images or videos
        '''
        model = YOLO(model_path)
        cap = cv2.VideoCapture(self.cfg['source'])
        pred_cfg = self.cfg.copy()
        pred_cfg.pop('source', None)
        pred_cfg.pop('conf', None)

        i = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            
            # if i > 5:
            #     break
                
            if success:
                # print(f'frame {i} read succeeded')
                # cv2.imwrite(f'my_video_frame_{i}.png', frame)

                # predict
                results = model.predict(frame, mode='predict', **pred_cfg)

                tensor = results[0].boxes.data
                zeros_column = torch.zeros(tensor.shape[0], 1, device=tensor.device)
                zeros_column.fill_(i)
                new_tensor = torch.cat((zeros_column, tensor), dim=1)

                # concat output tensor of each frame
                if i == 0:
                    stacked_tensor = new_tensor
                else:
                    stacked_tensor = torch.cat((stacked_tensor, new_tensor), dim=0)
                
                i += 1

        return stacked_tensor