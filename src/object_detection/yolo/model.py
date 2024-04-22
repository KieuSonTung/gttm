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
        '''

        # load custom parameter
        pred_cfg = self.cfg.copy()
        pred_cfg.pop('source', None)
        pred_cfg.pop('conf', None)
        pred_cfg.pop('save', None)

        # load model
        model = YOLO(model_path)

        # load video
        video_capture = cv2.VideoCapture(self.cfg['source'])
        
        # Check if the video file is opened successfully
        if not video_capture.isOpened():
            print("Error: Unable to open video file")
        
        # Initialize variables
        frame_count = 0
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        
        # Read the video frame by frame
        while True:
            
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count * fps)
            # Read a single frame
            ret, frame = video_capture.read()
            
            # Break the loop if there are no more frames
            if not ret:
                break
        
            # print(f'frame {frame_count} read succeeded')
            # cv2.imwrite(f'test_img/my_video_frame_{frame_count}.png', frame)
        
            results = model.predict(frame, mode='predict', save=False)
        
            tensor = results[0].boxes.data
            zeros_column = torch.zeros(tensor.shape[0], 1, device=tensor.device)
            zeros_column.fill_(frame_count)
            new_tensor = torch.cat((zeros_column, tensor), dim=1)
        
            # concat output tensor of each frame
            if frame_count == 0:
                stacked_tensor = new_tensor
            else:
                stacked_tensor = torch.cat((stacked_tensor, new_tensor), dim=0)
                        
            # Increment frame count
            frame_count += 1
        
        # Release the video capture object
        video_capture.release()

        return stacked_tensor