import yaml
import sys
from src.object_detection.yolo.exception import AppException
from src.object_detection.yolo.logger import logging
from pathlib import Path
import cv2


def read_yaml_file(file_path: str) -> dict:
    """
    Convert YAML file to dictionary.

    Args:
    - yaml_file (str): Path to the YAML file

    Returns:
    - dict: Dictionary containing YAML data
    """
    try:
        with open(file_path, "rb") as yaml_file:
            logging.info("Read yaml file successfully")
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise AppException(e, sys) from e


def load_video_to_list(path: Path) -> list:
    video_capture = cv2.VideoCapture(path)
    frame_ls = []

    if not video_capture.isOpened():
        print("Error: Unable to open video file")
        return 
    
    while True:
        ret, frame = video_capture.read()
            
        # Break the loop if there are no more frames
        if not ret:
            break

        frame_ls.append(frame)
    
    video_capture.release()

    return frame_ls