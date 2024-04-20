import yaml
import sys
from src.object_detection.yolo.exception import AppException
from src.object_detection.yolo.logger import logging


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