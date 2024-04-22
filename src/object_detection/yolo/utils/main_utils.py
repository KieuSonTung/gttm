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


def merge_dicts(dict1, dict2):
    """
    Merge two dictionaries, prioritizing values from dict2 in case of key conflicts.

    Args:
    - dict1 (dict): The first dictionary.
    - dict2 (dict): The second dictionary.

    Returns:
    - merged_dict (dict): The merged dictionary.
    """
    merged_dict = dict1.copy()
    merged_dict.update(dict2)
    return merged_dict