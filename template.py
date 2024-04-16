import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


project_name = "object-detection"

list_of_files = [
    # ".github/workflows/.gitkeep",
    # "data/.gitkeep",
    f"src/{project_name}/yolo/__init__.py",
    f"src/{project_name}/yolo/components/__init__.py",
    f"src/{project_name}/yolo/components/data_ingestion.py",
    f"src/{project_name}/yolo/components/data_validation.py",
    f"src/{project_name}/yolo/components/model_trainer.py",
    f"src/{project_name}/yolo/constant/__init__.py",
    f"src/{project_name}/yolo/constant/training_pipeline/__init__.py",
    f"src/{project_name}/yolo/constant/application.py",
    f"src/{project_name}/yolo/entity/config_entity.py",
    f"src/{project_name}/yolo/entity/artifacts_entity.py",
    f"src/{project_name}/yolo/exception/__init__.py",
    f"src/{project_name}/yolo/logger/__init__.py",
    f"src/{project_name}/yolo/pipeline/__init__.py",
    f"src/{project_name}/yolo/pipeline/training_pipeline.py",
    f"src/{project_name}/yolo/utils/__init__.py",
    f"src/{project_name}/yolo/utils/main_utils.py",
    "reseach/trials.ipynb",
    # "templates/index.html",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
]


for filepath in list_of_files:
    filepath = Path(filepath)

    filedir, filename = os.path.split(filepath)

    if filedir!="":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")

    
    if(not os.path.exists(filename)) or (os.path.getsize(filename) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filename}")

    
    else:
        logging.info(f"{filename} is already created")