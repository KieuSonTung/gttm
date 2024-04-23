from src.object_detection.model import ObjectDetection
import cv2
from src.object_detection.yolo.utils.main_utils import load_video_to_list


def run(video_path: str):
    # load model
    od_model = ObjectDetection(model_name='yolo', cfg_path='runs/detect/train/weights/best.pt')
    frames = load_video_to_list(video_path)
    
    for frame_count, frame in enumerate(frames, 1):
        output = od_model.infer(frame, frame_count)

    return output


results = run('/Users/kieusontung/Downloads/Untitled video - Made with Clipchamp.mp4')
print(results)