from src.object_detection.model import ObjectDetection
import cv2
import torch


def run(video_path: str):
    # load model
    od_model = ObjectDetection(model_name='yolo', cfg_path='runs/detect/train/weights/best.pt')

    # load video
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 1

    if not video_capture.isOpened():
            print("Error: Unable to open video file")
    
    while True:
        ret, frame = video_capture.read()
            
        # Break the loop if there are no more frames
        if not ret:
            break

        output = od_model.infer(frame, frame_count)

        # Increment frame count
        frame_count += 1

    # Release the video capture object
    video_capture.release()

    return output


# if __name__ == 'main':
results = run('/Users/kieusontung/Downloads/Untitled video - Made with Clipchamp.mp4')
print(results)