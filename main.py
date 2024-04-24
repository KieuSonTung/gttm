from src.object_detection.model import ObjectDetection
from src.object_tracking.bytetrack.byte_tracker import BYTETracker
import torch
import json
from src.object_detection.yolo.utils.main_utils import load_video_to_list, get_img_info


def run(video_path: str):
    
    img_info = get_img_info(video_path)
    frames = load_video_to_list(video_path)
    
    # load object detection model
    od_model = ObjectDetection(model_name='yolo', cfg_path='runs/detect/train/weights/best.pt')

    # Load Hyper Params to tune:
    with open('src/object_tracking/bytetrack/config.json', 'r') as f:
        config = json.load(f)
    
    # Extract hyperparameters from config
    # giá trị sau key là default nếu file config không có key đó
    aspect_ratio_thresh = config.get('aspect_ratio_thresh', 0.6)
    min_box_area = config.get('min_box_area', 10)
    track_thresh = config.get('track_thresh', 0.5)
    track_buffer = config.get('track_buffer', 30)
    match_thresh = config.get('match_thresh', 0.8)
    fuse_score = config.get('fuse_score', False)
    test_size = config.get('test_size', [640,640])
    frame_rate = img_info['fps']

    tracker = BYTETracker(
        track_thresh=track_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        fuse_score=fuse_score,
        frame_rate=frame_rate)
    
    results = []
    
    for frame_count, frame in enumerate(frames, 1):
        output = od_model.infer(frame, frame_count)

        if output is not None:
            # Lấy ra tọa độ x và y từ tensor
            x = output[:, 1]
            y = output[:, 2]

            # Tính toán w và h
            w = output[:, 3] - x
            h = output[:, 4] - y

            # Tạo tensor mới chứa x, y, w, h
            output = torch.stack((x, y, w, h, output[:, 5]), dim=1)
            
            online_targets = tracker.update(output, [img_info['height'], img_info['width']], test_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    
                    # save results
                    results.append([
                        frame_count,
                        tid,
                        "{:.2f}".format(tlwh[0]),
                        "{:.2f}".format(tlwh[1]),
                        "{:.2f}".format(tlwh[2]),
                        "{:.2f}".format(tlwh[3]),
                        "{:.2f}".format(t.score),
                        -1, -1, -1
                    ])

    return results


results = run('/Users/kieusontung/Downloads/Untitled video - Made with Clipchamp.mp4')
print(results)