from bytetrack.byte_tracker import BYTETracker
import numpy as np

class MockDetector(object):
    def __init__(self):
        pass

    def inference(self, image):
        """
        Tạo các mock sample (object giả được phát hiện trong frame)
        """
        outputs = np.random.rand(1, 10, 5) * 416 # 1 just for batch size as 1
        
        img_info = {
            'height': 416,
            'width': 416
        }
        return [
            outputs, img_info
        ]


def run():
    
    """mock up run
    - Width, height của mock video là 416x416 với 3 kênh màu RGB
    - Detection row có thể nhận nhiều hơn 5 trường
    + Dạng đơn giản nhất gồm tọa đồ và confidence: (x1, y1, x2, y2, obj_conf) 
    + Hoặc có thể thêm class confidence, class prediction với những model có hỗ trợ như yoloX: (x1, y1, x2, y2, obj_conf, class_conf, class_pred) 
    Returns:
        
    """
    # Hyper Params to tune:
    # Có thể tạo config file 
    aspect_ratio_thresh = 0.6
    min_box_area = 10
    track_thresh = 0.5
    track_buffer = 30
    match_thresh = 0.8
    fuse_score = False
    frame_rate = 30
    test_size = [416, 416]

    detector = MockDetector()
    tracker = BYTETracker(
        track_thresh=track_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        fuse_score=fuse_score,
        frame_rate=frame_rate)
    results = []
    mock_video = [
        np.random.rand(416, 416, 3),
        np.random.rand(416, 416, 3),
        np.random.rand(416, 416, 3),
        np.random.rand(416, 416, 3)
    ]

    for frame_id, image in enumerate(mock_video, 1):
        outputs, img_info = detector.inference(image)
        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], test_size)
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
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
    print("Mock up test results are:")
    print(results)
    return results


run()
