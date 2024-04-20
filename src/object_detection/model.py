from yolo.model import YOLOObjectDetection


class ObjectDetection():
    def __init__(self, model_name, cfg_path) -> None:
        self.model_name = model_name
        self.cfg_path = cfg_path

    def load_model(self):
        if self.model_name == 'yolo':
            self.model = YOLOObjectDetection(self.cfg_path)
    
    def train(self):
        results = self.model.train()
    
    def infer(self):
        pass


    
    

