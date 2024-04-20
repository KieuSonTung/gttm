from src.object_detection.yolo.model import YOLOObjectDetection


class ObjectDetection():
    def __init__(self, model_name, cfg_path) -> None:
        '''
        model_name (str): yolo, opencv, ...
        cfg_path (str): path to hyperparameters yaml file
        '''
        self.model_name = model_name
        self.cfg_path = cfg_path

    def load_model(self):
        if self.model_name == 'yolo':
            model = YOLOObjectDetection(self.cfg_path)

        return model
    
    def train(self):
        self.model = self.load_model()
        results = self.model.train()
    
    def infer(self):
        pass


    
    

