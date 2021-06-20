import logging
from celery_task.yolov5s import *
from celery import Task

from celery_task.worker import celery

class PredictTask(Task):
    """
    Abstraction of Celery's Task class to support loading ML model.
    """
    abstract = True

    def __init__(self):
        super().__init__()
        self.model = None

    def __call__(self, *args, **kwargs):
        """
        Load model on first call
        Avoids the need to load model on each task request
        """
        if not self.model:
            logging.info('Loading Model...')
            self.model = Yolov5s()
        return self.run(*args, **kwargs)


# img: base64 encoded image 
@celery.task(ignore_result=False,
        bind=True,
        base=PredictTask,
)
def predict_yolov5s(self, img_base64, url='http://localhost:8501/v1/models/yolov5s:predict'): # 'http://localhost:8501/v1/models/yolov5s:predict'
    img = self.model.decode_base64(img_base64)
    results, time_taken = self.model.predict(img, url)
    return {'results': results, 'time_taken': time_taken}