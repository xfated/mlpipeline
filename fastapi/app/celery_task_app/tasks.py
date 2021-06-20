import logging
from celery import Celery
from celery_task_app.yolov5s import *
from celery import Task

BROKER_URI = 'pyamqp://guest@localhost//'
BACKEND_URI = 'redis://localhost'

app = Celery(
    'tasks', 
    broker=BROKER_URI,
    backend=BACKEND_URI
)

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
@app.task(ignore_result=False,
        bind=True,
        base=PredictTask,
)
def predict_yolov5s(self, img_base64, url='http://localhost:8501/v1/models/yolov5s:predict'):
    img = self.model.decode_base64(img_base64)
    return self.model.predict(img, url)