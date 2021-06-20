from pydantic import BaseModel
from typing import List

class Image(BaseModel):
    base64Img: str
    
class Task(BaseModel):
    """ Celery task representation """
    task_id: str
    status: str

class Prediction(BaseModel):
    """ Single prediction task result """
    bbox: List[int]
    cat: str
    probability: float
    
class PredictionResponse(BaseModel):
    """ Prediction task response """
    task_id: str
    preds: List[Prediction]
    time_taken:float

