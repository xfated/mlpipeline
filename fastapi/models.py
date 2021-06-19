from pydantic import BaseModel
from typing import List

class Image(BaseModel):
    base64Img: str
    
class Task(BaseModel):
    """ Celery task representation """
    task_id: str
    status: str

class Prediction(BaseModel):
    """ Prediction task result """
    task_id: str
    bbox: List[int]
    cat: str
    probability: float
    