from pydantic import BaseModel
from typing import List

class Query(BaseModel):
    input_text: str
    top_k: int
    postal: str
    region: str
    
class Task(BaseModel):
    """ Celery task representation """
    task_id: str
    status: str

class Prediction(BaseModel):
    """ Single prediction task result """
    name: str
    address: str
    tags: List[str]
    about: str
    summary: str
    link: str
    rating: float
    
class PredictionResponse(BaseModel):
    """ Prediction task response """
    task_id: str
    preds: List[Prediction]
    time_taken:float
    error: str

