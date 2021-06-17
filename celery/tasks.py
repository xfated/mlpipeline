import os
from celery import Celery

BROKER_URI = 'pyamqp://guest@localhost//'
BACKEND_URI = 'redis://localhost'

app = Celery(
    'tasks', 
    broker=BROKER_URI,
    backend=BACKEND_URI
)

@app.task
def add(x, y):
    return x + y