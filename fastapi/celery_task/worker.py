import os
from celery import Celery

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "pyamqp://guest@localhost//")
CELERY_BACKEND_URL = os.getenv("CELERY_BACKEND_URL", "redis://localhost") # redis://redis_server:6379

celery = Celery(
    'tasks',
    backend=CELERY_BACKEND_URL,
    broker=CELERY_BROKER_URL
)
