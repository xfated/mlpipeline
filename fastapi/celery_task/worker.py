import logging
import os
from celery import Celery
from celery_task.ReviewEmbedding import RestReview
from celery import Task

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "pyamqp://guest@localhost//")
CELERY_BACKEND_URL = os.getenv("CELERY_BACKEND_URL", "redis://localhost") # redis://redis_server:6379

# print(CELERY_BACKEND_URL)
# class PredictTask(Task):
#     """
#     Abstraction of Celery's Task class to support loading ML model.
#     """
#     abstract = True

#     def __init__(self):
#         super().__init__()
#         self.model = None

#     def __call__(self, *args, **kwargs):
#         """
#         Load model on first call
#         Avoids the need to load model on each task request
#         """
#         if not self.model:
#             logging.info('Loading Model...')
#             tokenizer_path = os.getenv("TOKENIZER_PATH", "./token/review_emb-msmarco-distilbert-base-v3-v1")
#             rest_data_path = os.getenv("REST_DATA_PATH", "./restaurant_data")
#             self.model = RestReview(tokenizer_path, rest_data_path)
#         return self.run(*args, **kwargs)

celery = Celery(
    'tasks',
    backend=CELERY_BACKEND_URL,
    broker=CELERY_BROKER_URL,
    # task_cls=PredictTask
)

try: 
    logging.info('Loading Model...')
    tokenizer_path = os.getenv("TOKENIZER_PATH", "./token/review_emb-msmarco-distilbert-base-v3-v1")
    rest_data_path = os.getenv("REST_DATA_PATH", "./restaurant_data")
    model = RestReview(tokenizer_path, rest_data_path)
except Exception as e:
    logging.info(e)
    model = None