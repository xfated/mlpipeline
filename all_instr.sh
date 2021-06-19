# Start tensorflow serving
docker run -p 8501:8501 --name tfserving_yolov5s -t yolov5s_serving


# Start celery
docker run -d -p 5672:5672 rabbitmq:3.8.17
docker run -d -p 6379:6379 redis:alpine3.13
celery -A celery_task_app.tasks worker --loglevel=INFO