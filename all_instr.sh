# Start tensorflow serving
docker run -p 8501:8501 --name tfserving_yolov5s -t yolov5s_serving


# Start celery
docker run -d -p 5672:5672 rabbitmq:3.8.17
docker run -d -p 6379:6379 redis:alpine3.13
cd fastapi/
celery -A celery_task.tasks worker --loglevel=INFO

# Start FastAPI
cd fastapi/
uvicorn app.main:app --host 0.0.0.0 --port 8000


# Build containers
docker-compose up --build



## Start
cd fastapi
docker-compose up -d

cd tf_serving
docker-compose up -d
