#/bin/bash

## For local testing
# Start tensorflow serving
docker run -d -p 8501:8501 --name rest_review_distilbert -t rest_review_distilbert

# Start celery
docker run -d -p 5672:5672 rabbitmq:3.8.17
docker run -d -p 6379:6379 redis:alpine3.13
cd fastapi/
celery -A celery_task.tasks worker --loglevel=INFO

# Start FastAPI
cd fastapi/
uvicorn app.main:app --host 0.0.0.0 --port 8000

## For docker compose testing
# Build containers and run
cd fastapi
docker-compose up --build

## Detached mode
cd fastapi
docker-compose up -d