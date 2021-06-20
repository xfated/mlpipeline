# Running rabbitmq on docker
docker run -d -p 5672:5672 rabbitmq

# Running redis on docker
docker run -d -p 6379:6379 redis

# Testing celery task
import cv2
from tasks import predict_yolov5s
from yolov5s import Yolov5s

img = cv2.imread('../tf_serving/cat.jpg')
model = Yolov5s()
img_base64 = model.encode_base64(img)
print(len(img_base64))

res = predict_yolov5s.delay(img_base64)
res.get()