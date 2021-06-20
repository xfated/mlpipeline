import cv2
import requests
from celery_task_app.yolov5s import Yolov5s
import json

model = Yolov5s()
url = 'http://localhost:8000'

def postImg():
    global url

    img_path = '../../tf_serving/cat.jpg'
    img = cv2.imread(img_path)
    
    img_base64 = model.encode_base64(img)
    data = {'base64Img': img_base64}

    pred_url = url + '/gymEqpOD/predict'
    res = requests.post(pred_url, json=data)

    print(res.status_code)
    print(res.text)

def getResult():
    global url

    task_id = '946a4496-592c-4376-aa7d-77362112f6e1'
    result_url = url + '/gymEqpOD/result/' + task_id
    res = requests.get(result_url)
    
    img_path = '../../tf_serving/cat.jpg'
    img = cv2.imread(img_path)

    data = res.json()
    bbox = data['bbox']
    cat = data['cat']
    conf = data['probability']

    img = cv2.rectangle(img, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (255,0,0), 2)
    cv2.imshow('test',img)
    cv2.waitKey()

def test():
    r = requests.get(url)
    print(r.text)

if __name__ == "__main__":
    # postImg()
    getResult()
    # test()