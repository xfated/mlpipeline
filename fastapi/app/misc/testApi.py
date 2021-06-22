import cv2
import requests
from yolov5s import Yolov5s
import json

model = Yolov5s()
url = 'http://127.0.0.1:8000'

def postImg():
    global url

    img_path = '../../../tf_serving/cat.jpg'
    img = cv2.imread(img_path)
    
    img_base64 = model.encode_base64(img)
    data = {'base64Img': img_base64}

    pred_url = url + '/gymEqpOD/predict'
    res = requests.post(pred_url, json=data)

    print(res.status_code)
    print(res.text)

def getResult():
    global url

    task_id = '8ed85f7b-6685-46df-bffa-fa1afe5c23c2'
    result_url = url + '/gymEqpOD/result/' + task_id
    res = requests.get(result_url)
    
    img_path = '../../../tf_serving/cat.jpg'
    img = cv2.imread(img_path)

    print('test', res.text)
    data = res.json()
    
    pred = data['preds'][0]
    bbox = pred['bbox']
    cat = pred['cat']
    conf = pred['probability']

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