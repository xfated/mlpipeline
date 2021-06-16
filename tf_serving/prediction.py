import matplotlib.pyplot as plt
import requests
import base64
import json
import numpy as np
import cv2

url = 'http://localhost:8501/v1/models/yolov5s:predict'

# resize to target shape while maintaining aspect ratio. 
# input:  (h, w, c)
# output: (target, target, c)
def resize(img, target=640):
    h, w, c = img.shape
    # Get ratio for resize
    longside = max(h,w)
    ratio = target/longside
    new_width = int(w*ratio)
    new_height = int(h*ratio)

    # Resize while maintaining aspect ratio
    new_img = np.zeros((target, target, c))
    resized = cv2.resize(img, (new_width,new_height))
    new_img[:new_height,:new_width, :] = resized
    return new_img

# preproc img source to img
def preproc(img_path):
    img = cv2.imread(img_path)
    img = resize(img)
    # cv2.imwrite('test.jpg',img)
    img = img.astype(np.float32)
    img /= 255.0
    img = np.transpose(img, (2,0,1))
    img = np.expand_dims(img, 0)
    return img


def make_prediction(img_path):
    img = preproc(img_path)
    data = json.dumps({
        "signature_name": "serving_default",
        "instances": img.tolist()
    })
    headers = {"content-type": "application/json"}
    json_response = requests.post(url, data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']
    return predictions

if __name__ == "__main__":
    preds = make_prediction('cat.jpg')
    print(type.preds)