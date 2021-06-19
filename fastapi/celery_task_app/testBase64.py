import base64
import cv2
import numpy as np
from yolov5s import *

def testBase64():
    # Read image
    img_path = '../tf_serving/cat.jpg'
    img = cv2.imread(img_path)

    # Encode image
    img = cv2.imencode('.jpg', img)[1]
    img_base64 = base64.b64encode(img).decode()

    # Decode and display image
    img_nparray = np.frombuffer(base64.b64decode(img_base64), np.uint8)
    img_decoded = cv2.imdecode(img_nparray, cv2.IMREAD_COLOR)

    # Display
    print(type(img_decoded))
    cv2.imshow('test', img_decoded)
    cv2.waitKey()

def testYolov5s_base64():
    # Get model
    model = Yolov5s()

    # Read image
    img_path = '../tf_serving/cat.jpg'
    img = cv2.imread(img_path)

    # Encode
    img_base64 = model.encode_base64(img)
    
    print(len(img_base64))
    # Decode
    img_decoded = model.decode_base64(img_base64)

    cv2.imshow('test',img_decoded)
    cv2.waitKey()

if __name__ == "__main__":
    testYolov5s_base64()
    # testBase64()