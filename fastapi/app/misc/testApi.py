import cv2
import requests
import json

<<<<<<< Updated upstream
model = Yolov5s()
url = 'http://127.0.0.1:8000'

def postImg():
=======
url = 'http://127.0.0.1:8000'
# endpoint = 'restFinder'
endpoint = 'restFinderPostal'
# endpoint = 'restRandom'
# endpoint = 'restRandomPostal'
def postText():
>>>>>>> Stashed changes
    global url

    data = {'input_text': 'coffee tea or me?', 'top_k':5, 'postal':'238'}

    pred_url = url + f'/{endpoint}/predict'

    

    res = requests.post(pred_url, json=data)

    print(res.status_code)
    print(res.text)

def getResult():
    global url

<<<<<<< HEAD
    task_id = '8ed85f7b-6685-46df-bffa-fa1afe5c23c2'
=======
<<<<<<< Updated upstream
    task_id = '30378c82-94b1-4f5e-af53-e255aeca136a'
>>>>>>> a6c80bd... added tasks for postal code
    result_url = url + '/gymEqpOD/result/' + task_id
=======
    task_id = '5f91c004-a820-4365-8ca4-fb95c69dc1da'
    result_url = url + f'/{endpoint}/result/' + task_id
>>>>>>> Stashed changes
    res = requests.get(result_url)

<<<<<<< HEAD
    print('test', res.text)
    data = res.json()
    
=======
    # print('test', res.text)
    data = res.json()
<<<<<<< Updated upstream
    print(data)
>>>>>>> a6c80bd... added tasks for postal code
    pred = data['preds'][0]
    bbox = pred['bbox']
    cat = pred['cat']
    conf = pred['probability']

    img = cv2.rectangle(img, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (255,0,0), 2)
    cv2.imshow('test',img)
    cv2.waitKey()
=======
    preds = data['preds']
    for res in preds:
        print(res)
        print()
>>>>>>> Stashed changes

def test():
    r = requests.get(url)
    print(r.text)

if __name__ == "__main__":
    # postText()
    getResult()
    # test()