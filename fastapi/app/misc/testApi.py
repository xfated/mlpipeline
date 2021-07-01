import cv2
import requests
import json

url = 'http://127.0.0.1:8000'
# endpoint = 'restFinder'
endpoint = 'restFinderPostal'
# endpoint = 'restRandom'
# endpoint = 'restRandomPostal'
def postText():
    global url

    data = {'input_text': 'coffee tea or me?', 'top_k':5, 'postal':'238'}

    pred_url = url + f'/{endpoint}/predict'
    res = requests.post(pred_url, json=data)

    print(res.status_code)
    print(res.text)

def getResult():
    global url

    task_id = '331548d0-54f2-4b1b-9727-aa17b9c92d21'
    result_url = url + f'/{endpoint}/result/' + task_id
    res = requests.get(result_url)

    # print('test', res.text)
    data = res.json()
    preds = data['preds']
    for res in preds:
        print(res)
        print()

def test():
    r = requests.get(url)
    print(r.text)

if __name__ == "__main__":
    # postText()
    getResult()
    # test()