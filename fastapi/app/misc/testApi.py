import cv2
import requests
import json

def postText(url, endpoint, query, postal, top_k=5):
    data = {'input_text': query, 'top_k':5, 'postal':postal}

    pred_url = url + f'/{endpoint}/predict'
    res = requests.post(pred_url, json=data)

    print(res.status_code)
    print(res.text)

## Get result of Celery task. 
def getResult(url, endpoint, task_id):
    result_url = url + f'/{endpoint}/result/' + task_id
    res = requests.get(result_url)

    data = res.json()
    preds = data['preds']
    for res in preds:
        print(res)
        print()

# GET request at default endpoint, to check if service is up
def test():
    r = requests.get(url)
    print(r.text)

if __name__ == "__main__":
    ## Define http endpoint for testing
    # Testing with Google Kubernetes Engine
    gke_ip = '34.126.146.206'
    url = f'http://{gke_ip}:8000' 
    
    # Testing on local setup
    # url = 'http://127.0.0.1:8000'

    endpoint = 'restFinder'
    # endpoint = 'restFinderPostal'
    # endpoint = 'restRandom'
    # endpoint = 'restRandomPostal'

    ## Send HTTP request
    # Start task
    # query = 'I want some takoyaki'
    # postal = '238'
    # postText(url, endpoint, query, postal)
    
    # Get result
    task_id = '655f5f0f-20a0-4e5f-9be7-ede0e6a3c56d'
    getResult(url, endpoint, task_id)

    # Basic test
    # test()