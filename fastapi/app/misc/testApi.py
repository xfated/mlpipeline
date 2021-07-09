import requests
import json

def postText(url, endpoint, query, postal, region, top_k=5):
    data = {'input_text': query, 'top_k':top_k, 'postal':postal, 'region':region}

    pred_url = url + f'/{endpoint}/predict'
    res = requests.post(pred_url, json=data)

    print(res.status_code)
    print(res.text)

## Get result of Celery task. 
def getResult(url, endpoint, task_id):
    result_url = url + f'/{endpoint}/result/' + task_id
    res = requests.get(result_url)

    data = res.json()
    try:
        preds = data['preds']
        for res in preds:
            print(res)
            print()
    except Exception as e:
        print(data)
        # pass
    
# GET request at default endpoint, to check if service is up
def test():
    r = requests.get(url)
    print(r.text)

if __name__ == "__main__":
    ## Define http endpoint for testing
    # Testing with Google Kubernetes Engine
    # gke_ip = '34.126.146.206'
    # url = f'http://{gke_ip}:8000' 
    
    # Testing on local setup
    url = 'http://127.0.0.1:8000'
 
    endpoints = ['restFinder',       # 0
                'restFinder',        # 1
                'restFinderPostal',  # 2 
                'restFinderRegion',  # 3
                'restRandom',        # 4
                'restRandomPostal',  # 5
                'restRandomRegion',  # 6
                'restTopk',          # 7
                'restTopkPostal',    # 8
                'restTopkRegion']    # 9

    endpoint = endpoints[0]
    ## Send HTTP request
    # Start task
    query = 'Thai steamboat'
    postal = '64'
    region = 'North'
    top_k = 3
    # for endpoint in endpoints:
    # postText(url, endpoint, query, postal, region, top_k)
    
    # Get result
    task_id = 'fb6c6688-1eb4-4a89-bc73-a269974f65db'
    getResult(url, endpoint, task_id)
    # Basic test 
    # test()