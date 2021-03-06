import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from celery.result import AsyncResult
from fastapi.middleware.cors import CORSMiddleware

from celery_task.tasks import *
from app.models import Task, PredictionResponse, Query

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
async def root():
    print('hi')
    return {"message": "Hello World. this is your Restaurant Finder Speaking"}

## Search all restaurants
@app.post('/restFinder/predict', response_model=Task, status_code=202)
async def pred_rest(query: Query):
    task_id = predict_restaurants.delay(query.input_text, query.top_k)
    return {'task_id': str(task_id), 'status': 'Processing'}

## Search all restaurants Sorted
# @app.post('/restFinderSorted/predict', response_model=Task, status_code=202)
# async def pred_rest(query: Query):
#     task_id = predict_restaurants_sorted.delay(query.input_text, query.top_k)
#     return {'task_id': str(task_id), 'status': 'Processing'}

# @app.get('/restFinder/result/{task_id}', response_model=PredictionResponse, status_code=200,
#             responses={202: {'model': Task, 'description': 'Accepted, Not Ready'}})
# async def rest_result(task_id):
#     """ Fetch result for a given task_id """
#     task = AsyncResult(task_id)
#     if not task.ready():
#         print(app.url_path_for('pred_rest'))
#         return JSONResponse(status_code=202, content={'task_id': str(task_id), 'status': 'Processing'})
#     output = task.get()
#     results = output['results']
#     time_taken = output['time_taken']
#     return {'task_id': task_id, 'status': 'Success', 'preds': results, 'time_taken': time_taken}

## Search in Area
@app.post('/restFinderPostal/predict', response_model=Task, status_code=202)
async def pred_rest_postal(query: Query):
    task_id = predict_restaurants_postal.delay(query.input_text, query.top_k, query.postal)
    return {'task_id': str(task_id), 'status': 'Processing'}

## Search in Area Sorted
# @app.post('/restFinderPostalSorted/predict', response_model=Task, status_code=202)
# async def pred_rest_postal(query: Query):
#     task_id = predict_restaurants_postal_sorted.delay(query.input_text, query.top_k, query.postal)
#     return {'task_id': str(task_id), 'status': 'Processing'}

# @app.get('/restFinderPostal/result/{task_id}', response_model=PredictionResponse, status_code=200,
#             responses={202: {'model': Task, 'description': 'Accepted, Not Ready'}})
# async def rest_result_postal(task_id):
#     """ Fetch result for a given task_id """
#     task = AsyncResult(task_id)
#     if not task.ready():
#         print(app.url_path_for('pred_rest_postal'))
#         return JSONResponse(status_code=202, content={'task_id': str(task_id), 'status': 'Processing'})
#     output = task.get()
#     results = output['results']
#     time_taken = output['time_taken']
#     return {'task_id': task_id, 'status': 'Success', 'preds': results, 'time_taken': time_taken}

## Search in Region
@app.post('/restFinderRegion/predict', response_model=Task, status_code=202)
async def pred_rest_region(query: Query):
    task_id = predict_restaurants_region.delay(query.input_text, query.top_k, query.region)
    return {'task_id': str(task_id), 'status': 'Processing'}

## Search in Region Sorted
# @app.post('/restFinderRegionSorted/predict', response_model=Task, status_code=202)
# async def pred_rest_region(query: Query):
#     task_id = predict_restaurants_region_sorted.delay(query.input_text, query.top_k, query.region)
#     return {'task_id': str(task_id), 'status': 'Processing'}

# @app.get('/restFinderRegion/result/{task_id}', response_model=PredictionResponse, status_code=200,
#             responses={202: {'model': Task, 'description': 'Accepted, Not Ready'}})
# async def rest_result_region(task_id):
#     """ Fetch result for a given task_id """
#     task = AsyncResult(task_id)
#     if not task.ready():
#         print(app.url_path_for('pred_rest_region'))
#         return JSONResponse(status_code=202, content={'task_id': str(task_id), 'status': 'Processing'})
#     output = task.get()
#     results = output['results']
#     time_taken = output['time_taken']
#     return {'task_id': task_id, 'status': 'Success', 'preds': results, 'time_taken': time_taken}

## Random all
@app.post('/restRandom/predict', response_model=Task, status_code=202)
async def rand_all(query: Query):
    task_id = random_all.delay(k=query.top_k)
    return {'task_id': str(task_id), 'status': 'Processing'}

# @app.get('/restRandom/result/{task_id}', response_model=PredictionResponse, status_code=200,
#             responses={202: {'model': Task, 'description': 'Accepted, Not Ready'}})
# async def rand_all_result(task_id):
#     """ Fetch result for a given task_id """
#     task = AsyncResult(task_id)
#     if not task.ready():
#         print(app.url_path_for('rand_all'))
#         return JSONResponse(status_code=202, content={'task_id': str(task_id), 'status': 'Processing'})
#     output = task.get()
#     results = output['results']
#     time_taken = output['time_taken']
#     return {'task_id': task_id, 'status': 'Success', 'preds': results, 'time_taken': time_taken}

## Random in area
@app.post('/restRandomPostal/predict', response_model=Task, status_code=202)
async def rand_postal(query: Query):
    task_id = random_postal.delay(k=query.top_k, postal_code = query.postal)
    return {'task_id': str(task_id), 'status': 'Processing'}

# @app.get('/restRandomPostal/result/{task_id}', response_model=PredictionResponse, status_code=200,
#             responses={202: {'model': Task, 'description': 'Accepted, Not Ready'}})
# async def rand_result_postal(task_id):
#     """ Fetch result for a given task_id """
#     task = AsyncResult(task_id)
#     if not task.ready():
#         print(app.url_path_for('rand_postal'))
#         return JSONResponse(status_code=202, content={'task_id': str(task_id), 'status': 'Processing'})
#     output = task.get()
#     results = output['results']
#     time_taken = output['time_taken']
#     return {'task_id': task_id, 'status': 'Success', 'preds': results, 'time_taken': time_taken}

## Random in Region
@app.post('/restRandomRegion/predict', response_model=Task, status_code=202)
async def rand_region(query: Query):
    task_id = random_region.delay(k=query.top_k, region = query.region)
    return {'task_id': str(task_id), 'status': 'Processing'}

# @app.get('/restRandomRegion/result/{task_id}', response_model=PredictionResponse, status_code=200,
#             responses={202: {'model': Task, 'description': 'Accepted, Not Ready'}})
# async def rand_result_region(task_id):
#     """ Fetch result for a given task_id """
#     task = AsyncResult(task_id)
#     if not task.ready():
#         print(app.url_path_for('rand_region'))
#         return JSONResponse(status_code=202, content={'task_id': str(task_id), 'status': 'Processing'})
#     output = task.get()
#     results = output['results']
#     time_taken = output['time_taken']
#     return {'task_id': task_id, 'status': 'Success', 'preds': results, 'time_taken': time_taken}

## Topk all
@app.post('/restTopk/predict', response_model=Task, status_code=202)
async def topk_all_api(query: Query):
    task_id = topk_all.delay(k=query.top_k)
    return {'task_id': str(task_id), 'status': 'Processing'}

# @app.get('/restTopk/result/{task_id}', response_model=PredictionResponse, status_code=200,
#             responses={202: {'model': Task, 'description': 'Accepted, Not Ready'}})
# async def topk_all_result(task_id):
#     """ Fetch result for a given task_id """
#     task = AsyncResult(task_id)
#     if not task.ready():
#         print(app.url_path_for('topk_all'))
#         return JSONResponse(status_code=202, content={'task_id': str(task_id), 'status': 'Processing'})
#     output = task.get()
#     results = output['results']
#     time_taken = output['time_taken']
#     return {'task_id': task_id, 'status': 'Success', 'preds': results, 'time_taken': time_taken}

## Topk in area
@app.post('/restTopkPostal/predict', response_model=Task, status_code=202)
async def topk_postal_api(query: Query):
    task_id = topk_postal.delay(k=query.top_k, postal_code = query.postal)
    return {'task_id': str(task_id), 'status': 'Processing'}

# @app.get('/restTopkPostal/result/{task_id}', response_model=PredictionResponse, status_code=200,
#             responses={202: {'model': Task, 'description': 'Accepted, Not Ready'}})
# async def topk_result_postal(task_id):
#     """ Fetch result for a given task_id """
#     task = AsyncResult(task_id)
#     if not task.ready():
#         print(app.url_path_for('topk_postal'))
#         return JSONResponse(status_code=202, content={'task_id': str(task_id), 'status': 'Processing'})
#     output = task.get()
#     results = output['results']
#     time_taken = output['time_taken']
#     return {'task_id': task_id, 'status': 'Success', 'preds': results, 'time_taken': time_taken}

## Topk in Region
@app.post('/restTopkRegion/predict', response_model=Task, status_code=202)
async def topk_region_api(query: Query):
    task_id = topk_region.delay(k=query.top_k, region = query.region)
    return {'task_id': str(task_id), 'status': 'Processing'}

# @app.get('/restTopkRegion/result/{task_id}', response_model=PredictionResponse, status_code=200,
#             responses={202: {'model': Task, 'description': 'Accepted, Not Ready'}})
# async def topk_result_region(task_id):
#     """ Fetch result for a given task_id """
#     task = AsyncResult(task_id)
#     if not task.ready():
#         print(app.url_path_for('topk_region'))
#         return JSONResponse(status_code=202, content={'task_id': str(task_id), 'status': 'Processing'})
#     output = task.get()
#     results = output['results']
#     time_taken = output['time_taken']
#     return {'task_id': task_id, 'status': 'Success', 'preds': results, 'time_taken': time_taken}

## Default get result endpoint
@app.get('/result/{task_id}', response_model=PredictionResponse, status_code=200,
            responses={202: {'model': Task, 'description': 'Accepted, Not Ready'}})
async def topk_result_region(task_id):
    """ Fetch result for a given task_id """
    task = AsyncResult(task_id)
    if not task.ready():
        return JSONResponse(status_code=202, content={'task_id': str(task_id), 'status': 'Processing'})
    output = task.get()
    results = output['results']
    time_taken = output['time_taken']
    err_msg = output['err']
    return {'task_id': task_id, 'status': 'Success', 'preds': results, 'time_taken': time_taken, 'error': err_msg}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    