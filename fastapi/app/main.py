import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from celery.result import AsyncResult

from celery_task.tasks import predict_yolov5s
from app.models import Task, PredictionResponse , Image

app = FastAPI()

@app.get("/")
async def root():
    print('hi')
    return {"message": "Hello World. this is your Gym Equipment Classifier Speaking"}

@app.post('/gymEqpOD/predict', response_model=Task, status_code=202)
async def yolov5s_predict(img: Image):
    task_id = predict_yolov5s.delay(img.base64Img)
    return {'task_id': str(task_id), 'status': 'Processing'}

@app.get('/gymEqpOD/result/{task_id}', response_model=PredictionResponse, status_code=200,
            responses={202: {'model': Task, 'description': 'Accepted, Not Ready'}})
async def yolov5s_result(task_id):
    """ Fetch result for a given task_id """
    task = AsyncResult(task_id)
    if not task.ready():
        print(app.url_path_for('yolov5s_predict'))
        return JSONResponse(status_code=202, content={'task_id': str(task_id), 'status': 'Processing'})
    output = task.get()
    results = output['results']
    time_taken = output['time_taken']
    
    print(results)
    predictions = []
    for result in results:
        predictions.append(
            {
                'bbox':result['bbox'], 
                'cat': result['class'], 
                'probability':result['conf']
            }
        )
    return {'task_id': task_id, 'status': 'Success', 'preds': predictions, 'time_taken': time_taken}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    