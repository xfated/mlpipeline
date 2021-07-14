import logging
import os
from celery_task.worker import celery
from celery_task.worker import model
'''
Match with all restaurants
'''
@celery.task(ignore_result=False,
        bind=True
)
def predict_restaurants(self, query, top_k, url=os.getenv("TFSERVING_URL", 'http://localhost:8501/v1/models/rest_review_distilbert:predict')):
    results, time_taken, err = model.predict(query, top_k=top_k, url=url)
    return {'results': results, 'time_taken': time_taken, 'err': err}

# '''
# Match with all restaurants Sorted
# '''
# @celery.task(ignore_result=False,
#         bind=True,
#         base=PredictTask,
# )
# def predict_restaurants_sorted(self, query, top_k, url=os.getenv("TFSERVING_URL", 'http://localhost:8501/v1/models/rest_review_distilbert:predict')):
#     results, time_taken = model.predict(query, top_k=top_k, url=url)
#     # Sort based on rating. Desc
#     results = [(result, result['rating']) for result in results]
#     sorted(results,key=lambda x:(-x[1],x[0]))
#     results = [result for result, _ in results]
#     return {'results': results, 'time_taken': time_taken}

'''
Match with restaurants in area
'''
@celery.task(ignore_result=False,
        bind=True
)
def predict_restaurants_postal(self, query, top_k, postal_code, url=os.getenv("TFSERVING_URL", 'http://localhost:8501/v1/models/rest_review_distilbert:predict')):
    results, time_taken, err = model.predict_postal(query, top_k=top_k, postal_code=postal_code, url=url)
    return {'results': results, 'time_taken': time_taken, 'err': err}

# '''
# Match with restaurants in area Sorted
# '''
# @celery.task(ignore_result=False,
#         bind=True,
#         base=PredictTask,
# )
# def predict_restaurants_postal_sorted(self, query, top_k, postal_code, url=os.getenv("TFSERVING_URL", 'http://localhost:8501/v1/models/rest_review_distilbert:predict')):
#     results, time_taken = model.predict_postal(query, top_k=top_k, postal_code=postal_code, url=url)
#     # Sort based on rating. Desc
#     results = [(result, result['rating']) for result in results]
#     sorted(results,key=lambda x:(-x[1],x[0]))
#     results = [result for result, _ in results]
#     return {'results': results, 'time_taken': time_taken}


'''
Match with restaurants in region
'''
@celery.task(ignore_result=False,
        bind=True
)
def predict_restaurants_region(self, query, top_k, region, url=os.getenv("TFSERVING_URL", 'http://localhost:8501/v1/models/rest_review_distilbert:predict')):
    results, time_taken, err = model.predict_region(query, top_k=top_k, region=region, url=url)
    return {'results': results, 'time_taken': time_taken, 'err': err}

# '''
# Match with restaurants in region Sorted
# '''
# @celery.task(ignore_result=False,
#         bind=True,
#         base=PredictTask,
# )
# def predict_restaurants_region_sorted(self, query, top_k, region, url=os.getenv("TFSERVING_URL", 'http://localhost:8501/v1/models/rest_review_distilbert:predict')):
#     results, time_taken = model.predict_region(query, top_k=top_k, region=region, url=url)
#     # Sort based on rating. Desc
#     results = [(result, result['rating']) for result in results]
#     sorted(results,key=lambda x:(-x[1],x[0]))
#     results = [result for result, _ in results]
#     return {'results': results, 'time_taken': time_taken}


'''
Random select from all
'''
@celery.task(ignore_result=False,
        bind=True
)
def random_all(self, k):
    results, time_taken = model.search_random(k=k)
    return {'results': results, 'time_taken': time_taken, 'err': ''}

'''
Random select from area
'''
@celery.task(ignore_result=False,
        bind=True
)
def random_postal(self, k, postal_code):
    results, time_taken = model.search_postal_random(k=k, postal_code=postal_code)
    return {'results': results, 'time_taken': time_taken, 'err': ''}

'''
Random select from region
'''
@celery.task(ignore_result=False,
        bind=True
)
def random_region(self, k, region):
    results, time_taken = model.search_region_random(k=k, region=region)
    return {'results': results, 'time_taken': time_taken, 'err': ''}

'''
Select topk from all
'''
@celery.task(ignore_result=False,
        bind=True
)
def topk_all(self, k):
    results, time_taken = model.search_topk(k=k)
    return {'results': results, 'time_taken': time_taken, 'err': ''}


'''
Select topk from area
'''
@celery.task(ignore_result=False,
        bind=True
)
def topk_postal(self, k, postal_code):
    results, time_taken = model.search_postal_topk(k=k, postal_code=postal_code)
    return {'results': results, 'time_taken': time_taken, 'err': ''}

'''
Select topk from region
'''
@celery.task(ignore_result=False,
        bind=True
)
def topk_region(self, k, region):
    results, time_taken = model.search_region_topk(k=k, region=region)
    return {'results': results, 'time_taken': time_taken, 'err': ''}