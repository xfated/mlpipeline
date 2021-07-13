# mlpipeline

## fastapi
Contains implementation for 
- FastAPI HTTP server
- Celery to manage calls to TensorflowServing

## tf_serving
Contains files for my tokenizer

## kubernetes
Contains .yaml files to start Kubernetes service.
Currently pulling from a public repo at docker hub (considering setting it private? if this gets larger)



v3: with pretrained msmarco-distilbert-base-v4 - rest_review_distilbert_base
v3.1: with trained on reviews - rest_review_distilbert