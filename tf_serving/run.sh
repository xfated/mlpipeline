# https://www.tensorflow.org/tfx/serving/docker
# script to run docker container tensorflow/serving
docker run -p 8501:8501 --name tfserving_yolov5s2 --mount type=bind,source=C:/Users/User/Documents/portfolio/MLPipeline/tf_serving/yolov5s/,target=/models/yolov5s -e MODEL_NAME=yolov5s -t tensorflow/serving


# script to create own serving image 
# First run a serving image as a daemon:
docker run -d --name serving_base tensorflow/serving
# Next, copy your SavedModel to the container's model folder:
docker cp C:/Users/User/Documents/portfolio/MLPipeline/tf_serving/<model>/ serving_base:/models/<model>
    <model> dir:
    ./<model>/
        1
            assets/
            variables
            saved_model.pb
docker commit --change "ENV MODEL_NAME yolov5s" serving_base yolov5s_serving
# Stop serving_base
docker kill serving_base

# Running new container
docker run -p 8501:8501 --name tfserving_yolov5s -t yolov5s_serving

# ------------------------------------------
# For kubernetes deployment
# https://www.tensorflow.org/tfx/serving/serving_kubernetes#part_1_setup