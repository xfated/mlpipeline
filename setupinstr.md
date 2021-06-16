# Comparisons for ArcFace model in API
Analysis of time taken for each step on API call.  
Emphasis on time taken for face_embedding.
### MXNet
![mxnet_timings](readme_images/mxnet_timings.png)
![mxnet_gpu](readme_images/mxnet_gpu.png)
### OnnxRuntime
![onnx_timings](readme_images/onnx_timings.png)
![onnx_gpu](readme_images/onnx_gpu.png)
### TVM
![tvm_timings](readme_images/tvm_timings.png)
![tvm_gpu](readme_images/tvm_gpu.png)
### TensorRT

# Setting up Inference Frameworks
# ONNX Runtime
## Exporting and using onnxruntime
1. Preparing model: Export MXNet model to ONNX.
    1. Script available at [export_onnx.sh](ArcFace/export_onnx.sh)
    1. Have to split PRelu to Reshape + PRelu
    1. requirements: protobuf and numpy
    1. conda install -c conda-forge protobuf numpy
    1. onnxruntime-gpu==1.1.2 for CUDA v10.0
1. Loading model: Use ONNX Runtime to create inference session with ONNX model.
    1. Using onnx-runtime-gpu==1.1.2  
``` python
# Load model
model_path = '/data/users/kaiyang_tay/repo/models/arcface_onnx/th_chosen_model.onnx'
sess = rt.InferenceSession(model_path)

# Make inference
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

data = np.random.random((1,3,112,112)).astype(np.float32)
res = sess.run([output_name], {input_name: data})
```  

## Set up for ONNX Runtime in API
**worker .yml**
- pip:
    - onnxruntime-gpu==1.1.2  

**config and core/manager/face.py**
1. Store model as necessary  
1. Add path in config
1. Provide path as argument to ArcFaceManager in face.py

**repo/models/insightface/deploy/face_model.py**
``` python
class FaceModel:
    def __init__(self, args):
        ...
        self.model = rt.InferenceSession(args.model)
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name

    def get_feature(self,aligned):
        input_blob = np.expand_dims(aligned, axis=0)
        input_blob = input_blob.astype(np.float32)
        embedding = self.model.run([self.output_name], {self.input_name: input_blob})
        embedding = np.asarray(embedding[0])
        ...

```
  
<br />  

 
# TVM
Based on tvm version 0.8.dev0
## To build the libtvm.so file
Following instructions from [https://tvm.apache.org/docs/install/from_source.html](https://tvm.apache.org/docs/install/from_source.html) to build from source.
1. Clone git:
    ```
    git clone --recursive https://github.com/apache/tvm tvm
    ```
1. Building shared library:
    ```
    sudo apt-get update
    sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
    ```
1. Create build folder:
    ```
    mkdir build
    cp cmake/config.cmake build
    ```
1. Download pre-built version of LLVM from the [LLVM Download Page](https://releases.llvm.org/download.html)
    1. I used [LLVM 9.0.0 - Ubuntu 16.04](https://releases.llvm.org/9.0.0/clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz)
1. Edit config.cmake based on requirements. In my case:
    1. set(USE_CUDA_ON)
    1. set(USE_CUDNN ON)
    1. set(USE_GRAPH_RUNTIME ON)
    1. set(USE_GRAPH_RUNTIME_DEBUG ON)
    1. set(USE_LLBM /path/to/DOWNLOADED LLVM FROM ABOVE/bin/llvm-config)
1. Build tvm and related libraries
    ``` 
    cd build
    cmake ..
    make -j4
    ```
1. Python installation (either option)
    1. ```
        cd python; python setup.py install --user; cd ..
        ```
    1. ``` 
        export TVM_HOME=/path/to/tvm    
        export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH 
        ```

## Exporting TVM model from MXNet to lib/graph/params/
``` python
import tvm
from tvm import relay
import mxnet as mx

shape_dict = {"data":(1,3,112,112)}
model_folder = '/data/users/kaiyang_tay/groupface/frvt/frvt_2021/insightface/recognition/ArcFace/models/th_chosen_model/model'
sym, arg_params, aux_params = mx.model.load_checkpoint(model_folder, 50)
mod, relay_params = relay.frontend.from_mxnet(sym, shape_dict, arg_params=arg_params, aux_params=aux_params)

# compiling
func = mod['main']
func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)
target = "cuda" # "cuda"
with tvm.transform.PassContext(opt_level=3):
    graph, lib, params = relay.build_module.build(func, target, params=relay_params)

## Export to tar/json/params
lib.export_library("arcface_tvm/lib.tar")
with open("arcface_tvm/graph.json","w") as fo:
    fo.write(graph)
with open("arcface_tvm/param.params","wb") as fo:
    fo.write(relay.save_param_dict(params))
```
## Set up for tvm in API
1. copy tvm directory to /home/docker/workspace (e.g. add tvm dir to ./selfie_to_ic_face_matching) 
1. export path in boot/docker/celery/face/entrypoint.sh
    - ```
        export TVM_HOME=/home/docker/workspace/tvm && \
        export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH && \
        ```
1. file changes:
    - **config and core/manager/face.py**
        1. Store model as necessary  
        1. Add path in config
        1. Provide path as argument to ArcFaceManager in face.py

### Option 1: Build model from MXNet or onnx (example shown is for MXNet)
- **worker .yml**
    - dependencies:
        - pytest  
- **boot/docker/celery/face/entrypoint.sh**
    ```
    sudo chmod -R 777 ./ && \
    ```
- **repo/models/insightface/deploy/face_model.py**
    ``` python
        class FaceModel:
            def __init__(self, args):
                ...
                import tvm.relay as relay
                shape_dict={"data":(1,3,112,112)}
                # path and version of mxnet model
                model_folder, version = args.model.split(',')
                sym, arg_params, aux_params = mx.model.load_checkpoint(model_folder, version)
                mod, relay_params = relay.frontend.from_mxnet(sym, shape_dict,             arg_params=arg_params, aux_params=aux_param$
                func=mod['main']
                func=relay.Function(func.params, func.body, None, func.type_params, func.attrs)
                target ="cuda"
                with tvm.transform.PassContext(opt_level=3):
                     lib = relay.build(func, target, params=relay_params)
                
                tvm_ctx = tvm.gpu(0)
                self.model = graph_runtime.GraphModule(lib["default"](tvm_ctx))

            def get_feature(self,aligned):
                input_blob = np.array([aligned],dtype=np.float32)
                self.model.set_input("data",tvm.nd.array(input_blob))
                self.model.run()
                embedding = self.model.get_output(0).asnumpy()
                ...
    ```

### Option 2: Build model from lib/graph/params
- **repo/models/insightface/deploy/face_model.py**
    ``` python
        class FaceModel:
            def __init__(self, args):
                ...
                loaded_json = open(os.path.join(args.model,'graph.json')).read()
                loaded_lib = tvm.runtime.load_module(os.path.join(args.model,'lib.tar'))
                loaded_params = bytearray(open(os.path.join(args.model,'param.params'),'rb').read())
                tvm_ctx = tvm.gpu(0) # was build and exported for gpu
                self.model =  graph_runtime.create(loaded_json, loaded_lib, tvm_ctx)
                self.model.load_params(loaded_params)

            def get_feature(self,aligned):
                input_blob = np.array([aligned],dtype=np.float32)
                self.model.set_input("data",tvm.nd.array(input_blob))
                self.model.run()
                embedding = self.model.get_output(0).asnumpy()
                ...
    ```


# TensorRT 
1. The following steps are the set up process based on instructions at [TensorRT build environment](https://github.com/NVIDIA/TensorRT/tree/release/7.0).  
Reference: https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-700/tensorrt-install-guide/index.html
1. Prepare TensorRT OSS: (Might not be necessary)
    1. Download TensorRT OSS:
        ```
        git clone -b master https://github.com/nvidia/TensorRT TensorRT -b release/7.0
        cd TensorRT  
        git submodule update --init --recursive
        export TRT_SOURCE=`pwd`
        ```
    1. Install dependencies:
        - ```
            conda install -c conda-forge protobuf numpy
            ```
        - add this line in CMakeLists.txt (before line 46)
            - ```
                set(CMAKE_CUDA_COMPILER "/usr/local/cuda-10.0/bin/nvcc")
            ```
    1. Building TensorRT-OSS: 
        ```
        cd TensorRT
        mkdir -p build && cd build
        cmake .. -DTRT_LIB_DIR=$TRT_RELEASE/lib -DTRT_BIN_DIR=`pwd`/out -DCUDA_VERSION=10.0
        make -j$(nproc)
        ```
1. Get TensorRT-7.0.0.11
    1. Download and extract TensorRT build from [NVIDIA Developer Zone](https://developer.nvidia.com/nvidia-tensorrt-download):
        - I am using "TensorRT 7.0.0.11 for Ubuntu 16.04 and CUDA 10.0 TAR package"
        - Extract
            ``` 
            tar -xvzf TensorRT-7.0.0.11.Ubuntu-16.04.x86_64-gnu.cuda-10.0.cudnn7.6.tar.gz
            export TRT_RELEASE=`pwd`/TensorRT-7.0.0.11
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TRT_RELEASE/lib
            ```
1. Get onnx-tensorrt:
    1. Install protobuf (for onnx-tensorrt). Find releases [here](https://github.com/protocolbuffers/protobuf/releases)
        ```
        sudo apt-get install autoconf automake libtool curl make g++ unzip
        wget https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protobuf-all-3.15.6.tar.gz
        tar -xvzf protobuf-all-3.15.6.tar.gz
        rm protobuf-all-3.15.6.tar.gz
        cd protobuf-3.15.6
        ./configure
        make
        make check
        sudo make install
        sudo ldconfig
        ```
    1. Building onnx-tensorrt
        ```
        git clone --recurse-submodules https://github.com/onnx/onnx-tensorrt.git -b 7.0
        cd onnx-tensorrt
        <!-- if error, remove build and restart -->
        mkdir build && cd build
        cmake .. -DTENSORRT_ROOT=$TRT_RELEASE && make -j
        export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
        pip install pycuda
        ```
    1. To get python bindings for ONNX-TensorRT:
        - cd TensorRT-7.0.0.11/python
        - python -m pip install tensorrt-7.0.0.11-cp${python-version}-none-linux_x86_64.whl

## Set up for TensorRT in API
1. copy TensorRT-7.0.0.11 and onnx-tensorrt directory to /home/docker/workspace. (e.g. add dirs to ./selfie_to_ic_face_matching) [ Should have been built in the previous steps ]
1. export paths (tensorrt and onnx-tensorrt) in boot/docker/celery/face/entrypoint.sh
    - ```
        export TRT_RELEASE=/home/docker/workspace/TensorRT-7.0.0.11 && \
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TRT_RELEASE/lib && \
        export LD_LIBRARY_PATH=/home/docker/workspace/onnx-tensorrt/build:$LD_LIBRARY_PATH && \
        ```
1. Add pip install (for python bindings) in entrypoint (after source activate venv)
    - ```
        python -m pip install /home/docker/workspace/TensorRT-7.0.0.11/python/tensorrt-7.0.0.11-cp36-none-linux_x86_64.whl && \
        ```
1. file changes:
    - **worker .yml**
        - pip:
            - pycuda==2020.1
            - onnx=1.6.0
    - **config and core/manager/face.py**
        1. Store model as necessary  
        1. Add path in config
        1. Provide path as argument to ArcFaceManager in face.py
    - **repo/models/insightface/deploy/face_model.py**
    ``` python
        class FaceModel:
            def __init__(self, args):
                ...
                import onnx
                sys.path.append('/home/docker/workspace/onnx-tensorrt/')
                import onnx_tensorrt.backend as backend
                self.model = onnx.load(os.path.join(args.model,'th_chosen_model.onnx')
                self.model = backend.prepare(model, device='CUDA:0')

            def get_feature(self,aligned):
                input_blob = np.array([aligned],dtype=np.float32)
                embedding = np.copy(self.model.run(input_blob)[0][0])
                ...
    ```


## Angular Margin Loss for Deep Face Recognition
### Citation

If you find this project useful in your research, please consider to cite the following related papers:

```

@inproceedings{deng2019arcface,
  title={Arcface: Additive angular margin loss for deep face recognition},
  author={Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4690--4699},
  year={2019}
}

@inproceedings{deng2020subcenter,
  title={Sub-center ArcFace: Boosting Face Recognition by Large-scale Noisy Web Faces},
  author={Deng, Jiankang and Guo, Jia and Liu, Tongliang and Gong, Mingming and Zafeiriou, Stefanos},
  booktitle={Proceedings of the IEEE Conference on European Conference on Computer Vision},
  year={2020}
}

```
