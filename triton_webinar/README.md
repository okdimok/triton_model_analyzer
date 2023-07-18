```
                                     ┌──────────────────────────────┐
                                     │                              │
                                     │                              │
                                     │        Triton server         │
                                     │                              │
                                     │                              │
                                     │      ┌────────────────┐      │
                                     │      │                │      │
                                ┌────┼──────►   ONNX model   │      │
                                │    │      │                │      │
                                │    │      └────────────────┘      │
                                │    │                              │
                                │    │                              │
┌───────────────────────┐       │    │                              │
│                       │       │    │                              │
│                       │       │    │      ┌──────────────────┐    │
│                       │       │    │      │                  │    │
│       Client app      ├───────┼────┼──────►  TensorRT model  │    │
│                       │       │    │      │                  │    │
│                       │       │    │      └──────────────────┘    │
│                       │       │    │                              │
└───────────────────────┘       │    │                              │
                                │    │                              │
                                │    │      ┌──────────────────┐    │
                                │    │      │                  │    │
                                └────┼──────►   BLS model      │    │
                                     │      │                  │    │
                                     │      └──────────────────┘    │
                                     │                              │
                                     └──────────────────────────────┘
```

## Topics
* Containers(TensorRT, Triton, SDK and Client) overview
* Create a MNIST PyTorch model and export it to ONNX
* Optimize it using TensorRT
* Setup Triton to serve TensorRT optimized model, ONNX model and the BLS example model
* Create demo client app and test it
* Demonstrate perf_analyzer and model_analyzer


## Commands
1. Clone this repository into the location, which isn't mounted. Otherwise, docker won't be able to mount it to the container.
```
git clone https://github.com/okdimok/triton_model_analyzer.git
cd triton_model_analyzer/triton_webinar
# git clone -b release/8.6 https://github.com/NVIDIA/TensorRT.git
```

2. Run the PyTorch container
```
docker run --runtime=nvidia --gpus all --shm-size=1g --ulimit memlock=-1 -v "$(pwd)":/workspace/ext -ti nvcr.io/nvidia/pytorch:23.05-py3
```

3. Inside the container, export the model
```
cd /workspace/ext
trtexec --onnx=model.onnx --saveEngine=model.engine
```

4. Copy the models created to the model repository

5. Start Triton Inference Server
```
docker run --runtime=nvidia --gpus all --shm-size=1g --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 -p 8002:8002 -v /home/semadu/workspace/inference_demo/:/home/workspace --ulimit stack=67108864 -ti nvcr.io/nvidia/tritonserver:23.05-py3
```

6. Start the container with the Triton clients
```
docker run --gpus all --rm -it -v /var/run/docker.sock:/var/run/docker.sock -v /home/semadu/workspace/inference_demo/inference/model_repo/:/home/semadu/workspace/inference_demo/inference/model_repo -v /home/semadu/workspace/inference_demo/inference:/home/semadu/workspace/inference_demo/inference --net host nvcr.io/nvidia/tritonserver:23.05-py3-sdk
```

model-analyzer profile --model-repository /home/semadu/workspace/inference_demo/inference/model_repo --profile-models mnist_trt --triton-launch-mode=docker --output-model-repository-path /home/semadu/workspace/inference_demo/inference/output --run-config-search-max-concurrency 2 --run-config-search-max-model-batch-size 2 --run-config-search-max-instance-count 2 --override-output-model-repository

perf_analyzer -m mnist_trt