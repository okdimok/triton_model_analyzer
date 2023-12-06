```
                                     ┌───────────────────────────────┐
                                     │                               │
                                     │        Triton server          │
                                     │                               │
                                     │      ┌──────────────────┐     │
                                ┌────┼──────►    ONNX model    │     │
                                │    │      └──────────────────┘     │
┌───────────────────────┐       │    │                               │
│                       │       │    │      ┌──────────────────┐     │
│       Client app      ├───────┼────┼──────►  TensorRT model  │     │
│                       │       │    │      └──────────────────┘     │
└───────────────────────┘       │    │                               │
                                │    │      ┌──────────────────┐     │
                                ├────┼──────► TorchScript model│     │
                                │    │      └──────────────────┘     │
                                │    │                               │
                                │    │      ┌──────────────────┐     │
                                └────┼──────►     BLS model    │     │
                                     │      └──────────────────┘     │
                                     │                               │
                                     └───────────────────────────────┘
```

Check out the [on-demand recording](https://info.nvidia.com/indepth-review-triton-inference-webinar.html?ondemandrgt=yes#) to learn more details, about what's happening in the code.

## Topics
* Containers (TensorRT, Triton, SDK and Client) overview
* Create a MNIST PyTorch model and export it to ONNX and PyTorch TorchScript formats
* Optimize it using TensorRT
* Setup Triton to serve TensorRT optimized model, ONNX model and the BLS example model
* Create demo client app and test it
* Demonstrate perf_analyzer and model_analyzer


## Commands
1. Clone this repository into the location, which isn't mounted. Otherwise, docker won't be able to mount it to the container.
```
git clone https://github.com/okdimok/triton_model_analyzer.git
cd triton_model_analyzer/triton_webinar
```

2. Run the PyTorch container
```
docker run --rm --gpus all --shm-size=1g --ulimit memlock=-1 \
 --name pytorch_trt \
 -v "$(pwd)":/workspace/ext -ti nvcr.io/nvidia/pytorch:23.05-py3
cd /workspace/ext/onnx_export
```

3. Inside the container, train the model and save it to `.pt` and to `.onnx`
```
python sample.py
```

4. Convert the onnx model to the `.trt` as it is.
```
trtexec --onnx=model.onnx --saveEngine=model_batch1.trt --exportProfile=trt.log
```
Note the log has the line
```
[07/19/2023-08:44:55] [W] Dynamic dimensions required for input: input, but no shapes were provided. Automatically overriding shape to: 1x1x28x28
```
That's becuase we haven't specified the target shape for the batch dimension. Let's it's in our best interest to allow as big batch as possible. To specify it, we can check the trtexec options via the cmd

```
trtexec -h | tee trtexec_help.txt
grep -I Shape -A9 trtexec_help.txt
```
The response we get:
```
  --minShapes=spec                   Build with dynamic shapes using a profile with the min shapes provided
  --optShapes=spec                   Build with dynamic shapes using a profile with the opt shapes provided
  --maxShapes=spec                   Build with dynamic shapes using a profile with the max shapes provided
  --minShapesCalib=spec              Calibrate with dynamic shapes using a profile with the min shapes provided
  --optShapesCalib=spec              Calibrate with dynamic shapes using a profile with the opt shapes provided
  --maxShapesCalib=spec              Calibrate with dynamic shapes using a profile with the max shapes provided
                                     Note: All three of min, opt and max shapes must be supplied.
                                           However, if only opt shapes is supplied then it will be expanded so
                                           that min shapes and max shapes are set to the same values as opt shapes.
                                           Input names can be wrapped with escaped single quotes (ex: 'Input:0').
                                     Example input shapes spec: input0:1x3x256x256,input1:1x3x128x128
                                     Each input shape is supplied as a key-value pair where key is the input name and
                                     value is the dimensions (including the batch dimension) to be used for that input.
                                     Each key-value pair has the key and value separated using a colon (:).
                                     Multiple input shapes can be provided via comma-separated key-value pairs.
```

So, we specify
```
trtexec --onnx=model.onnx --saveEngine=model.trt --exportProfile=trt.log \
     --minShapes=input:1x1x28x28 \
     --optShapes=input:64x1x28x28 `# opt stands for optimal` \
     --maxShapes=input:128x1x28x28
```
Now we can exit the container with `Ctrl+D`

5. Copy the models created to the model repository

```
cp onnx_export/model.trt inference/model_repo/mnist_trt/1/model.plan
cp onnx_export/model.onnx inference/model_repo/mnist_onnx/1/model.onnx
cp onnx_export/model.pt inference/model_repo/mnist_pt/1/model.pt
```

6. Start Triton Inference Server
```
docker run --rm --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
-p 8000:8000 -p 8001:8001 -p 8002:8002 `# theese are default ports for HTTP, GRPC and metrics` \
--name triton_server \
-v "$(pwd)":/workspace/ext -ti nvcr.io/nvidia/tritonserver:23.05-py3 \
tritonserver --model-repository /workspace/ext/inference/model_repo \
--log-verbose 4 `# this is a very detailed log, that may affect performance`
```

7. Now in another terminal start the container with the Triton clients
```
cd triton_model_analyzer/triton_webinar
docker run --gpus all --rm -it -v /var/run/docker.sock:/var/run/docker.sock \
--name triton_clients \
-v "$(pwd)":/workspace/ext \
-v "$(pwd)":"$(pwd)" --env EXTWD="$(pwd)" \
--net host nvcr.io/nvidia/tritonserver:23.05-py3-sdk
cd /workspace/ext
```

8. Let's first check the config of the models loaded (see [1](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_model_repository.md) and [2](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_model_configuration.md))
```
curl -X POST http://127.0.0.1:8000/v2/repository/index
curl http://127.0.0.1:8000/v2/models/mnist_trt/config | jq
```

9. Now let's run our very simple client
```
cd /workspace/ext/client
python client.py
cd /workspace/ext
```

It should predict 7, the correct answer.


10. Now let's measure the approximate performance of the model with perf_analyzer. One can save perf_analyzer help for later usage:
```
perf_analyzer --help 2>&1 | tee perf_analyzer_help.txt
```

11. Then measure the performance of the trt model

```
perf_analyzer -m mnist_trt --shape "input:1,28,28"
```
My output:
```
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 1546.36 infer/sec, latency 645 usec
```

The command above is using the inefficient HTTP. The optimal launch will probably use GRPC, and use several streams to rely on dynamic batching:

```
perf_analyzer -m mnist_trt --shape "input:1,28,28" \
-b 1 `# we still specify batch = 1 and rely on server batching, not on the client one` \
-i gRPC \
--concurrency-range 1:257:64
```

My output:
```
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 1464.21 infer/sec, latency 682 usec
Concurrency: 65, throughput: 30403.9 infer/sec, latency 2136 usec
Concurrency: 129, throughput: 28366.5 infer/sec, latency 4545 usec
Concurrency: 193, throughput: 31912.4 infer/sec, latency 6045 usec
Concurrency: 257, throughput: 35058.6 infer/sec, latency 7328 usec
```

We see dynamic batching adds 20x to the model performance. 
See [Recommended Configuration Process](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#recommended-configuration-process) on how to optimize the performace of the model further.

12. Let's compare the trt, the onnx and the pt model on concurrency 65.

```
perf_analyzer -m mnist_onnx --shape "input:1,28,28" \
-b 1 `# we still specify batch = 1 and rely on server batching, not on the client one` \
-i gRPC \
--concurrency-range 65
```

```
Inferences/Second vs. Client Average Batch Latency
Concurrency: 65, throughput: 26951.8 infer/sec, latency 2410 usec
```

```
perf_analyzer -m mnist_pt --shape "input:1,28,28" \
-b 1 `# we still specify batch = 1 and rely on server batching, not on the client one` \
-i gRPC \
--concurrency-range 65
```
```
Inferences/Second vs. Client Average Batch Latency
Concurrency: 65, throughput: 4799.15 infer/sec, latency 13543 usec
```

Finally, these are the results of the comparison for me. We see TensorRT gives the highest throughput having the lowest latency.

| Framework           | Throughput (infer/sec) | Throuput Relative | Latency    | Latency Relative |
|---------------------|------------|-------------------|------------|------------------|
| PyTorch TorchScript | 4799.15    | 1X                | 13543 usec |  6.3X            |
| ONNX Runtime        | 26951.8    | 5.6X              | 2410 usec  |  1.12X           |
| TensorRT            | 30403.9    | 6.3X              | 2136 usec  |  1X              |

13. Finally, this is a sample model analyzer command, to seek through the trt model configurations. For that, stop the Triton Server Container and then in the client container run. See the [docs](https://github.com/triton-inference-server/model_analyzer/blob/main/docs/config_search.md) for further details.

```
cd "${EXTWD}"
model-analyzer profile --help  | tee model_analyzer_profile_help.txt
model-analyzer profile --model-repository "${EXTWD}/inference/model_repo" \
--profile-models mnist_trt --triton-launch-mode=docker \
--output-model-repository-path "${EXTWD}/inference/output" \
--run-config-search-max-concurrency 256 \
--run-config-search-min-concurrency 16 \
--run-config-search-max-instance-count 4 \
--run-config-search-max-model-batch-size 128 \
--run-config-search-mode quick \
--override-output-model-repository
```

Check out the files from the end of the log.

The second part of the webinar is available [here](../triton_webinar_pytriton_model_analyzer/README.md)