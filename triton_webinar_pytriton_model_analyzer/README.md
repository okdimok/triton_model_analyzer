This is the demo for the second part of the inference webinar series. See the first demo [here](../triton_webinar/README.md) for the basic understanding of Triton.

## Topics
* Create a MNIST PyTorch model
* Wrap it into PyTriton and serve
* Export the model to ONNX and TensorRT 
* Setup Triton to serve TensorRT optimized model, and the BLS example model
* Demonstrate perf_analyzer and model_analyzer


## Commands

### Prepare the repo and models
1. Clone this repository into the location, which isn't mounted. Otherwise, docker won't be able to mount it to the container.
```bash
git clone https://github.com/okdimok/triton_model_analyzer.git
cd triton_model_analyzer/triton_webinar_pytriton_model_analyzer
```

2. Pull all the required docker containers
```bash
docker pull nvcr.io/nvidia/pytorch:23.11-py3 
docker pull nvcr.io/nvidia/tritonserver:23.11-py3 
docker pull nvcr.io/nvidia/tritonserver:23.11-py3-sdk 
```

3. Run the PyTorch container.
```bash
docker run -it --rm --gpus all --shm-size=1g --ulimit memlock=-1 \
 --name pytorch_trt \
 -p 8000-8002:8000-8002 \
 -v "$(pwd -P)":/workspace/ext \
 nvcr.io/nvidia/pytorch:23.11-py3 \
 bash
```

4. Inside the container, train a simple mnist model, save it to `.pt`, `.onnx` and convert the onnx model to the `.trt`. For more details, see [demo 1](../triton_webinar/README.md)
```bash
cd /workspace/ext/onnx_export; \
 python sample.py; \
 trtexec --onnx=model.onnx --saveEngine=model.trt --exportProfile=trt.log \
     --minShapes=input:1x1x28x28 \
     --optShapes=input:64x1x28x28 `# opt stands for optimal` \
     --maxShapes=input:128x1x28x28
```
### PyTriton 
5. Install PyTriton
```bash
pip install nvidia-pytriton
```

6. Start PyTriton server
```bash
cd /workspace/ext/onnx_export;
python pytriton_server.py
```

### Clients

7. Now in another terminal start the container with the Triton clients
```bash
cd triton_model_analyzer/triton_webinar_pytriton_model_analyzer
docker run --gpus all --rm -it -v /var/run/docker.sock:/var/run/docker.sock \
--name triton_clients \
-v "$(pwd)":/workspace/ext \
-v "$(pwd)":"$(pwd)" --env EXTWD="$(pwd)" \
--net host nvcr.io/nvidia/tritonserver:23.11-py3-sdk \
bash
cd /workspace/ext
```

8. Let's first check the config of the models loaded (see [1](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_model_repository.md) and [2](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_model_configuration.md))
```
curl -X GET http://127.0.0.1:8000/v2/models/stats | jq
curl http://127.0.0.1:8000/v2/models/MnistInfer/config | jq
```

9. Now let's run our very simple client
```
cd /workspace/ext/client
python client.py --model MnistInfer --input-name image --output-name predictions
cd /workspace/ext
```

It should predict 7, the correct answer.

### Perf Analyzer
10. Finally, let's measure the performance of this model, as we serve it.
```bash
perf_analyzer -m MnistInfer --shape "input:1,28,28" \
-b 1 `# we still specify batch = 1 and rely on server batching, not on the client one` \
-i gRPC \
--concurrency-range 1:65:64
```

On A100 I get
```
Concurrency: 1, throughput: 501.624 infer/sec, latency 1992 usec
Concurrency: 65, throughput: 1463.2 infer/sec, latency 44347 usec
```

If you compare these results vs the results in [demo 1](../triton_webinar/README.md), you can see the PyTriton overhead. In that demo, the slowest option, TorchScript, had a througput of 4799.15 infer/sec with concurrency 65. In our case, there are a couple of reasons, why the overhead is big:

* Extensive use of Python itself during computation. If your inference is just loading an inference-optimized model and mostly relies on C-code, the overhead is much smaller.
* The model is small and quick. For the bigger model, more time is spent in the compute itself.
* We do not optimize our model for inference in any way. For example, we don't use TorchScript.

11. Let's analyze this performance a little deeper. For this let's check out the perf_analyzer help and [perf_analyzer docs](https://github.com/triton-inference-server/client/blob/main/src/c++/perf_analyzer/README.md) and specifically [this part](https://github.com/triton-inference-server/client/blob/main/src/c%2B%2B/perf_analyzer/docs/measurements_metrics.md#visualizing-latency-vs-throughput) and also [CLI docs](https://github.com/triton-inference-server/client/blob/main/src/c%2B%2B/perf_analyzer/docs/cli.md). We will first check the basic csv, and then have a look at additional available metrics
```bash
perf_analyzer -m MnistInfer --shape "input:1,28,28" \
-b 1 `# we still specify batch = 1 and rely on server batching, not on the client one` \
-i gRPC \
--concurrency-range 1:257:64 \
-f perf.csv 
perf_analyzer -m MnistInfer --shape "input:1,28,28" \
-b 1 `# we still specify batch = 1 and rely on server batching, not on the client one` \
-i gRPC \
--concurrency-range 1:257:64 \
-f perf_verbose.csv \
--collect-metrics \
--verbose-csv
echo -e "\07" # this is a bell sound
```

12. Following the docs, let's open [the spreadsheet](https://docs.google.com/spreadsheets/d/1S8h0bWBBElHUoLd2SOvQPzZzRiQ55xjyqodm_9ireiw)

We see the performance is indeed limited by the process of computing the outputs, and not by some other overheads. Let's now deploy the `.onnx` and `.trt` versions of the model, to demanstrate the Model Analyzer capabilities.


















First, exit the PyTorch container with `Ctrl+D`

5. Copy the models created to the model repository

```
cd /workspace/ext
mkdir -p inference/model_repo/mnist_trt/1/
mkdir -p inference/model_repo/mnist_onnx/1/
cp onnx_export/model.trt inference/model_repo/mnist_trt/1/model.plan
cp onnx_export/model.onnx inference/model_repo/mnist_onnx/1/model.onnx
```

6. Start Triton Inference Server
```bash
docker run --rm --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
-p 8000-8002:8000-8002 `# theese are default ports for HTTP, GRPC and metrics` \
--name triton_server \
-v "$(pwd)":/workspace/ext -ti nvcr.io/nvidia/tritonserver:23.11-py3 \
tritonserver --model-repository /workspace/ext/inference/model_repo \
--log-verbose 0 `# change this to 4 for a very detailed log, that may affect performance`
```


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