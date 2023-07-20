import numpy as np
from PIL import Image
import tritonclient.http as httpclient

def preprocess(image):
    image = image.resize((28, 28))
    image = np.array(image).astype(np.float32)
    image = image.reshape(1, 28, 28, 1)
    image = image.transpose(0, 3, 1, 2)

    return image

def postprocess_output(preds):
    return np.argmax(np.squeeze(preds))

if __name__ == '__main__':
    image = Image.open('image.png')
    image = preprocess(image)


    # Create the inference context for the model.
    model_name = "mnist_onnx"
    model_version = 1
    triton_client = httpclient.InferenceServerClient(url="localhost:8000")
    inputs = []
    outputs = []
    inputs.append(httpclient.InferInput('input', [1, 1, 28, 28], "FP32"))
    inputs[0].set_data_from_numpy(image)
    outputs.append(httpclient.InferRequestedOutput('output'))
    results = triton_client.infer(model_name, inputs, outputs=outputs)
    output = np.squeeze(results.as_numpy('output'))
    pred = postprocess_output(output)

    print(f"HTTP Service | Prediction: {pred}")

