import numpy as np
from PIL import Image
import tritonclient.http as httpclient
import argparse

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        "-i",
        default="image.png",
        type=str,
        help="Specify input image.",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="mnist_onnx",
        type=str,
        help="Specify the target model.",
    )
    parser.add_argument(
        "--input-name",
        "-n",
        default="input",
        type=str,
        help="Specify the input name.",
    )
    parser.add_argument(
        "--output-name",
        "-o",
        default="output",
        type=str,
        help="Specify the output name.",
    )
    return parser.parse_args()

def preprocess(image):
    image = image.resize((28, 28))
    image = np.array(image).astype(np.float32)
    image = image.reshape(1, 28, 28, 1)
    image = image.transpose(0, 3, 1, 2)

    return image

def postprocess_output(preds):
    return np.argmax(np.squeeze(preds))

if __name__ == '__main__':
    args = _parse_args()
    image = Image.open(args.image)
    image = preprocess(image)


    # Create the inference context for the model.
    model_name = args.model
    model_version = 1
    triton_client = httpclient.InferenceServerClient(url="localhost:8000")
    inputs = []
    outputs = []
    inputs.append(httpclient.InferInput(args.input_name, [1, 1, 28, 28], "FP32"))
    inputs[0].set_data_from_numpy(image)
    outputs.append(httpclient.InferRequestedOutput(args.output_name))
    results = triton_client.infer(model_name, inputs, outputs=outputs)
    output = np.squeeze(results.as_numpy(args.output_name))
    pred = postprocess_output(output)

    print(f"HTTP Service | Prediction: {pred}")

