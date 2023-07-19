#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys

# This sample uses an MNIST PyTorch model to create a TensorRT Inference Engine
import model
import numpy as np
import torch

import tensorrt as trt

sys.path.insert(1, os.path.join(sys.path[0], ".."))

import pdb
# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class ModelData(object):
    INPUT_NAME = "data"
    INPUT_SHAPE = (1, 1, 28, 28)
    OUTPUT_NAME = "prob"
    OUTPUT_SIZE = 10
    DTYPE = trt.float32


# Loads a random test case from pytorch's DataLoader
def load_random_test_case(model, pagelocked_buffer):
    # Select an image at random to be the test case.
    img, expected_output = model.get_random_testcase()
    # Copy to the pagelocked input buffer
    np.copyto(pagelocked_buffer, img)
    return expected_output

def export_model(model):
    dummy_input = torch.randn([1, 1, 28, 28]).to('cpu')
    input_names = ['input']
    output_names = ['output']

    # variable batch_size 
    dynamic_axes= {'input':{ 0:'batch_size'}, 'output':{ }}

    torch.onnx.export(
        model,
        dummy_input,
        f="model.onnx",
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )

    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted.save('model.pt') # Save

def main():
    # Train the PyTorch model
    mnist_model = model.MnistModel()
    mnist_model.learn()
    export_model(mnist_model.network)

if __name__ == "__main__":
    main()
