#!/usr/bin/env python
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import threading
from queue import Queue
from threading import Lock

import numpy as np
import torch  # pytype: disable=import-error
import torch.nn.functional as functional  # pytype: disable=import-error
import torch.optim as optim  # pytype: disable=import-error
from torch.optim.lr_scheduler import StepLR  # pytype: disable=import-error

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig

from model import Net  # pytype: disable=import-error # isort:skip

LOGGER = logging.getLogger("examples.online_learning_mnist.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

class InferModel:
    def __init__(self) -> None:
        self.device = torch.device("cuda")
        self.model = Net()
        checkpoint = torch.load('model.ckpt')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
    @batch
    def infer(self, image):
        """Infer function is used in inference endpoint."""
        data_tensor = torch.from_numpy(image).to(self.device)
        res = self.model(data_tensor)
        res = res.numpy(force=True)
        return {'predictions': res}


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging in debug mode.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    log_verbose = 1 if args.verbose else 0
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

    with Triton(config=TritonConfig(log_verbose=log_verbose)) as triton:
        LOGGER.info("Loading model")
        infer_model = InferModel()
        triton.bind(
            model_name="MnistInfer",
            infer_func=infer_model.infer,
            inputs=[
                # image for classification
                # note tensor name matches the kw argument of the function
                Tensor(name="image", dtype=np.float32, shape=(1, 28, 28)),
            ],
            outputs=[
                # predictions taken from softmax layer
                # they're taken as a key from the function output dict
                Tensor(name="predictions", dtype=np.float32, shape=(-1,)),
            ],
            config=ModelConfig(max_batch_size=64),
            strict=True,
        )

        LOGGER.info("Serving model")
        triton.serve()

if __name__ == "__main__":
    main()