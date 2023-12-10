#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
from typing import Iterable

import torch
import numpy as np
from nemo.collections.tts.models import HifiGanModel

import model_navigator as nav
from model_navigator.api.config import Sample


class TTSWrapper(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        model.eval()
        self.model = model

    def forward(self, spec):
        # NeMo defines special forward for export
        return self.model.forward_for_export(spec)

def get_model():
    model = HifiGanModel.from_pretrained(model_name="tts_en_hifigan")
    return TTSWrapper(model)

def get_dataloader():
    return [{"spec": np.random.randn(128, 80, 256).astype(np.float32)} for _ in range(100)]

def get_dynamic_axes():
    return {"spec": {0: "batchsize", 2: "seqlen"}}

def get_verify_function():
    """Define verify function that compares outputs of the torch model and the optimized model."""
    def verify_func(ys_runner: Iterable[Sample], ys_expected: Iterable[Sample]) -> bool:
        for y_runner, y_expected in zip(ys_runner, ys_expected):
            if not all(
                np.allclose(a, b, rtol=1.0e-3, atol=1.0e-3) for a, b in zip(y_runner.values(), y_expected.values())
            ):
                return False
        return True

    return verify_func

# API docs: https://triton-inference-server.github.io/model_navigator/latest/package/package_optimize_api/
package = nav.torch.optimize(
    model=get_model(),
    input_names=["spec"],
    output_names=["audio"],
    dataloader=get_dataloader(),
    verify_func=get_verify_function(),  # verify_func is optional but recommended.
    custom_configs=[
        nav.OnnxConfig(opset=17, dynamic_axes=get_dynamic_axes()),
    ],
)

nav.package.save(package=package, path="tts_hifigan.nav")
