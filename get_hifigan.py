# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from nemo.collections.tts.models import HifiGanModel

model = HifiGanModel.from_pretrained(model_name="tts_hifigan")
model.export("./hifigan.onnx")

model = HifiGanModel.from_pretrained(model_name="tts_hifigan")
model.export("./hifigan.pt")