# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from nemo.collections.tts.models import HifiGanModel

# print("HifiGanModel available models:")
# print(HifiGanModel.list_available_models())
# print("#" * 10)

model = HifiGanModel.from_pretrained(model_name="tts_en_hifigan")
model.export("./hifigan.onnx")

model = HifiGanModel.from_pretrained(model_name="tts_en_hifigan")
model.export("./hifigan.pt")