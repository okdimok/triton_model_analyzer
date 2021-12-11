# /usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

curl -kv -X POST 'http://127.0.0.1:8000/v2/models/hifigan/infer' \
 -H 'accept: application/json' \
 -H 'Content-Type: application/octet-stream' \
 -H 'connection: keep-alive' \
 -d @hifigan_curl_data.json
