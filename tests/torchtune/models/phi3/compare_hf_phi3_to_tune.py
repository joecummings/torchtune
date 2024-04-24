# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Must run the following commands before the script is runnable

1. `pip install transformers`
2. `tune download microsoft/Phi-3-mini-128k-instruct --output-dir ./model --ignore-patterns ""`
3. `tune download microsoft/Phi-3-mini-4k-instruct --output-dir ./model --ignore-patterns ""`

"""

import torch
from torchtune.models.phi3 import phi3_mini_128k, phi3_mini_4k
from torchtune.utils import FullModelHFCheckpointer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Check the 4K model
with torch.device("cuda"):
    hf_model_4k = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True
    ).eval()

with torch.device("cuda"):
    tune_model_4k = phi3_mini_4k().eval()
    checkpointer = FullModelHFCheckpointer(
        checkpoint_dir="./model",
        checkpoint_files=[
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ],
        model_type="phi3",
        output_dir="./",
    )
    state_dict = checkpointer.load_checkpoint()
    tune_model_4k.load_state_dict(state_dict["model"])

import pdb
pdb.set_trace()

for i in range(5):
    with torch.no_grad():
        inputs = torch.randint(0, 32_000, (4, 2000))
        hf_output = hf_model_4k(inputs).get("logits")
        tune_output = tune_model_4k(inputs)
        try:
            assert torch.allclose(hf_output, tune_output)
        except:
            import pdb
            pdb.set_trace()


# Check the 128K model
hf_model_4k = AutoModelForCasualLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
tune_model_4k = phi3_mini_128k()
checkpointer = ...
state_dict = checkpointer.load_checkpoint()
with torch.device("cuda"):
    tune_model_128k.load_state_dict(state_dict["model"])

for _ in range(5):
    inputs = torch.randint(0, 32_000, (4, 8000))
    hf_output = hf_model_128k(inputs)
    tune_output = tune_model_128k(inputs)
    assert hf_output == tune_output
