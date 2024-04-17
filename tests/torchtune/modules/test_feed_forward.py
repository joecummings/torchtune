# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest

import torch

from tests.test_utils import assert_expected, fixed_init_model
from torch import nn, Tensor

from torchtune.modules import FeedForward
from torchtune.utils.seed import set_seed


@pytest.fixture(autouse=True)
def random():
    set_seed(0)


class TestFeedForward:
    """Class for testing FFN implementation."""

    @pytest.fixture
    def input_params(self) -> Tuple[int, int]:
        dim = 4096
        hidden_dim = 11008  # Scaled for SwiGLU
        return dim, hidden_dim

    @pytest.fixture
    def input(self, input_params: Tuple[int, int]) -> Tensor:
        dim, _ = input_params
        return torch.randn(1, dim)

    @pytest.fixture
    def ffn(self, input_params: Tuple[int, int]) -> FeedForward:
        dim, hidden_dim = input_params
        gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        down_proj = nn.Linear(hidden_dim, dim, bias=False)
        up_proj = nn.Linear(dim, hidden_dim, bias=False)
        ff = FeedForward(
            gate_proj=gate_proj, down_proj=down_proj, up_proj=up_proj
        ).eval()
        fixed_init_model(ff)
        ff.eval()
        return ff

    def test_forward(self, input: Tensor, ffn: FeedForward) -> None:
        with torch.no_grad():
            x_out = ffn(input)
        assert_expected(x_out.mean(), torch.tensor(251.5356), atol=1e-7, rtol=1e-3)
        assert_expected(x_out.max(), torch.tensor(503.0614), atol=1e-7, rtol=1e-3)


class TestMoELayerForward:
    """Class for testing MoE layer forward pass."""

    @pytest.fixture
    def input_params(self) -> Tuple[int, int]:
        dim = 4096
        num_experts = 16
        num_experts_per_token = 2
        return dim, num_experts, num_experts_per_token

    @pytest.fixture
    def input(self, input_params: Tuple[int, int]) -> Tensor:
        dim, _, _ = input_params
        return torch.randn(1, dim)

    @pytest.fixture
    def moe_layer(self, input_params: Tuple[int, int]) -> MoELayer:
        dim, num_experts, num_experts_per_token = input_params
        expert = nn.Linear(dim, dim, bias=False)
        moe_layer = MoELayer(
            embed_dim=dim,
            num_experts=num_experts,
            expert=expert,
            num_experts_per_token=num_experts_per_token,
        ).eval()
        fixed_init_model(moe_layer)
        moe_layer.eval()
        return moe_layer

    def test_forward(self, input: Tensor, moe_layer: MoELayer) -> None:
        with torch.no_grad():
            x_out = moe_layer(input)
        assert_expected(x_out.mean(), torch.tensor(251.5356), atol=1e-7, rtol=1e-3)
        assert_expected(x_out.max(), torch.tensor(503.0614), atol=1e-7, rtol=1e-3)
