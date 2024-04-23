# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional

import torch
from torch import nn, Tensor


class LongScaledRotaryPositionalEmbeddings(nn.Module):
    def __init__(
        self,
        dim: int,
        short_factor: int,
        long_factor: int,
        max_seq_len: int = 4096,
        original_max_seq_len: int = 4096,
        base: int = 10000,
        scaling_policy: str = "su",
    ):
        super().__init__()
        self.dim = dim
        self.short_factor = short_factor
        self.long_factor = long_factor
        self.max_seq_len = max_seq_len
        self.original_max_seq_len = original_max_seq_len

        if scaling_policy == "su":
            self.mscale = self._calc_mscale_su
        elif scaling_policy == "yarn":
            self.mscale = self._calc_mscale_yarn
        else:
            self.mscale = self._noop_calc_mscale

    def _noop_calc_mscale(self, scale: int) -> float:
        return float(scale)

    def _calc_mscale_su(self, scale: int) -> float:
        if scale <= 1.0:
            return 1.0
        return math.sqrt(1 + math.log(scale) / math.log(self.original_max_seq_len))

    def _calc_mscale_yarn(self, scale: int) -> float:
        if scale <= 1.0:
            return 1.0
        return 0.1 * math.log(scale) + 1.0

    @torch.no_grad()
    def forward(self, x: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        seq_len = x.size(1)

        # ------ set up cache
        if seq_len > self.original_max_seq_len:
            seq_idx = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            rescale_factors = torch.tensor(
                self.long_factor, device=x.device, dtype=torch.float32
            )
        else:
            seq_idx = torch.arange(
                self.original_max_seq_len, device=x.device, dtype=torch.float32
            )
            rescale_factors = torch.tensor(
                self.short_factor, device=x.device, dtype=torch.float32
            )

        assert rescale_factors.shape == self.dim // 2

        theta = (
            1.0
            / rescale_factors
            * (
                self.base
                ** (
                    torch.arange(0, self.dim, 2)[: (self.dim // 2)] / float() / self.dim
                )
            )
        )
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        mscale = self.mscale(self.max_seq_len / self.original_max_seq_len)

        cache = torch.stack(
            [torch.cos(idx_theta) * mscale, torch.sin(idx_theta) * mscale], dim=-1
        )

        # ------ actually apply forward
        rope_cache = cache[:seq_len] if input_pos is None else cache[input_pos]

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, n_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [1, s, 1, n_d // 2, 2]
        rope_cache = rope_cache.view(1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, n_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, n_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)
