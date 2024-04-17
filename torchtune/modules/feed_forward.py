# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import copy

from torch import nn, Tensor


class FeedForward(nn.Module):
    """This class implements the feed-forward network derived from Llama2.

    Args:
        gate_proj (nn.Module): Projection from input dim to hidden dim, fed through activation
            and multiplied by up_proj.
        down_proj (nn.Module): Final projection to output dim.
        up_proj (nn.Module): Projection from input dim to hidden dim, multiplied by
            activation(gate_proj).
        activation (nn.Module): Activation function to use. Default is nn.SiLU().
    """

    def __init__(
        self,
        *,
        gate_proj: nn.Module,
        down_proj: nn.Module,
        up_proj: nn.Module,
        activation: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        self.w1 = gate_proj
        self.w2 = down_proj
        self.w3 = up_proj
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.activation(self.w1(x)) * self.w3(x))


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Return a list of ``n`` identical layers.

    Args:
        module (nn.Module): module to be cloned
        n (int): number of clones

    Returns:
        nn.ModuleList: list of ``n`` identical layers
    """
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(n)])


class MoELayer(nn.Module):
    """This class implements the MoE layer derived from Mixtral.

    Args:
        embed_dim (int): input dimension to the gate
        num_experts (int): number of experts
        expert (nn.Module): expert module (usually a normal MLP)
        num_experts_per_token (int): number of experts to select per token

    Examples:
        >>> mlp = FeedForward(
                gate_proj=nn.Linear(8, 8),
                down_proj=nn.Linear(8, 8),
                up_proj=nn.Linear(8, 8)
            )
        >>> MoELayer(embed_dim=4096, num_experts=16, expert=mlp, num_experts_per_token=2)
        MoELayer(
            (gate): Linear(in_features=4096, out_features=16, bias=False)
            (experts): ModuleList(
                (0-15): 16 x FeedForward(
                (w1): Linear(in_features=8, out_features=8, bias=True)
                (w2): Linear(in_features=8, out_features=8, bias=True)
                (w3): Linear(in_features=8, out_features=8, bias=True)
                (activation): SiLU()
                )
            )
        )

    """
    def __init__(
        self,
        embed_dim: int,
        num_experts: int,
        expert: nn.Module,
        num_experts_per_token: int,
    ) -> None:
        super().__init__()
        self.gate = nn.Linear(embed_dim, num_experts, bias=False)
        self.experts = _get_clones(expert, num_experts)
        self.num_experts_per_token = num_experts_per_token

    def forward(self, x: Tensor) -> Tensor:
        """Forward method for sparse mixture of experts model"""
        bsz, seq_len, embed_dim = x.shape
        # shape: [b*s, d]
        x = x.view(-1, embed_dim)

        # shape: [b, s, num_experts]
        router = self.gate(x)

        # select experts
        probs, selected_experts = torch.topk(router, self.num_experts_per_token)
        probs = probs.softmax(dim=1).type_as(x)

        # shape: [b*s, num_experts_per_token, d]
        expert_outputs = torch.stack(
            [
                self.experts[i](x[selected_experts == i])
                for i in range(self.num_experts)
            ],
            dim=2,
        )

        # shape: [b*s, d]
        output = torch.sum(expert_outputs * probs[:, :, None], dim=2)

        return output.view(bsz, seq_len, -1)
