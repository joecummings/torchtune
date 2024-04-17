# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List

from torchtune.models.mixtral._component_builders import mixtral, lora_mixtral

from torchtune.modules import Tokenizer, TransformerDecoder
from torchtune.modules.peft import LORA_ATTN_MODULES
from functools import partial


"""
Model builders build specific instantiations using component builders. For example
the ``mixtral_8x7b`` model builder uses the ``mixtral`` component builder.
"""


def mixtral_8x7b() -> TransformerDecoder:
    """
    Builder for creating a Mixtral 8x7B model initialized w/ the default 7b parameter values
    from https://mistral.ai/news/mixtral-of-experts/


    Returns:
        TransformerDecoder: Instantiation of Mixtral 8x7B model
    """
    return mixtral(
        num_experts=8,
        num_experts_per_token=2,
        vocab_size=32_000,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        intermediate_dim=14336,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-5,
    )


def mixtral_tokenizer(path: str) -> Tokenizer:
    tokenizer = Tokenizer.from_file(path)
    # Original tokenizer has no pad_id, which causes indexing errors when batch training
    tokenizer.pad_id = 0
    return tokenizer


def lora_mixtral_8x7b(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Builder for creating a Mixtral 8x7B model with LoRA enabled.

    Args:
        lora_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        apply_lora_to_output (bool): whether to apply LoRA to the model's final output projection.
            Default: False
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        quantize_base (bool): Whether to quantize base model weights

    Raises:
        RuntimeError: If ``quantize_base`` is True. LoRA quantization is not currently supported.

    Returns:
        TransformerDecoder: Instantiation of Mixtral 8x7B model with LoRA applied
    """
    return lora_mixtral(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        num_experts=8,
        num_experts_per_token=2,
        vocab_size=32_000,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        intermediate_dim=14336,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=10_000,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        quantize_base=quantize_base,
    )
