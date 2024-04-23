# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Optional
from functools import partial

from torch import nn

from torchtune.models.phi3._component_builders import phi3

from torchtune.modules import TransformerDecoder
from torchtune.modules.tokenizers import SentencePieceTokenizer
from torchtune.modules.peft import LORA_ATTN_MODULES


"""
Model builders build specific instantiations using component builders. For example
the llama3_8b model builder uses the llama3 component builder to create the
Llama3 8B model.
"""

LONG_FACTORS = [
    1.0299999713897705,
    1.0499999523162842,
    1.0499999523162842,
    1.0799999237060547,
    1.2299998998641968,
    1.2299998998641968,
    1.2999999523162842,
    1.4499999284744263,
    1.5999999046325684,
    1.6499998569488525,
    1.8999998569488525,
    2.859999895095825,
    3.68999981880188,
    5.419999599456787,
    5.489999771118164,
    5.489999771118164,
    9.09000015258789,
    11.579999923706055,
    15.65999984741211,
    15.769999504089355,
    15.789999961853027,
    18.360000610351562,
    21.989999771118164,
    23.079999923706055,
    30.009998321533203,
    32.35000228881836,
    32.590003967285156,
    35.56000518798828,
    39.95000457763672,
    53.840003967285156,
    56.20000457763672,
    57.95000457763672,
    59.29000473022461,
    59.77000427246094,
    59.920005798339844,
    61.190006256103516,
    61.96000671386719,
    62.50000762939453,
    63.3700065612793,
    63.48000717163086,
    63.48000717163086,
    63.66000747680664,
    63.850006103515625,
    64.08000946044922,
    64.760009765625,
    64.80001068115234,
    64.81001281738281,
    64.81001281738281
]

SHORT_FACTORS = [
    1.05,
    1.05,
    1.05,
    1.1,
    1.1,
    1.1500000000000001,
    1.2000000000000002,
    1.2500000000000002,
    1.3000000000000003,
    1.3500000000000003,
    1.5000000000000004,
    2.000000000000001,
    2.000000000000001,
    2.000000000000001,
    2.000000000000001,
    2.000000000000001,
    2.000000000000001,
    2.000000000000001,
    2.000000000000001,
    2.000000000000001,
    2.000000000000001,
    2.000000000000001,
    2.000000000000001,
    2.000000000000001,
    2.000000000000001,
    2.000000000000001,
    2.000000000000001,
    2.000000000000001,
    2.000000000000001,
    2.000000000000001,
    2.000000000000001,
    2.000000000000001,
    2.0500000000000007,
    2.0500000000000007,
    2.0500000000000007,
    2.1000000000000005,
    2.1000000000000005,
    2.1000000000000005,
    2.1500000000000004,
    2.1500000000000004,
    2.3499999999999996,
    2.549999999999999,
    2.5999999999999988,
    2.5999999999999988,
    2.7499999999999982,
    2.849999999999998,
    2.849999999999998,
    2.9499999999999975
]


def phi3_mini_4k() -> TransformerDecoder:
    """
    Builder for creating a Phi-3 model initialized w/ the default 3.5b (mini) parameter values
    and 4K context length.

    Returns:
        TransformerDecoder: Instantiation of Phi-3 Mini 4K model
    """
    return phi3(
        vocab_size=32_064,
        num_layers=32,
        num_heads=32,
        num_kv_heads=32,
        embed_dim=3072,
        original_max_seq_len=4096,
        max_seq_len=4096,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=10000.0,
    )


def phi3_mini_128k() -> TransformerDecoder:
    """
    Builder for creating a Phi-3 model initialized w/ the default 3.5b (mini) parameter values
    and 128K context length.

    Returns:
        TransformerDecoder: Instantiation of Phi-3 Mini 128K model
    """
    return phi3(
        vocab_size=32_064,
        num_layers=32,
        num_heads=32,
        num_kv_heads=32,
        embed_dim=3072,
        original_max_seq_len=4096,
        max_seq_len=131072,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=10000.0,
        rope_scaling_type="su",
        rope_scaling_factors=(SHORT_FACTORS, LONG_FACTORS),
    )


def phi3_tokenizer(path: str) -> SentencePieceTokenizer:
    sp_tokenizer = SentencePieceTokenizer(path)
    sp_tokenizer.pad_id = 32_000
    return sp_tokenizer


# def lora_llama3_8b(
#     lora_attn_modules: List[LORA_ATTN_MODULES],
#     apply_lora_to_mlp: bool = False,
#     apply_lora_to_output: bool = False,
#     lora_rank: int = 8,
#     lora_alpha: float = 16,
#     quantize_base: bool = False,
# ) -> TransformerDecoder:
#     """
#     Builder for creating a Llama3 8B model with LoRA enabled.

#     The Llama3 defaults are the same as in :func:`~torchtune.models.llama3.llama3_8b`,
#     while LoRA default params are based on
#     https://github.com/tloen/alpaca-lora/blob/8bb8579e403dc78e37fe81ffbb253c413007323f/finetune.py#L41-L43.

#     Args:
#         lora_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
#             LoRA should be applied to in each self-attention block. Options are
#             ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
#         apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
#             Default: False
#         apply_lora_to_output (bool): whether to apply LoRA to the model's final output projection.
#             Default: False
#         lora_rank (int): rank of each low-rank approximation
#         lora_alpha (float): scaling factor for the low-rank approximation
#         quantize_base (bool): Whether to quantize base model weights

#     Returns:
#         TransformerDecoder: Instantiation of Llama3 8B model with LoRA applied
#     """
#     return lora_llama3(
#         lora_attn_modules=lora_attn_modules,
#         apply_lora_to_mlp=apply_lora_to_mlp,
#         apply_lora_to_output=apply_lora_to_output,
#         vocab_size=128_256,
#         num_layers=32,
#         num_heads=32,
#         num_kv_heads=8,
#         embed_dim=4096,
#         max_seq_len=8192,
#         intermediate_dim=14336,
#         attn_dropout=0.0,
#         norm_eps=1e-5,
#         rope_base=500000.0,
#         lora_rank=lora_rank,
#         lora_alpha=lora_alpha,
#         lora_dropout=0.05,
#         quantize_base=quantize_base,
#     )


# def lora_llama3_70b(
#     lora_attn_modules: List[LORA_ATTN_MODULES],
#     apply_lora_to_mlp: bool = False,
#     apply_lora_to_output: bool = False,
#     lora_rank: int = 8,
#     lora_alpha: float = 16,
#     quantize_base: bool = False,
# ) -> TransformerDecoder:
#     """
#     Builder for creating a Llama3 70B model with LoRA enabled.

#     The Llama3 defaults are the same as in :func:`~torchtune.models.llama3.llama3_70b`,
#     while LoRA default params are based on
#     https://github.com/tloen/alpaca-lora/blob/8bb8579e403dc78e37fe81ffbb253c413007323f/finetune.py#L41-L43.

#     Args:
#         lora_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
#             LoRA should be applied to in each self-attention block. Options are
#             ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
#         apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
#             Default: False
#         apply_lora_to_output (bool): whether to apply LoRA to the model's final output projection.
#             Default: False
#         lora_rank (int): rank of each low-rank approximation
#         lora_alpha (float): scaling factor for the low-rank approximation
#         quantize_base (bool): Whether to quantize base model weights

#     Returns:
#         TransformerDecoder: Instantiation of Llama3 8B model with LoRA applied
#     """
#     return lora_llama3(
#         lora_attn_modules=lora_attn_modules,
#         apply_lora_to_mlp=apply_lora_to_mlp,
#         apply_lora_to_output=apply_lora_to_output,
#         vocab_size=128_256,
#         num_layers=80,
#         num_heads=64,
#         num_kv_heads=8,
#         embed_dim=8192,
#         max_seq_len=8192,
#         intermediate_dim=28672,
#         attn_dropout=0.0,
#         norm_eps=1e-5,
#         rope_base=500000.0,
#         lora_rank=lora_rank,
#         lora_alpha=lora_alpha,
#         lora_dropout=0.05,
#         quantize_base=quantize_base,
#     )


# qlora_llama3_8b = partial(lora_llama3_8b, quantize_base=True)

# qlora_llama3_8b.__doc__ = """
# Builder for creating a Llama3 model with QLoRA enabled. Base model weights in linear layers
# that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
# Please see `lora_llama3_8b` for full API arguments.
# """
