from typing import List

from torchtune.models.phi3._component_builders import phi3
from torchtune.models.phi3._sentencepiece import SentencePieceTokenizer

from torchtune.modules import TransformerDecoder
from torchtune.modules.peft import LORA_ATTN_MODULES
from functools import partial

import torch


"""
Model builders build specific instantiations using component builders. For example
the ``phi3_mini`` model builder uses the ``phi3`` component builder.
"""


def phi3_mini() -> TransformerDecoder:
    """
    Builder for creating the Phi3 Mini 4K Instruct Model.
    Ref: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct

    Returns:
        TransformerDecoder: Instantiation of Phi3 Mini 4K Instruct Model
    """
    return phi3(
        vocab_size=32_064,
        num_layers=32,
        num_heads=32,
        num_kv_heads=32,
        embed_dim=3072,
        intermediate_dim=8192,
        max_seq_len=4096,
        attn_dropout=0.0,
        norm_eps=1e-5,
    )

def phi3_tokenizer(path: str) -> SentencePieceTokenizer:
    """Phi-3 Mini tokenizer.

    Args:
        path (str): Path to the SPM tokenizer model.

    Note:
        This tokenizer includes typical LM EOS and BOS tokens like
        <s>, </s>, and <unk>. However, to support chat completion,
        it is also augmented with special tokens like <|endoftext|>
        and <|assistant|>.

    Returns:
        SentencePieceTokenizer: Instantiation of the SPM tokenizer.
    """
    tokenizer = SentencePieceTokenizer(path)
    return tokenizer
