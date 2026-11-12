"""Left-padded batches match single-item runs on a tiny random Llama."""

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from amon.cache.extract import forward_hidden, left_pad


def _tiny_llama():
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        max_position_embeddings=64,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config)
    model.eval()
    return model


def test_batched_left_pad_matches_single_item_runs():
    model = _tiny_llama()
    sequences = [
        [1, 2, 3, 4, 5],
        [6, 7, 8],
        [9, 10, 11, 12],
    ]
    layers = [1, 2, 3]
    ids, mask = left_pad(sequences, pad_id=0)
    with torch.inference_mode():
        batched = forward_hidden(model, ids, mask, layers)
        for row, sequence in enumerate(sequences):
            one_ids, one_mask = left_pad([sequence], pad_id=0)
            single = forward_hidden(model, one_ids, one_mask, layers)
            for layer in layers:
                for which in (0, 1):
                    left = batched[layer][which][row]
                    right = single[layer][which][0]
                    cosine = torch.nn.functional.cosine_similarity(left.float(), right.float(), dim=0)
                    assert float(cosine) >= 0.999
