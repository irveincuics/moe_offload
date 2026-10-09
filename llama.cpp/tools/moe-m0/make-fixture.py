#!/usr/bin/env python3
"""Small random Qwen3 MoE fixture. Functional testing only, never benchmark evidence."""
import json
from pathlib import Path
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType

ROOT = Path("/ssd/test/m0/fixture")
SOURCE = Path("/ssd/models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-BF16.gguf")


def main():
    if not Path("/.dockerenv").exists():
        raise RuntimeError("run inside moe-llama-dev")
    ROOT.mkdir(exist_ok=False)
    source = GGUFReader(SOURCE)
    output = ROOT / "random-qwen3moe.gguf"
    writer = GGUFWriter(output, "qwen3moe")
    for key, field in source.fields.items():
        if key.startswith("tokenizer."):
            writer.add_key_value(key, field.contents(), field.types[0],
                                 field.types[-1] if field.types[0] == GGUFValueType.ARRAY else None)
    hidden, intermediate, experts, layers = 64, 128, 16, 2
    metadata = {"context_length": 4096, "embedding_length": hidden, "block_count": layers,
                "feed_forward_length": intermediate * 2, "expert_feed_forward_length": intermediate,
                "expert_count": experts, "expert_used_count": 2, "attention.head_count": 4,
                "attention.head_count_kv": 2, "attention.key_length": 16, "attention.value_length": 16,
                "rope.dimension_count": 16}
    for key, value in metadata.items():
        writer.add_uint32("qwen3moe." + key, value)
    writer.add_float32("qwen3moe.attention.layer_norm_rms_epsilon", 1e-6)
    writer.add_float32("qwen3moe.rope.freq_base", 1000000.0)
    writer.add_name("M0 RANDOM FIXTURE - NOT A QUALITY OR PERFORMANCE MODEL")
    rng = np.random.default_rng(20260928)
    vocab_size = len(source.fields["tokenizer.ggml.tokens"].data)
    def tensor(name, shape, norm=False):
        data = np.ones(shape, dtype=np.float32) if norm else (rng.standard_normal(shape) * .02).astype(np.float32)
        writer.add_tensor(name + ".weight", data)
    tensor("token_embd", (vocab_size, hidden))
    tensor("output_norm", (hidden,), True)
    tensor("output", (vocab_size, hidden))
    for layer in range(layers):
        prefix = f"blk.{layer}."
        for name in ("attn_norm", "ffn_norm"):
            tensor(prefix + name, (hidden,), True)
        for name in ("attn_q_norm", "attn_k_norm"):
            tensor(prefix + name, (16,), True)
        tensor(prefix + "attn_q", (64, hidden))
        tensor(prefix + "attn_k", (32, hidden))
        tensor(prefix + "attn_v", (32, hidden))
        tensor(prefix + "attn_output", (hidden, 64))
        tensor(prefix + "ffn_gate_inp", (experts, hidden))
        for name in ("ffn_up_exps", "ffn_gate_exps"):
            tensor(prefix + name, (experts, intermediate, hidden))
        tensor(prefix + "ffn_down_exps", (experts, hidden, intermediate))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    (ROOT / "manifest.json").write_text(json.dumps({"purpose": "random-weight functional fixture only",
                                                     "tokenizer_source": str(SOURCE), "seed": 20260928}, indent=2))
    print(output)


if __name__ == "__main__":
    main()
