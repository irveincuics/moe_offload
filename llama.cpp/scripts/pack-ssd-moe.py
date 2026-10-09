#!/usr/bin/env python3
"""Pack Qwen3 MoE expert bytes without changing the source GGUF."""
import argparse
import hashlib
import os
from pathlib import Path
import re
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gguf-py"))
import numpy as np
from gguf import GGUFReader, GGUFWriter

ALIGNMENT = 4096
PROJECTIONS = ("gate", "up", "down")


def inventory(reader):
    if reader.fields["general.architecture"].contents() != "qwen3moe":
        raise ValueError("only qwen3moe is supported")
    if "split.count" in reader.fields and reader.fields["split.count"].contents() != 1:
        raise ValueError("merge the split GGUF before packing")
    if sys.byteorder != "little" or reader.byte_order == "S":
        raise ValueError("packing requires little-endian GGUF on a little-endian host")
    layers = {}
    for tensor in reader.tensors:
        match = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight", tensor.name)
        if match:
            layers.setdefault(int(match[1]), {})[match[2]] = tensor
    if sorted(layers) != list(range(int(reader.fields["qwen3moe.block_count"].contents()))):
        raise ValueError("missing expert layers")
    records = []
    for layer, tensors in sorted(layers.items()):
        if set(tensors) != set(PROJECTIONS):
            raise ValueError(f"layer {layer}: expected gate, up and down")
        group = [tensors[p] for p in PROJECTIONS]
        experts = int(group[0].shape[2]) if len(group[0].shape) == 3 else 0
        if not experts or any(len(t.shape) != 3 or int(t.shape[2]) != experts or t.n_bytes % experts for t in group):
            raise ValueError(f"layer {layer}: invalid expert dimensions")
        sizes = [t.n_bytes // experts for t in group]
        record_size = sum(size + min(size, 512) for size in sizes)
        record_size = (record_size + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        records.append((layer, group, experts, sizes, record_size))
    return records


def record_bytes(source, group, experts, sizes, record_size, expert):
    data = np.zeros(record_size, dtype=np.int8)
    cursor = 0
    for tensor, size in zip(group, sizes):
        guard = min(size, 512)
        length = size + (guard if expert + 1 < experts else 0)
        source.seek(tensor.data_offset + expert * size)
        view = memoryview(data).cast("B")[cursor:cursor + length]
        if source.readinto(view) != length:
            raise ValueError(f"short source read: {tensor.name}, expert {expert}")
        cursor += size + guard
    return data


def pack(model, output):
    model, output = Path(model), Path(output)
    if output.exists():
        raise FileExistsError(output)
    before = model.stat()
    reader = GGUFReader(model)
    records = inventory(reader)
    fd, temporary = tempfile.mkstemp(prefix=output.name + ".partial-", dir=output.parent)
    os.close(fd)
    writer = None
    try:
        writer = GGUFWriter(temporary, "ssd-moe-pack")
        writer.add_custom_alignment(ALIGNMENT)
        writer.add_uint64("ssd_pack.version", 1)
        writer.add_uint64("ssd_pack.source_size", before.st_size)
        writer.add_uint64("ssd_pack.source_mtime_ns", before.st_mtime_ns)
        writer.add_string("ssd_pack.source_name", model.name)
        for layer, group, experts, sizes, record_size in records:
            for tensor in group:
                prefix = "ssd_pack." + tensor.name + "."
                writer.add_uint64(prefix + "type", int(tensor.tensor_type))
                writer.add_uint64(prefix + "source_offset", tensor.data_offset)
                for dim, value in enumerate(tensor.shape):
                    writer.add_uint64(prefix + f"ne{dim}", int(value))
            for expert in range(experts):
                writer.add_tensor_info(f"blk.{layer}.expert.{expert}", (record_size,), np.dtype(np.int8), record_size)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        with model.open("rb", buffering=0) as source:
            for layer, group, experts, sizes, record_size in records:
                for expert in range(experts):
                    writer.write_tensor_data(record_bytes(source, group, experts, sizes, record_size, expert))
                print(f"packed layer {layer}: {experts} records, {record_size} bytes/record", flush=True)
        for stream in writer.fout:
            stream.flush()
            os.fsync(stream.fileno())
        writer.close()
        writer = None
        after = model.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino, before.st_dev) != (after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev):
            raise ValueError("source changed during packing")
        # Publish without replacing an existing output, including a concurrent writer.
        os.link(temporary, output)
    finally:
        if writer is not None:
            writer.close()
        os.unlink(temporary)
    print(f"created {output} ({output.stat().st_size} bytes)")


def verify(model, packed):
    model = Path(model)
    source_reader, pack_reader = GGUFReader(model), GGUFReader(packed)
    records = inventory(source_reader)
    stat = model.stat()
    for key, value in (("version", 1), ("source_size", stat.st_size), ("source_mtime_ns", stat.st_mtime_ns)):
        if pack_reader.fields["ssd_pack." + key].contents() != value:
            raise ValueError(f"source/pack metadata mismatch: {key}")
    tensors = {t.name: t for t in pack_reader.tensors}
    if len(tensors) != sum(r[2] for r in records):
        raise ValueError("incorrect record count")
    count = 0
    digest = hashlib.sha256()
    # Reader alignment metadata can yield a numpy uint32. Offsets exceed 4 GiB.
    expected_offset = int(pack_reader.data_offset)
    with model.open("rb", buffering=0) as source:
        for layer, group, experts, sizes, record_size in records:
            for tensor in group:
                prefix = "ssd_pack." + tensor.name + "."
                values = {"type": int(tensor.tensor_type), "source_offset": tensor.data_offset}
                values.update({f"ne{dim}": int(value) for dim, value in enumerate(tensor.shape)})
                for key, value in values.items():
                    if pack_reader.fields[prefix + key].contents() != value:
                        raise ValueError(f"tensor metadata mismatch: {prefix + key}")
            for expert in range(experts):
                tensor = tensors[f"blk.{layer}.expert.{expert}"]
                expected = record_bytes(source, group, experts, sizes, record_size, expert)
                if tensor.data_offset != expected_offset or tensor.data_offset % ALIGNMENT or tensor.n_bytes != record_size or not np.array_equal(tensor.data.view(np.int8), expected):
                    raise ValueError(f"record mismatch: {tensor.name}")
                digest.update(memoryview(expected))
                expected_offset += record_size
                count += 1
    if expected_offset != Path(packed).stat().st_size:
        raise ValueError("unexpected trailing or missing data")
    if model.stat().st_mtime_ns != stat.st_mtime_ns:
        raise ValueError("source changed during verification")
    print(f"verified {count} records byte-for-byte; payload sha256={digest.hexdigest()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify", action="store_true", help="verify an existing pack against the source")
    args = parser.parse_args()
    if args.verify:
        verify(args.model, args.output)
    else:
        pack(args.model, args.output)


if __name__ == "__main__":
    main()
