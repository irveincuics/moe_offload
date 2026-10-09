#!/usr/bin/env python3
"""Functional tests for the lossless SSD expert pack format."""
import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("ssd_pack", Path(__file__).with_name("pack-ssd-moe.py"))
pack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack)
from gguf import GGMLQuantizationType


class PackTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.model = self.root / "source.gguf"
        self.output = self.root / "packed.gguf"

    def tearDown(self):
        self.directory.cleanup()

    def fixture(self, bf16=False, missing=False):
        writer = pack.GGUFWriter(self.model, "qwen3moe")
        writer.add_uint32("qwen3moe.block_count", 2)
        for layer in range(2):
            for index, projection in enumerate(pack.PROJECTIONS):
                if missing and layer == 1 and projection == "down":
                    continue
                shape = (4, 7, 5) if projection == "down" else (4, 5, 7)
                dtype = pack.np.uint16 if bf16 else pack.np.float32
                data = (pack.np.arange(140) + 1000 * layer + 100 * index).astype(dtype).reshape(shape)
                writer.add_tensor(f"blk.{layer}.ffn_{projection}_exps.weight", data,
                                  raw_dtype=GGMLQuantizationType.BF16 if bf16 else None)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def test_roundtrip_and_guard_bytes(self):
        for bf16 in (False, True):
            with self.subTest(bf16=bf16):
                self.fixture(bf16=bf16)
                output = self.root / f"packed-{bf16}.gguf"
                pack.pack(self.model, output)
                pack.verify(self.model, output)

    def test_does_not_overwrite(self):
        self.fixture()
        self.output.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            pack.pack(self.model, self.output)
        self.assertEqual(self.output.read_bytes(), b"keep")

    def test_verification_offsets_cross_uint32_boundary(self):
        self.fixture()
        pack.pack(self.model, self.output)
        source_reader = pack.GGUFReader(self.model)
        packed_reader = pack.GGUFReader(self.output)
        base = 2**32 - pack.ALIGNMENT
        tensors = []
        offset = base
        for tensor in packed_reader.tensors:
            tensors.append(SimpleNamespace(name=tensor.name, data_offset=offset,
                                           n_bytes=tensor.n_bytes, data=tensor.data))
            offset += tensor.n_bytes
        # Sparse file and synthetic offsets exercise large-file arithmetic with small records.
        with self.output.open("r+b") as stream:
            stream.truncate(offset)
        reader = SimpleNamespace(fields=packed_reader.fields, tensors=tensors,
                                 data_offset=pack.np.uint32(base))
        with patch.object(pack, "GGUFReader", side_effect=[source_reader, reader]):
            pack.verify(self.model, self.output)

    def test_incomplete_layer_rejected(self):
        self.fixture(missing=True)
        with self.assertRaisesRegex(ValueError, "expected gate"):
            pack.pack(self.model, self.output)
        self.assertFalse(self.output.exists())

    def test_stale_source_rejected(self):
        self.fixture()
        pack.pack(self.model, self.output)
        stamp = self.model.stat()
        os.utime(self.model, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1))
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            pack.verify(self.model, self.output)

    def test_corrupt_payload_rejected(self):
        self.fixture()
        pack.pack(self.model, self.output)
        reader = pack.GGUFReader(self.output)
        offset = reader.tensors[0].data_offset
        with self.output.open("r+b") as stream:
            stream.seek(offset)
            value = stream.read(1)
            stream.seek(offset)
            stream.write(bytes([value[0] ^ 1]))
        with self.assertRaisesRegex(ValueError, "record mismatch"):
            pack.verify(self.model, self.output)


if __name__ == "__main__":
    unittest.main()
