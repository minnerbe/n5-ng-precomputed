import sys, os, json
import numpy as np
import tensorstore as ts

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

# formulas MUST match the Java side (PrecomputedTensorstoreTest)
def f_raw(x, y, z, c):
    return (x + 7 * y + 53 * z + 1009 * c) % 65521

def f_seg(x, y, z, c):
    return ((x // 4) * 13 + (y // 4) * 7 + (z // 2) * 3 + c) % 50

def f_seg64(x, y, z, c):
    return (((x // 4) + (y // 4) + (z // 2)) % 7) + (1 << 40)

def write(name, dtype, encoding, size, chunk, nc, fn, voltype="image",
          cseg_block=None, sharding=None):
    path = os.path.join(OUT, name)
    scale = {"encoding": encoding, "resolution": [1, 1, 1],
             "size": size, "chunk_size": chunk, "key": "1_1_1"}
    if cseg_block is not None:
        scale["compressed_segmentation_block_size"] = cseg_block
    if sharding is not None:
        scale["sharding"] = sharding
    spec = {
        "driver": "neuroglancer_precomputed",
        "kvstore": {"driver": "file", "path": path},
        "multiscale_metadata": {"data_type": dtype, "num_channels": nc, "type": voltype},
        "scale_metadata": scale,
    }
    store = ts.open(spec, create=True, open=True).result()
    sx, sy, sz = size
    npdt = {"uint8": np.uint8, "uint16": np.uint16, "uint32": np.uint32,
            "uint64": np.uint64}[dtype]
    data = np.fromfunction(fn, (sx, sy, sz, nc), dtype=np.int64).astype(npdt)
    store[...] = data
    print("wrote", name)

manifest = []
def add(name, dtype, encoding, size, chunk, nc, formula, **kw):
    write(name, dtype, encoding, size, chunk, nc, {"f_raw": f_raw, "f_seg": f_seg, "f_seg64": f_seg64}[formula],
          voltype=("segmentation" if encoding == "compressed_segmentation" else "image"), **kw)
    manifest.append({"name": name, "dtype": dtype, "encoding": encoding, "size": size,
                     "chunk": chunk, "num_channels": nc, "formula": formula,
                     "cseg_block": kw.get("cseg_block"), "sharded": "sharding" in kw})

add("raw_u16_c1", "uint16", "raw", [10, 8, 6], [4, 4, 4], 1, "f_raw")
add("raw_u8_c1", "uint8", "raw", [10, 8, 6], [4, 4, 4], 1, "f_raw")
add("cseg_u32", "uint32", "compressed_segmentation", [10, 8, 6], [8, 8, 8], 1, "f_seg", cseg_block=[4, 4, 4])
add("cseg_u64", "uint64", "compressed_segmentation", [9, 7, 5], [8, 8, 8], 1, "f_seg64", cseg_block=[4, 4, 4])

try:
    add("sharded_u16", "uint16", "raw", [16, 16, 16], [8, 8, 8], 1, "f_raw",
        sharding={"@type": "neuroglancer_uint64_sharded_v1", "preshift_bits": 0,
                  "hash": "identity", "minishard_bits": 2, "shard_bits": 0,
                  "minishard_index_encoding": "raw", "data_encoding": "gzip"})
except Exception as e:
    print("sharded generation failed:", e)

with open(os.path.join(OUT, "manifest.json"), "w") as fh:
    json.dump(manifest, fh)
print("manifest:", [m["name"] for m in manifest])
