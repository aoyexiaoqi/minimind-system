#!/usr/bin/env python3
"""断言脚本 A：全量扫描 out_full 二进制 token 流（numpy，系统 python3）。
检查：非法 token id、id0/id2 计数、分隔符与保留条数对账、文档长度（token）分布。"""
import json, os, sys
import numpy as np

OUT = os.path.expanduser("~/workspace/minimind-system/data_pipeline/out_full")
VOCAB = 6400
EOS = 2

man = json.load(open(os.path.join(OUT, "manifest.json")))
kept = man["stats"]["kept"]
kept_tokens = man["stats"]["kept_tokens"]
tail_pad = man["tail_pad_tokens"]

parts = []
for c in man["chunks"]:
    parts.append(np.fromfile(os.path.join(OUT, c["file"]), dtype=np.uint16))
arr = np.concatenate(parts)
total_stream = int(arr.size)
assert total_stream == man["total_blocks"] * 512, (total_stream, man["total_blocks"])
stream = arr[:kept_tokens]
pad_region = arr[kept_tokens:]
assert int((pad_region == 0).sum()) == tail_pad == int(pad_region.size), "tail pad 区域应全为 id 0"

max_id = int(stream.max())
illegal = int((stream >= VOCAB).sum())
cnt0 = int((stream == 0).sum())
cnt2 = int((stream == EOS).sum())

pos = np.nonzero(stream == EOS)[0]
if pos.size:
    bodies = np.empty(pos.size, dtype=np.int64)
    bodies[0] = pos[0]
    bodies[1:] = np.diff(pos) - 1
    trailing = kept_tokens - 1 - int(pos[-1])
else:
    bodies = np.array([], dtype=np.int64)
    trailing = kept_tokens

def pct(a, q):
    return float(np.percentile(a, q)) if a.size else None

res = {
    "total_stream_tokens_incl_pad": total_stream,
    "real_tokens": kept_tokens,
    "tail_pad_tokens": tail_pad,
    "tail_pad_all_zero": True,
    "vocab_size": VOCAB,
    "max_token_id": max_id,
    "illegal_token_ids_ge_vocab": illegal,
    "count_id0_in_real_stream": cnt0,
    "count_eos_id2_in_real_stream": cnt2,
    "kept_docs_manifest": kept,
    "eos_minus_kept": cnt2 - kept,
    "trailing_tokens_after_last_eos": trailing,
    "doc_body_len_tokens": {
        "n_segments": int(bodies.size),
        "p10": pct(bodies, 10), "p50": pct(bodies, 50), "p90": pct(bodies, 90),
        "mean": float(bodies.mean()) if bodies.size else None,
        "min": int(bodies.min()) if bodies.size else None,
        "max": int(bodies.max()) if bodies.size else None,
        "zero_len_segments": int((bodies == 0).sum()),
    },
    "eos_per_million_tokens_appended_sep": kept / kept_tokens * 1e6,
    "eos_per_million_tokens_all_id2": cnt2 / kept_tokens * 1e6,
}
print(json.dumps(res, ensure_ascii=False, indent=1))
with open(os.path.expanduser("~/workspace/minimind-system/data_pipeline/assert_binary.json"), "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
