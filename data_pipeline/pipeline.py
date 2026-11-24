#!/usr/bin/env python3
"""minimind-system 数据治理管线（设计定版 §一 数据策略 / §五 搭建顺序第 1 步）

功能：
  build-refs  从训练语料按 run03 笔记本 cell 6 的完全相同算法（蓄水池抽样 seed=42、
              reservoir=120000）复现冻结验证集 res[11200:11712]，并连同双提示集
              构建 13-gram 去污染参照集（字符级 13-gram 的 sha1 集合）。
  process     流式处理语料：规则清洗 → sha256 去重 → 13-gram 去污染 → 预分词 →
              512 token 打包（文档间以分隔符 token 连接）→ uint16 二进制分块 + 清单。
  verify      回读分块 .bin，校验尺寸/块数并解码抽样块，确认产物可读。

分隔符说明：定版/任务书写「<|endoftext|> 分隔」，但本分词器中 <|endoftext|> 是
id 0（pad/unk），模型真正的文档结束/停止符是 eos <|im_end|>（id 2，上游
PretrainDataset 以 [bos]+text+[eos] 训练）。故默认分隔符取 eos（id 2）；
可用 --separator endoftext 改为 id 0。选择会写进 manifest。

运行环境：~/workspace/.venvs/minimind-data（仅需 tokenizers）。
"""
import argparse, hashlib, json, os, random, re, sys, time

GRAM = 13
VAL_SLICE = (11200, 11712)
RESERVOIR_SIZE = 120000
SEED = 42
BLOCK = 512

CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def gram_hashes(text: str):
    """字符 13-gram 的 sha1 集合（文本短于 13 字时返回空集，另行整串处理）。"""
    if len(text) < GRAM:
        return set()
    return {hashlib.sha1(text[i:i + GRAM].encode("utf-8")).hexdigest()
            for i in range(len(text) - GRAM + 1)}


# ---------------------------------------------------------------- build-refs
def build_refs(corpus: str, prompts_main: str, prompts_shadow: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    # 1) 蓄水池抽样：与 run03 cell 6 逐行同算法（random.Random(42).randint）
    rng = random.Random(SEED)
    res = []
    with open(corpus, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < RESERVOIR_SIZE:
                res.append(line)
            else:
                j = rng.randint(0, i)
                if j < RESERVOIR_SIZE:
                    res[j] = line
    total_lines = i + 1
    assert len(res) == RESERVOIR_SIZE, len(res)
    # 2) 冻结验证集：res[11200:11712]，按 cell 6 clean_idx 口径（解析+非空+sha256 去重）
    seen, val_texts, bad = set(), [], 0
    for line in res[VAL_SLICE[0]:VAL_SLICE[1]]:
        try:
            text = json.loads(line).get("text", "")
            assert isinstance(text, str) and text.strip()
        except Exception:
            bad += 1
            continue
        d = sha256_text(text)
        if d in seen:
            continue
        seen.add(d)
        val_texts.append(text)
    assert len(val_texts) == 512, f"val items {len(val_texts)} != 512"
    with open(os.path.join(out_dir, "val_texts.jsonl"), "w", encoding="utf-8") as f:
        for t in val_texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    # 3) 去污染参照：验证集 13-gram + 双提示集 13-gram；短提示（<13 字）整串子串匹配
    grams = set()
    for t in val_texts:
        grams |= gram_hashes(t)
    short_prompts = []
    n_prompt_grams = 0
    for path in (prompts_main, prompts_shadow):
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            t = json.loads(line)["prompt"]
            if len(t) < GRAM:
                short_prompts.append(t)
            else:
                g = gram_hashes(t)
                n_prompt_grams += len(g)
                grams |= g
    with open(os.path.join(out_dir, "decontam_grams.txt"), "w") as f:
        for g in sorted(grams):
            f.write(g + "\n")
    meta = {"corpus": corpus, "corpus_lines": total_lines,
            "reservoir_size": RESERVOIR_SIZE, "seed": SEED,
            "val_slice": f"res[{VAL_SLICE[0]}:{VAL_SLICE[1]}]",
            "val_items": len(val_texts), "val_bad_lines": bad,
            "val_texts_sha256": sha256_text("\n".join(val_texts)),
            "decontam_grams": len(grams), "prompt_grams": n_prompt_grams,
            "short_prompts_substring": short_prompts,
            "elapsed_sec": round(time.time() - t0, 1)}
    with open(os.path.join(out_dir, "refs_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------- process
def clean_text(raw: str):
    """规则清洗：去控制字符（保留 \\n \\t）→ strip。返回 (cleaned, n_ctrl_removed)。"""
    cleaned, n = CTRL_RE.subn("", raw)
    return cleaned.strip(), n


def process(corpus: str, refs_dir: str, out_dir: str, tokenizer_path: str,
            limit: int | None, separator: str, blocks_per_chunk: int,
            min_chars: int):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)
    eos_id = tok.token_to_id("<|im_end|>")
    eot_id = tok.token_to_id("<|endoftext|>")
    sep_id = eos_id if separator == "eos" else eot_id
    assert sep_id is not None and eos_id is not None

    grams = set()
    with open(os.path.join(refs_dir, "decontam_grams.txt")) as f:
        grams = {line.strip() for line in f if line.strip()}
    refs_meta = json.load(open(os.path.join(refs_dir, "refs_meta.json")))
    short_prompts = refs_meta.get("short_prompts_substring", [])

    os.makedirs(out_dir, exist_ok=True)
    stats = {"lines_read": 0, "invalid_json": 0, "empty_text": 0,
             "ctrl_chars_removed": 0, "too_short": 0, "dup_dropped": 0,
             "decontam_dropped": 0, "kept": 0, "kept_tokens": 0}
    seen = set()
    buf = []            # 待打包 token 流
    chunks = []         # chunk 清单
    chunk_idx = 0
    cur = None          # 当前 chunk 的 array('H')
    from array import array
    t0 = time.time()

    def flush_block(block):
        nonlocal cur, chunk_idx
        if cur is None:
            cur = array("H")
        cur.extend(block)
        if len(cur) >= blocks_per_chunk * BLOCK:
            write_chunk()

    def write_chunk():
        nonlocal cur, chunk_idx
        if cur is None or len(cur) == 0:
            return
        name = f"chunk-{chunk_idx:04d}.bin"
        with open(os.path.join(out_dir, name), "wb") as f:
            cur.tofile(f)
        chunks.append({"file": name, "blocks": len(cur) // BLOCK,
                         "tokens": len(cur)})
        chunk_idx += 1
        cur = None

    def emit(ids):
        buf.extend(ids)
        while len(buf) >= BLOCK:
            flush_block(buf[:BLOCK])
            del buf[:BLOCK]

    with open(corpus, encoding="utf-8") as f:
        for line in f:
            if limit is not None and stats["lines_read"] >= limit:
                break
            stats["lines_read"] += 1
            try:
                text = json.loads(line).get("text", "")
                assert isinstance(text, str)
            except Exception:
                stats["invalid_json"] += 1
                continue
            text, n_ctrl = clean_text(text)
            stats["ctrl_chars_removed"] += n_ctrl
            if not text:
                stats["empty_text"] += 1
                continue
            if len(text) < min_chars:
                stats["too_short"] += 1
                continue
            d = sha256_text(text)
            if d in seen:
                stats["dup_dropped"] += 1
                continue
            seen.add(d)
            if any(sp in text for sp in short_prompts) or (gram_hashes(text) & grams):
                stats["decontam_dropped"] += 1
                continue
            ids = tok.encode(text, add_special_tokens=False).ids
            emit(ids + [sep_id])
            stats["kept"] += 1
            stats["kept_tokens"] += len(ids) + 1

    tail_pad = 0
    if buf:
        tail_pad = BLOCK - len(buf)
        flush_block(buf + [0] * tail_pad)   # 末块以 pad(id 0) 补齐，真实长度见 kept_tokens
        buf = []
    write_chunk()

    manifest = {"corpus": corpus, "limit": limit, "refs_dir": refs_dir,
                "tokenizer": tokenizer_path,
                "tokenizer_sha256": hashlib.sha256(open(tokenizer_path, "rb").read()).hexdigest(),
                "separator": {"choice": separator,
                              "token": "<|im_end|>" if separator == "eos" else "<|endoftext|>",
                              "id": sep_id,
                              "note": "默认 eos(<|im_end|>,id2)=模型训练停止符；<|endoftext|>(id0) 在本分词器中为 pad/unk"},
                "block_size": BLOCK, "dtype": "uint16",
                "stats": stats, "tail_pad_tokens": tail_pad,
                "chunks": chunks,
                "total_blocks": sum(c["blocks"] for c in chunks),
                "elapsed_sec": round(time.time() - t0, 1)}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------- verify
def verify(out_dir: str, tokenizer_path: str):
    from tokenizers import Tokenizer
    import struct
    tok = Tokenizer.from_file(tokenizer_path)
    man = json.load(open(os.path.join(out_dir, "manifest.json")))
    total_blocks = 0
    for c in man["chunks"]:
        path = os.path.join(out_dir, c["file"])
        size = os.path.getsize(path)
        assert size == c["blocks"] * BLOCK * 2, (path, size, c["blocks"])
        total_blocks += c["blocks"]
    assert total_blocks == man["total_blocks"]
    first = os.path.join(out_dir, man["chunks"][0]["file"])
    with open(first, "rb") as f:
        raw = f.read(BLOCK * 2 * 2)
    ids = list(struct.unpack(f"<{BLOCK * 2}H", raw))
    print(f"chunks={len(man['chunks'])} total_blocks={total_blocks} size_check=PASS")
    print("block0 decoded:", tok.decode(ids[:BLOCK], skip_special_tokens=False)[:200])
    print("block1 first tokens:", ids[BLOCK:BLOCK + 10])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-refs")
    b.add_argument("--corpus", required=True)
    b.add_argument("--prompts-main", required=True)
    b.add_argument("--prompts-shadow", required=True)
    b.add_argument("--out-dir", required=True)
    p = sub.add_parser("process")
    p.add_argument("--corpus", required=True)
    p.add_argument("--refs-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--separator", choices=["eos", "endoftext"], default="eos")
    p.add_argument("--blocks-per-chunk", type=int, default=4096)
    p.add_argument("--min-chars", type=int, default=10)
    v = sub.add_parser("verify")
    v.add_argument("--out-dir", required=True)
    v.add_argument("--tokenizer", required=True)
    a = ap.parse_args()
    if a.cmd == "build-refs":
        build_refs(a.corpus, a.prompts_main, a.prompts_shadow, a.out_dir)
    elif a.cmd == "process":
        process(a.corpus, a.refs_dir, a.out_dir, a.tokenizer, a.limit,
                a.separator, a.blocks_per_chunk, a.min_chars)
    else:
        verify(a.out_dir, a.tokenizer)
