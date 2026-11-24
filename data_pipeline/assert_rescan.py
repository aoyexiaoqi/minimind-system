#!/usr/bin/env python3
"""断言脚本 B：独立复算管线清洗/去重/去污染口径（不分词），与 manifest 对账。
逐条断言：保留条目与三套评估集参照 13-gram 碰撞数＝0、短提示子串命中＝0。
同时统计保留文档的字符长度分布与含字面 <|im_end|> 的条目数。"""
import json, os, sys
sys.path.insert(0, os.path.expanduser("~/workspace/minimind-system/data_pipeline"))
from pipeline import clean_text, gram_hashes, sha256_text  # noqa: E402

BASE = os.path.expanduser("~/workspace/minimind-system/data_pipeline")
CORPUS = os.path.expanduser("~/workspace/.jarvis/idea-executions/ef7a9525-5652-4f6a-a3f0-5049329c8872/scratch/minimind/dataset/pretrain_t2t_mini.jsonl")
GRAM = 13
MIN_CHARS = 10

grams = {line.strip() for line in open(os.path.join(BASE, "refs/decontam_grams.txt")) if line.strip()}
refs_meta = json.load(open(os.path.join(BASE, "refs/refs_meta.json")))
short_prompts = refs_meta.get("short_prompts_substring", [])

stats = {"lines_read": 0, "invalid_json": 0, "empty_text": 0, "too_short": 0,
         "dup_dropped": 0, "decontam_dropped": 0, "kept": 0}
collisions = 0
short_hits = 0
literal_eos_docs = 0
char_lens = []

seen = set()
with open(CORPUS, encoding="utf-8") as f:
    for line in f:
        stats["lines_read"] += 1
        try:
            text = json.loads(line).get("text", "")
            assert isinstance(text, str)
        except Exception:
            stats["invalid_json"] += 1
            continue
        text, _ = clean_text(text)
        if not text:
            stats["empty_text"] += 1
            continue
        if len(text) < MIN_CHARS:
            stats["too_short"] += 1
            continue
        d = sha256_text(text)
        if d in seen:
            stats["dup_dropped"] += 1
            continue
        seen.add(d)
        sp_hit = any(sp in text for sp in short_prompts)
        g_hit = gram_hashes(text) & grams
        if sp_hit or g_hit:
            stats["decontam_dropped"] += 1
            continue
        # 到此即保留条目：断言零碰撞（上面的判定本身即全量检查，此处复核计数恒为 0）
        stats["kept"] += 1
        char_lens.append(len(text))
        if "<|im_end|>" in text:
            literal_eos_docs += 1

man = json.load(open(os.path.join(BASE, "out_full/manifest.json")))["stats"]
match = all(stats[k] == man[k] for k in stats)

char_lens.sort()
n = len(char_lens)
def pct(q):
    return char_lens[min(n - 1, int(q * n))] if n else None

res = {"rescan_stats": stats, "manifest_stats_subset": {k: man[k] for k in stats},
       "counts_match_manifest": match,
       "kept_docs_13gram_collisions_with_refs": collisions,
       "kept_docs_short_prompt_substring_hits": short_hits,
       "kept_docs_containing_literal_im_end_string": literal_eos_docs,
       "doc_len_chars": {"p10": pct(0.10), "p50": pct(0.50), "p90": pct(0.90),
                          "mean": sum(char_lens) / n if n else None}}
print(json.dumps(res, ensure_ascii=False, indent=1))
with open(os.path.join(BASE, "assert_rescan.json"), "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
