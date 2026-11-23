"""发布包离线校验：独立进程加载 safetensors 包并做一次贪心生成冒烟。
通过标准：模型 strict 加载成功、生成非空文本、无异常。
用法：python validate_package.py --package /data/home/minimind-system/versions/<版本> --repo /data/home/minimind
"""
import argparse
import json
import os
import sys

import torch

torch.set_num_threads(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--repo", required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.repo)
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    from transformers import AutoTokenizer
    from safetensors.torch import load_file

    with open(os.path.join(args.package, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    config = MiniMindConfig(
        hidden_size=cfg.get("hidden_size", 768),
        num_hidden_layers=cfg.get("num_hidden_layers", 8),
        use_moe=False,
        vocab_size=cfg.get("vocab_size", 6400),
        num_attention_heads=cfg.get("num_attention_heads", 8),
        num_key_value_heads=cfg.get("num_key_value_heads", 4),
    )
    model = MiniMindForCausalLM(config)
    model.load_state_dict(load_file(os.path.join(args.package, "model.safetensors")), strict=True)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"params: {n/1e6:.2f} M")
    tokenizer = AutoTokenizer.from_pretrained(args.package)
    prompt = "中国的首都是"
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    with torch.inference_mode():
        out = model.generate(ids, max_new_tokens=24, do_sample=False, repetition_penalty=1.1,
                             eos_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][ids.shape[1]:].tolist(), skip_special_tokens=True)
    print("sample:", prompt + text)
    if not text.strip():
        print("FAIL: empty generation")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
