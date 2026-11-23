"""把 minimind 训练检查点（.pth state_dict）导出为 safetensors 发布包。

用法（在 InstaBox 上）：
  python export_to_safetensors.py \
    --checkpoint /data/home/minimind/<save_dir>/pretrain_768.pth \
    --repo /data/home/minimind \
    --out /data/home/minimind-system/versions/20261123-pretrain-run01 \
    --version 20261123-pretrain-run01 --step 600 --loss 5.1380
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

import torch
from safetensors.torch import save_file


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--repo", required=True, help="minimind 仓库路径（取 model/ 下 tokenizer 文件）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--step", type=int, default=600)
    ap.add_argument("--loss", type=float, default=5.1380)
    ap.add_argument("--hidden", type=int, default=768)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--code-head", default="")
    args = ap.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"FAIL: checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    raw = torch.load(args.checkpoint, map_location="cpu")
    state_dict = raw.get("model", raw) if isinstance(raw, dict) else raw
    if not isinstance(state_dict, dict):
        print("FAIL: checkpoint is not a state_dict")
        sys.exit(1)
    n_params = sum(v.numel() for v in state_dict.values() if hasattr(v, "numel"))
    print(f"checkpoint tensors: {len(state_dict)}, params: {n_params/1e6:.2f} M")
    if not (60e6 < n_params < 68e6):
        print("FAIL: param count not in expected 64M range (60-68M)")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    st_path = os.path.join(args.out, "model.safetensors")
    save_file(state_dict, st_path, metadata={"format": "pt"})

    config = {
        "hidden_size": args.hidden,
        "num_hidden_layers": args.layers,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "vocab_size": 6400,
        "use_moe": False,
        "dropout": 0.0,
    }
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    for name in ("tokenizer.json", "tokenizer_config.json"):
        src = os.path.join(args.repo, "model", name)
        if not os.path.exists(src):
            print(f"FAIL: tokenizer file missing: {src}")
            sys.exit(1)
        shutil.copyfile(src, os.path.join(args.out, name))

    manifest = {
        "version": args.version,
        "source_checkpoint": args.checkpoint,
        "train_step": args.step,
        "train_loss_final": args.loss,
        "param_count": n_params,
        "safetensors_sha256": sha256_of(st_path),
        "code_head": args.code_head,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "format": "safetensors",
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("OK package written:", args.out)
    print("safetensors sha256:", manifest["safetensors_sha256"])


if __name__ == "__main__":
    main()
