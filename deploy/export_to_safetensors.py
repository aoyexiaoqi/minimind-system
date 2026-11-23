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
    tensors = {k: v for k, v in state_dict.items() if hasattr(v, "numel")}
    raw_params = sum(v.numel() for v in tensors.values())
    # 权重绑定（embed_tokens 与 lm_head 共享同一张量）时 state_dict 里会有
    # 两个键指向同一块存储，按原始求和会重复计一遍词表矩阵（6400x768≈4.92M）。
    # 参数量校验按唯一存储计数；同时打印共享存储的键对，便于人工核对。
    seen = {}
    unique_params = 0
    shared_pairs = []
    for k, v in tensors.items():
        sig = (v.data_ptr(), tuple(v.shape), str(v.dtype))
        if sig in seen and v.numel() > 0:
            shared_pairs.append((seen[sig], k))
        else:
            seen[sig] = k
            unique_params += v.numel()
    print(f"checkpoint tensors: {len(state_dict)}, raw params: {raw_params/1e6:.2f} M, "
          f"unique params: {unique_params/1e6:.2f} M")
    for a, b in shared_pairs:
        print(f"shared storage: {a} == {b}")
    n_params = unique_params
    if not (60e6 < n_params < 68e6):
        print("FAIL: unique param count not in expected 64M range (60-68M)")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    st_path = os.path.join(args.out, "model.safetensors")
    # safetensors 不接受多个键共享同一存储：共享的张量克隆一份再存（值完全相同，
    # 加载端按模型类自身的绑定关系使用，不受影响）。
    save_dict = {}
    seen_ptr = set()
    for k, v in tensors.items():
        if v.numel() > 0 and v.data_ptr() in seen_ptr:
            save_dict[k] = v.clone()
        else:
            seen_ptr.add(v.data_ptr())
            save_dict[k] = v
    save_file(save_dict, st_path, metadata={"format": "pt"})

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
        "param_count_raw_state_dict": raw_params,
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
