"""Sprint 1 shared helpers: spec, data, model, checkpoint/resume I/O.
Spec: run01 -> 1200 steps (trunk 960: warmup 120 to 3e-4 then constant; anneal 240: cosine 3e-4 -> 3e-5),
effective batch 65536 tokens/step (128 x 512), micro-batch 16 x 8 accum, grad clip 1.0, AdamW upstream defaults,
mixture per step 90 new (sequential) / 26 replay (consumed blocks, random) / 12 anchors (fixed pool), bf16."""
import os, sys, json, math, time, random, hashlib, shutil
import numpy as np
import torch
from safetensors.torch import load_file, save_file

ROOTW = '/content/work/sprint1'
REPO = f'{ROOTW}/minimind'
sys.path.insert(0, REPO)
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

BASE_PKG = f'{ROOTW}/base_pkg'
DATA_DIR = f'{ROOTW}/data/out_full'
ANCHOR_PATH = f'{ROOTW}/data/anchor_pool.json'
RUN_DIR = '/content/drive/MyDrive/minimind-system/runs/sprint1-20261124'
LOCAL_CKPT = f'{ROOTW}/checkpoints'
LOCAL_RESUME = f'{ROOTW}/resume'
LOG_PATH = f'{ROOTW}/train.log'

TOTAL_STEPS = 1200
TRUNK_STEPS = 960
WARMUP = 120
LR_MAX = 3e-4
LR_MIN_ANNEAL = 3e-5
TOKENS_PER_STEP = 65536
SEQ_LEN = 512
SEQ_PER_STEP = 128
MICRO_BATCH = 16
ACCUM = 8
N_NEW, N_REPLAY, N_ANCHOR = 90, 26, 12
GRAD_CLIP = 1.0
SEED = 20261124
BASE_SHA256 = '2efb5c8d3b048bacfa42c6ff0ed021d9a3f3bf4efc2671dd86c7e5faf3cf3ed9'

SPEC = dict(total_steps=TOTAL_STEPS, trunk_steps=TRUNK_STEPS, warmup=WARMUP, lr_max=LR_MAX,
            lr_min_anneal=LR_MIN_ANNEAL, tokens_per_step=TOKENS_PER_STEP, seq_len=SEQ_LEN,
            seq_per_step=SEQ_PER_STEP, micro_batch=MICRO_BATCH, accum=ACCUM, n_new=N_NEW,
            n_replay=N_REPLAY, n_anchor=N_ANCHOR, grad_clip=GRAD_CLIP, seed=SEED,
            base_version='20261123-pretrain-run01', base_sha256=BASE_SHA256,
            code_head='3d85c71e5cd86b8622107eb1f5745bdb4cc5590a', precision='bf16',
            optimizer='AdamW(lr=3e-4, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01, PyTorch/upstream defaults)',
            anchor_rule='blocks with >=4 <|im_end|> (id 2), first 2048 in chunk order, data/anchor_pool.json',
            clean_baseline='no EOS weighting, no unlikelihood, no decoding change')

def lr_at(step):
    if step <= WARMUP:
        return LR_MAX * step / WARMUP
    if step <= TRUNK_STEPS:
        return LR_MAX
    a = step - TRUNK_STEPS
    return LR_MIN_ANNEAL + 0.5 * (LR_MAX - LR_MIN_ANNEAL) * (1 + math.cos(math.pi * a / (TOTAL_STEPS - TRUNK_STEPS)))

def load_blocks():
    man = json.load(open(f'{DATA_DIR}/manifest.json'))
    arrs = [np.fromfile(f"{DATA_DIR}/{ch['file']}", dtype=np.uint16) for ch in man['chunks']]
    return np.concatenate(arrs).reshape(-1, SEQ_LEN)

def load_anchors():
    return json.load(open(ANCHOR_PATH))['pool']

def build_model():
    cfg = json.load(open(f'{BASE_PKG}/config.json'))
    config = MiniMindConfig(hidden_size=cfg['hidden_size'], num_hidden_layers=cfg['num_hidden_layers'],
                            use_moe=False, vocab_size=cfg.get('vocab_size', 6400),
                            num_attention_heads=cfg['num_attention_heads'],
                            num_key_value_heads=cfg['num_key_value_heads'])
    model = MiniMindForCausalLM(config)
    model.load_state_dict(load_file(f'{BASE_PKG}/model.safetensors'), strict=True)
    return model.to(torch.bfloat16).cuda()

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()

def save_package(model, step):
    pkg = f'{LOCAL_CKPT}/step_{step}'
    os.makedirs(pkg, exist_ok=True)
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    save_file(sd, f'{pkg}/model.safetensors', metadata={'format': 'pt'})
    for name in ['config.json', 'tokenizer.json', 'tokenizer_config.json']:
        shutil.copyfile(f'{BASE_PKG}/{name}', f'{pkg}/{name}')
    return sha256_file(f'{pkg}/model.safetensors')

VAL_PATH = f'{ROOTW}/data/val_fixed.json'
VAL_MAX_SEQ_LEN = 340   # V2.1 formal-eval parity (v21_eval_s1.py: MAX_SEQ_LEN=340)
VAL_BATCH_SIZE = 16     # V2.1 formal-eval parity (BATCH_SIZE=16, per-sequence weighted)
CKPT_EVERY = 100        # checkpoint interval in steps (was a literal 100 in the loop; default unchanged)

_val_tokenizer = None

def load_val_texts():
    """Fixed validation texts: the same val_fixed.json the formal eval uses
    (512 JSONL records of {"text": ...}). Returns None when the file is
    missing or empty -- training then proceeds without val-loss recording."""
    if not os.path.isfile(VAL_PATH) or os.path.getsize(VAL_PATH) == 0:
        return None
    texts = []
    with open(VAL_PATH, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(str(json.loads(line)['text']))
    return texts or None

def _get_val_tokenizer():
    global _val_tokenizer
    if _val_tokenizer is None:
        from transformers import AutoTokenizer
        _val_tokenizer = AutoTokenizer.from_pretrained(BASE_PKG)
    return _val_tokenizer

def compute_val_loss(model, val_texts):
    """Anchor-step validation loss, byte-for-byte the V2.1 formal eval's
    eval_val_loss semantics: PretrainDataset encoding (bos + text truncated
    to 338 + eos, padded to 340, pad positions labelled -100), sequential
    batches of 16, sum(batch_mean_loss * batch_size) / total_items.
    Deterministic (no shuffling, no RNG consumption), so recording it never
    perturbs the training trajectory. Caller wraps in eval()/no_grad and
    restores train() mode. Returns (val_loss, n_items)."""
    tok = _get_val_tokenizer()
    bos, eos, pad = tok.bos_token_id, tok.eos_token_id, tok.pad_token_id
    seqs = []
    for text in val_texts:
        ids = tok(str(text), add_special_tokens=False,
                  max_length=VAL_MAX_SEQ_LEN - 2, truncation=True).input_ids
        ids = [bos] + ids + [eos]
        ids = ids + [pad] * (VAL_MAX_SEQ_LEN - len(ids))
        seqs.append(ids)
    device = next(model.parameters()).device
    total, count = 0.0, 0
    for i in range(0, len(seqs), VAL_BATCH_SIZE):
        batch = torch.tensor(seqs[i:i + VAL_BATCH_SIZE], dtype=torch.long, device=device)
        labels = batch.clone()
        labels[batch == pad] = -100
        out = model(batch, labels=labels)
        total += out.loss.item() * batch.size(0)
        count += batch.size(0)
    return total / max(count, 1), count

def atomic_torch_save(state, path):
    """Write to a temp file, then os.replace: a half-written checkpoint is
    never visible as a usable resume point."""
    tmp = f'{path}.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)

def sync_to_drive(step):
    t0 = time.time()
    dst_ckpt = f'{RUN_DIR}/checkpoints/step_{step}'
    os.makedirs(dst_ckpt, exist_ok=True)
    for name in ['model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json']:
        shutil.copyfile(f'{LOCAL_CKPT}/step_{step}/{name}', f'{dst_ckpt}/{name}')
    dst_res = f'{RUN_DIR}/resume'
    os.makedirs(dst_res, exist_ok=True)
    src = f'{LOCAL_RESUME}/step_{step}.pt'
    if os.path.exists(src):
        shutil.copyfile(src, f'{dst_res}/step_{step}.pt')
    if step == TRUNK_STEPS:
        fork = f'{LOCAL_RESUME}/anneal_fork_step_{step}.pt'
        if os.path.exists(fork):
            shutil.copyfile(fork, f'{dst_res}/anneal_fork_step_{step}.pt')
    return round(time.time() - t0, 1)