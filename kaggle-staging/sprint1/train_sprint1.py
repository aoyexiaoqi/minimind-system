#!/usr/bin/env python3
"""Sprint 1 training loop: run01 -> 1200 steps (trunk 960 + anneal 240), T4, bf16."""
import os, sys, json, math, time, random, argparse, shutil
import numpy as np
import torch
sys.path.insert(0, '/content/work/sprint1')
from sprint1_common import *

def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')

def save_resume(model, optimizer, step, mix_rng, cursor, loss_hist, phase, val_hist=None):
    os.makedirs(LOCAL_RESUME, exist_ok=True)
    state = {'step': step, 'phase': phase, 'spec': SPEC, 'model': model.state_dict(),
             'optimizer': optimizer.state_dict(), 'mix_rng': mix_rng.getstate(), 'cursor': cursor,
             'loss_history': loss_hist, 'val_history': val_hist or [],
             'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                     'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()}}
    atomic_torch_save(state, f'{LOCAL_RESUME}/step_{step}.pt')
    if step == TRUNK_STEPS:
        atomic_torch_save(state, f'{LOCAL_RESUME}/anneal_fork_step_{step}.pt')

def eval_anchor(model, step, val_texts, val_hist):
    """Compute the fixed-set validation loss at a checkpoint step and append
    it to val_hist (stored inside every resume checkpoint from now on).
    Returns the val loss, or None when no val set is available."""
    if not val_texts:
        return None
    model.eval()
    with torch.no_grad():
        vl, n = compute_val_loss(model, val_texts)
    model.train()
    val_hist.append({'step': step, 'val_loss': round(vl, 4), 'val_items': n})
    return vl

def find_resume():
    for d in [LOCAL_RESUME, f'{RUN_DIR}/resume']:
        if os.path.isdir(d):
            pts = [f for f in os.listdir(d) if f.startswith('step_') and f.endswith('.pt')]
            if pts:
                pts.sort(key=lambda f: int(f[5:-3]))
                src = os.path.join(d, pts[-1])
                if d != LOCAL_RESUME:
                    os.makedirs(LOCAL_RESUME, exist_ok=True)
                    shutil.copyfile(src, os.path.join(LOCAL_RESUME, pts[-1]))
                    src = os.path.join(LOCAL_RESUME, pts[-1])
                return src
    return None

def make_batch(data, anchors, mix_rng, cursor):
    avail_replay = min(N_REPLAY, cursor)
    n_new = N_NEW + (N_REPLAY - avail_replay)
    new_idx = list(range(cursor, cursor + n_new))
    new_cursor = cursor + n_new
    replay_idx = mix_rng.choices(range(cursor), k=avail_replay) if avail_replay > 0 else []
    anchor_idx = mix_rng.choices(anchors, k=N_ANCHOR)
    idx = new_idx + replay_idx + anchor_idx
    assert len(idx) == SEQ_PER_STEP, f'batch size {len(idx)} != {SEQ_PER_STEP}'
    mix_rng.shuffle(idx)
    return idx, new_cursor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['fresh', 'resume'], default='fresh')
    ap.add_argument('--max-steps', type=int, default=TOTAL_STEPS)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    data = load_blocks(); anchors = load_anchors()
    log(f'data blocks {data.shape}, anchors {len(anchors)}, spec {json.dumps(SPEC, ensure_ascii=False)}')
    val_texts = load_val_texts(); val_hist = []
    log(f'val_fixed: {len(val_texts) if val_texts else 0} items (anchor val loss {"ON" if val_texts else "OFF - val file missing/empty"})')
    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX)
    mix_rng = random.Random(SEED)
    cursor = 0; start_step = 0; loss_hist = []
    if args.mode == 'resume':
        path = find_resume()
        assert path, 'no resume package found'
        st = torch.load(path, map_location='cuda', weights_only=False)
        model.load_state_dict(st['model'], strict=True)
        optimizer.load_state_dict(st['optimizer'])
        mix_rng.setstate(st['mix_rng']); cursor = st['cursor']; start_step = st['step']; loss_hist = st['loss_history']
        val_hist = st.get('val_history', [])  # pre-patch checkpoints carry no val_history
        random.setstate(st['rng']['python']); np.random.set_state(st['rng']['numpy'])
        torch.set_rng_state(st['rng']['torch']); torch.cuda.set_rng_state(st['rng']['cuda'])
        log(f'RESUMED from {path} at step {start_step}, cursor {cursor}')
    model.train()
    max_steps = 5 if args.smoke else min(args.max_steps, TOTAL_STEPS)
    step_times = []; t_start = time.time()
    for step in range(start_step + 1, max_steps + 1):
        lr = lr_at(step)
        for g in optimizer.param_groups:
            g['lr'] = lr
        idx, cursor = make_batch(data, anchors, mix_rng, cursor)
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        tot_loss = 0.0
        for m in range(ACCUM):
            mb = idx[m * MICRO_BATCH:(m + 1) * MICRO_BATCH]
            x = torch.tensor(data[mb], dtype=torch.long, device='cuda')
            out = model(x, labels=x)
            (out.loss / ACCUM).backward()
            tot_loss += out.loss.item() / ACCUM
        if not math.isfinite(tot_loss):
            log(f'LOSS NOT FINITE at step {step}: {tot_loss} - STOP (last checkpoint remains for resume)')
            raise SystemExit(3)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        dt = time.time() - t0
        step_times.append(dt)
        loss_hist.append({'step': step, 'loss': round(tot_loss, 4), 'lr': lr})
        if step % 10 == 0 or step <= 3:
            log(f"step {step}/{TOTAL_STEPS} loss {tot_loss:.4f} lr {lr:.2e} grad_norm {float(gn):.2f} {dt:.2f}s/step tok/s {TOKENS_PER_STEP/dt:.0f}")
        if step == 50:
            sel = step_times[5:50]
            mean_s = sum(sel) / len(sel)
            rec = {'calib_steps': '6-50', 'mean_sec_per_step': round(mean_s, 3), 'tok_per_sec': round(TOKENS_PER_STEP / mean_s, 1), 'est_total_min_1200': round(mean_s * TOTAL_STEPS / 60, 1), 'est_trunk_min_960': round(mean_s * TRUNK_STEPS / 60, 1)}
            json.dump(rec, open(f'{ROOTW}/throughput.json', 'w'), indent=2)
            shutil.copyfile(f'{ROOTW}/throughput.json', f'{RUN_DIR}/throughput.json')
            log(f'THROUGHPUT CALIB {rec}')
        if not args.smoke and step % CKPT_EVERY == 0:
            phase = 'trunk' if step <= TRUNK_STEPS else 'anneal'
            vl = eval_anchor(model, step, val_texts, val_hist)
            sha = save_package(model, step)
            save_resume(model, optimizer, step, mix_rng, cursor, loss_hist, phase, val_hist)
            sync_s = sync_to_drive(step)
            json.dump(loss_hist, open(f'{ROOTW}/loss_history.json', 'w'))
            json.dump(val_hist, open(f'{ROOTW}/val_history.json', 'w'))
            log(f'CHECKPOINT step {step} ({phase}) sha256 {sha} drive_sync {sync_s}s cursor {cursor}' + (f' val_loss {vl:.4f}' if vl is not None else ''))
    if not args.smoke and max_steps > start_step and max_steps % CKPT_EVERY != 0:
        # Final-step forced save: a run ending between checkpoint steps must
        # still hand over a resume point at its true last step (the Modal
        # baton that ended at step 960 previously fell back to step 900).
        phase = 'trunk' if max_steps <= TRUNK_STEPS else 'anneal'
        vl = eval_anchor(model, max_steps, val_texts, val_hist)
        sha = save_package(model, max_steps)
        save_resume(model, optimizer, max_steps, mix_rng, cursor, loss_hist, phase, val_hist)
        sync_s = sync_to_drive(max_steps)
        json.dump(val_hist, open(f'{ROOTW}/val_history.json', 'w'))
        log(f'FINAL CHECKPOINT step {max_steps} ({phase}) sha256 {sha} drive_sync {sync_s}s cursor {cursor}' + (f' val_loss {vl:.4f}' if vl is not None else ''))
    json.dump(loss_hist, open(f'{ROOTW}/loss_history.json', 'w'))
    log(f'DONE steps {start_step+1}..{max_steps} elapsed {round((time.time()-t_start)/60,1)} min')
    if not args.smoke:
        shutil.copyfile(LOG_PATH, f'{RUN_DIR}/train.log')
        shutil.copyfile(f'{ROOTW}/loss_history.json', f'{RUN_DIR}/loss_history.json')
        if os.path.exists(f'{ROOTW}/val_history.json'):
            shutil.copyfile(f'{ROOTW}/val_history.json', f'{RUN_DIR}/val_history.json')

if __name__ == '__main__':
    main()