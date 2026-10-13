"""
run_qwen_experiment.py — Combined Qwen cache + Algorithm 1 experiment

Generates Qwen LLM predictions ONLY for the samples needed by the experiment
(default 50K), saves a partial cache, then runs Algorithm 1.

Usage (on cluster):
  # tmux 1: small then large
  python -u run_qwen_experiment.py --llm_type small --max_samples 50000 2>&1 | tee run_v3_small.log
  python -u run_qwen_experiment.py --llm_type large --max_samples 50000 2>&1 | tee run_v3_large.log

  # tmux 2: base
  python -u run_qwen_experiment.py --llm_type base --max_samples 50000 2>&1 | tee run_v3_base.log

What it does:
  1. Load v3 dataset (mydataset.0.v3.pickle.gz)
  2. Determine the 50K sample indices the experiment will use (seed-based permutation)
  3. Generate Qwen predictions ONLY for those 50K samples (~4% of full dataset)
  4. Save partial cache as llm_cache_amazon_v3_{type}_partial{N}.pkl.gz
  5. Run Algorithm 1 experiment
  6. Save results as results_algo1_amazon_v3_{type}_c100.0_pfull.json
"""

import torch
import argparse
import numpy as np
import json
import gzip
import pickle
import gc
import sys
import os
import time
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

from cache_llm_predictions_amazon import (
    loadMyDataset, get_entity_embeddings as get_entity_embeddings_cache,
    V3_MODELS, V3_ST_MODEL,
    compute_similarities,
)


# ================================================================
# Subset wrapper — presents a slice of the dataset to generate functions
# ================================================================
class DatasetSubset:
    """Thin wrapper that makes a subset of the dataset look like a full dataset."""
    def __init__(self, dataset, indices):
        self.indices = np.array(sorted(indices))  # sorted for sequential access
        self.text_titles = [dataset.text_titles[i] for i in self.indices]
        self.text_contents = [dataset.text_contents[i] for i in self.indices]
    def __len__(self):
        return len(self.indices)


# ================================================================
# Sherman-Morrison rank-1 inverse update: O(d^2)
# ================================================================
def sm_update(V_inv, x):
    if not np.all(np.isfinite(x)) or np.linalg.norm(x) < 1e-10:
        return V_inv
    Vx = V_inv @ x
    denom = 1.0 + x @ Vx
    if abs(denom) < 1e-12:
        return V_inv
    return V_inv - np.outer(Vx, Vx) / denom


class EasyAcc:
    def __init__(self):
        self.n = 0
        self.sum = 0
    def __iadd__(self, v):
        self.n += 1
        self.sum += v
        return self
    def mean(self):
        return self.sum / max(self.n, 1)


# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Combined Qwen cache generation + Algorithm 1 experiment'
    )
    parser.add_argument('--llm_type', type=str, required=True,
                        choices=['small', 'base', 'large'])
    parser.add_argument('--max_samples', type=int, default=50000,
                        help='Number of samples for Algorithm 1 experiment')
    parser.add_argument('--llm_confidence', type=float, default=100.0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_vllm', action='store_true',
                        help='Use HuggingFace instead of vLLM')
    parser.add_argument('--skip_experiment', action='store_true',
                        help='Only generate cache, skip experiment')
    parser.add_argument('--top_k', type=int, default=100,
                        help='Store top-K sims per sample in cache')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    model_family = 'v3'
    llm_type = args.llm_type
    full_cache_path = f'llm_cache_amazon_v3_{llm_type}.pkl.gz'

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ============================================================
    # STEP 1: Load dataset
    # ============================================================
    print(f"\n{'='*70}")
    print(f"STEP 1: Loading v3 dataset")
    print(f"{'='*70}")
    dataset = loadMyDataset('v3')
    N_total = len(dataset)
    print(f"Dataset: {N_total} samples")

    entity_embd = get_entity_embeddings_cache(dataset)
    K = entity_embd.shape[0]
    print(f"Entities: {K}")

    # Determine which sample indices the experiment will use
    N = min(N_total, args.max_samples)
    np.random.seed(args.seed)
    perm = np.random.permutation(N)
    needed_indices = set(perm)  # the unique indices we need predictions for

    partial_cache_path = f'llm_cache_amazon_v3_{llm_type}_partial{N}.pkl.gz'

    # ============================================================
    # STEP 2: Get LLM predictions for experiment samples
    # ============================================================
    # Priority: full cache > partial cache > generate fresh
    llm_labels_map = {}  # idx -> predicted label

    if os.path.exists(full_cache_path):
        print(f"\n{'='*70}")
        print(f"STEP 2: Full cache {full_cache_path} exists, loading...")
        print(f"{'='*70}")
        with gzip.open(full_cache_path, 'rb') as f:
            cache = pickle.load(f)
        all_labels = cache['labels']
        for idx in needed_indices:
            llm_labels_map[idx] = int(all_labels[idx])
        print(f"  Loaded predictions for {len(llm_labels_map)} samples from full cache")

    elif os.path.exists(partial_cache_path):
        print(f"\n{'='*70}")
        print(f"STEP 2: Partial cache {partial_cache_path} exists, loading...")
        print(f"{'='*70}")
        with gzip.open(partial_cache_path, 'rb') as f:
            cache = pickle.load(f)
        llm_labels_map = cache['labels_map']
        missing = needed_indices - set(llm_labels_map.keys())
        print(f"  Loaded {len(llm_labels_map)} cached predictions, {len(missing)} missing")
        if missing:
            print(f"  WARNING: {len(missing)} needed samples not in partial cache, regenerating...")
            llm_labels_map = {}  # force regenerate

    if not llm_labels_map:
        print(f"\n{'='*70}")
        print(f"STEP 2: Generating Qwen-{llm_type} predictions for {N} experiment samples")
        print(f"  (only {N/N_total*100:.1f}% of full dataset)")
        print(f"{'='*70}")
        t0 = time.time()

        # Create subset dataset
        subset = DatasetSubset(dataset, needed_indices)
        print(f"  Subset: {len(subset)} samples")

        # Generate text predictions for the subset
        if args.no_vllm:
            from cache_llm_predictions_amazon import generate_v3_predictions_hf
            predictions = generate_v3_predictions_hf(subset, llm_type, device,
                                                     batch_size=args.batch_size)
        else:
            from cache_llm_predictions_amazon import generate_v3_predictions_vllm
            predictions = generate_v3_predictions_vllm(subset, llm_type)

        gen_time = time.time() - t0
        print(f"\nGenerated {len(predictions)} predictions in {gen_time:.0f}s")
        print(f"Sample predictions: {predictions[:5]}")

        # Compute similarities and get labels
        top_k = args.top_k if args.top_k > 0 else None
        labels, top_sims, top_indices = compute_similarities(
            predictions, entity_embd, V3_ST_MODEL, device, top_k=top_k
        )

        # Map back to original dataset indices
        for i, orig_idx in enumerate(subset.indices):
            llm_labels_map[int(orig_idx)] = int(labels[i])

        # Save partial cache
        cache = {
            'labels_map': llm_labels_map,
            'subset_indices': subset.indices,
            'labels': labels,
            'top_sims': top_sims,
            'top_indices': top_indices,
            'top_k': top_k,
            'config': {
                'dataset': 'amazoncat13k',
                'model_family': 'v3',
                'llm_type': llm_type,
                'llm_model': V3_MODELS[llm_type],
                'st_model': V3_ST_MODEL,
                'num_samples': len(predictions),
                'num_entities': K,
                'storage': f'partial-{N}',
                'seed': args.seed,
            }
        }
        print(f"Saving partial cache to {partial_cache_path}...")
        with gzip.open(partial_cache_path, 'wb') as f:
            pickle.dump(cache, f)
        print(f"Cache saved! ({len(labels)} predictions)")

        # Quick accuracy check
        correct = sum(1 for i, orig_idx in enumerate(subset.indices)
                      if labels[i] in dataset.ys[orig_idx])
        print(f"LLM accuracy (on experiment subset): {correct/len(labels)*100:.2f}%")

        total_time = time.time() - t0
        print(f"Total cache generation time: {total_time:.0f}s ({total_time/60:.1f}m)")

        # Free LLM memory
        del predictions, labels, top_sims, top_indices, subset
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if args.skip_experiment:
        print("\n--skip_experiment: done.")
        return

    # ============================================================
    # STEP 3: Run Algorithm 1 experiment
    # ============================================================
    print(f"\n{'='*70}")
    print(f"STEP 3: Algorithm 1 experiment (v3/{llm_type}, {args.max_samples} samples)")
    print(f"{'='*70}")

    # Get numpy arrays for the experiment
    Xs = dataset.Xs.numpy() if isinstance(dataset.Xs, torch.Tensor) else dataset.Xs
    d_full = Xs.shape[1]
    d = d_full // 2
    contexts = Xs[:, :d].astype(np.float32)
    contexts = np.nan_to_num(contexts, nan=0.0, posinf=0.0, neginf=0.0)

    entity_np = entity_embd.numpy() if isinstance(entity_embd, torch.Tensor) else entity_embd
    entity_np = entity_np.astype(np.float32)
    entity_np = np.nan_to_num(entity_np, nan=0.0, posinf=0.0, neginf=0.0)

    true_labels = dataset.ys

    print(f"Contexts: ({N}, {d}), Entities: ({K}, {d})")

    # --- Setup (reuse same seed/perm as above) ---
    # perm was already computed above with the same seed

    lam = 0.1
    linucb_alpha = 1.0
    alpha_decay = 0.9999
    inv_sqrt_c = 1.0 / np.sqrt(args.llm_confidence)

    V_inv_warm = (1.0 / lam) * np.eye(d, dtype=np.float64)
    b_warm = np.zeros(d, dtype=np.float64)
    V_inv_cold = (1.0 / lam) * np.eye(d, dtype=np.float64)
    b_cold = np.zeros(d, dtype=np.float64)

    acc_router = EasyAcc()
    acc_llm = EasyAcc()
    acc_cold = EasyAcc()
    llm_rounds = 0
    total_rounds = 0

    Z = entity_np.astype(np.float64)

    tag = f"algo1_amazon_v3_{llm_type}_c{args.llm_confidence}_pfull"
    fname = f"results_{tag}.json"

    history = {
        'step': [], 'hybrid': [], 'llm': [], 'linucb_cold': [],
        'llm_fraction': [],
        'config': {
            'algorithm': 'confidence_gated_routing',
            'dataset': 'amazoncat13k',
            'model_family': 'v3',
            'features': f'elementwise_{d}d',
            'linucb_alpha': linucb_alpha, 'lam': lam,
            'llm_type': llm_type, 'alpha_decay': alpha_decay,
            'llm_confidence': args.llm_confidence,
            'passive_full': True,
            'num_samples': N, 'num_actions': K,
        }
    }
    print(f"Results file: {fname}")
    print(f"Total experiment samples: {N}")
    print("=" * 90)

    t0 = time.time()
    for step_i in range(N):
        idx = perm[step_i]
        cx = contexts[idx].astype(np.float64)
        ys_true = true_labels[idx]
        llm_choice = llm_labels_map[idx]
        current_alpha = max(0.1, linucb_alpha * (alpha_decay ** step_i))
        total_rounds += 1

        # --- LLM track ---
        r_llm = 1.0 if llm_choice in ys_true else 0.0
        acc_llm += r_llm

        # --- Cold LinUCB ---
        theta_cold = V_inv_cold @ b_cold
        weighted_cold = cx * theta_cold
        mu_cold = Z @ weighted_cold
        Cx_cold = (V_inv_cold * cx[:, None]) * cx[None, :]
        ZCx_cold = Z @ Cx_cold
        var_cold = np.sum(ZCx_cold * Z, axis=1)
        ucb_cold = mu_cold + current_alpha * np.sqrt(np.maximum(var_cold, 0))
        ucb_cold = np.nan_to_num(ucb_cold, nan=-np.inf)
        cold_choice = int(np.argmax(ucb_cold))
        r_cold = 1.0 if cold_choice in ys_true else 0.0
        acc_cold += r_cold
        x_cold = cx * Z[cold_choice]
        V_inv_cold = sm_update(V_inv_cold, x_cold)
        b_cold += r_cold * x_cold

        # --- Router (Algorithm 1) ---
        theta_warm = V_inv_warm @ b_warm
        weighted_warm = cx * theta_warm
        mu_warm = Z @ weighted_warm
        Cx_warm = (V_inv_warm * cx[:, None]) * cx[None, :]
        ZCx_warm = Z @ Cx_warm
        var_warm = np.sum(ZCx_warm * Z, axis=1)
        ucb_warm = mu_warm + current_alpha * np.sqrt(np.maximum(var_warm, 0))
        ucb_warm = np.nan_to_num(ucb_warm, nan=-np.inf)
        linucb_choice = int(np.argmax(ucb_warm))

        x_llm = cx * Z[llm_choice]
        x_linucb = cx * Z[linucb_choice]
        u_llm = inv_sqrt_c * np.linalg.norm(x_llm)
        u_linucb = np.sqrt(max(0.0, x_linucb @ V_inv_warm @ x_linucb))

        if u_llm < u_linucb:
            router_choice = llm_choice
            r_router = 1.0 if router_choice in ys_true else 0.0
            llm_rounds += 1
            V_inv_warm = sm_update(V_inv_warm, x_llm)
            b_warm += r_router * x_llm
        else:
            router_choice = linucb_choice
            r_router = 1.0 if router_choice in ys_true else 0.0
            V_inv_warm = sm_update(V_inv_warm, x_linucb)
            b_warm += r_router * x_linucb

        acc_router += r_router

        # --- Log ---
        if (step_i + 1) % 1000 == 0 or step_i == N - 1:
            llm_frac = llm_rounds / max(total_rounds, 1)
            history['step'].append(step_i + 1)
            history['hybrid'].append(acc_router.mean())
            history['llm'].append(acc_llm.mean())
            history['linucb_cold'].append(acc_cold.mean())
            history['llm_fraction'].append(llm_frac)
            elapsed = time.time() - t0
            rate = (step_i + 1) / max(elapsed, 1)
            eta = (N - step_i - 1) / max(rate, 1)
            print(
                f"Step {step_i+1:7d}/{N} | "
                f"Router: {acc_router.mean():.4f} | "
                f"LLM: {acc_llm.mean():.4f} | "
                f"Cold: {acc_cold.mean():.4f} | "
                f"LLM%: {llm_frac:.1%} | "
                f"ETA: {eta/60:.0f}m"
            )
            with open(fname, "w") as f:
                json.dump(history, f)

    print("=" * 90)
    print(
        f"FINAL | Router: {acc_router.mean():.4f} | "
        f"LLM: {acc_llm.mean():.4f} | "
        f"Cold LinUCB: {acc_cold.mean():.4f}"
    )
    print(f"LLM routing: {llm_rounds}/{total_rounds} = {llm_rounds/max(total_rounds,1):.1%}")
    total_time = time.time() - t0
    print(f"Experiment time: {total_time:.0f}s ({total_time/60:.1f}m)")
    with open(fname, "w") as f:
        json.dump(history, f)
    print(f"Saved: {fname}")


if __name__ == '__main__':
    main()
