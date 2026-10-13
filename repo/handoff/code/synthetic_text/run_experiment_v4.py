"""
run_experiment_v4.py — Full experiment pipeline for v4 dataset.

Pipeline:
  1. Load synthetic_text_dataset_v4.pkl.gz (10000 docs, 20 categories, d=200)
  2. Run LLM → cache predictions (or load existing cache)
  3. Run Algorithm 1 with c-sweep
  4. Report multiplication results

Usage:
  # Quick test (flan-t5-small, already cached):
  python -u run_experiment_v4.py --model flan-t5-small

  # All local models:
  python -u run_experiment_v4.py --all_local

  # Cluster models:
  python -u run_experiment_v4.py --model qwen-3b

  # Skip cache (use existing):
  python -u run_experiment_v4.py --model flan-t5-small --skip_cache

  # Just run the c-sweep on all existing caches:
  python -u run_experiment_v4.py --sweep_only
"""

import argparse
import gzip
import json
import os
import pickle
import time
import warnings

import numpy as np
from numpy.linalg import norm
from scipy.linalg import svd, subspace_angles
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ================================================================
# Model configs
# ================================================================
MODEL_CONFIGS = {
    'flan-t5-small': {'hf_name': 'google/flan-t5-small', 'type': 'seq2seq'},
    'flan-t5-base': {'hf_name': 'google/flan-t5-base', 'type': 'seq2seq'},
    'flan-t5-large': {'hf_name': 'google/flan-t5-large', 'type': 'seq2seq'},
    'qwen-0.5b': {'hf_name': 'Qwen/Qwen2.5-0.5B-Instruct', 'type': 'causal'},
    'qwen-3b': {'hf_name': 'Qwen/Qwen2.5-3B-Instruct', 'type': 'causal'},
    'qwen-7b': {'hf_name': 'Qwen/Qwen2.5-7B-Instruct', 'type': 'causal'},
}

DATASET_PATH = 'synthetic_text_dataset_v4.pkl.gz'


def load_dataset():
    print("Loading dataset...")
    with gzip.open(DATASET_PATH, 'rb') as f:
        dataset = pickle.load(f)
    N = dataset['doc_features'].shape[0]
    K = dataset['arm_features'].shape[0]
    d = dataset['doc_features'].shape[1]
    print(f"  {N} documents, {K} categories, d={d}")
    return dataset


# ================================================================
# LLM Prediction Cache
# ================================================================
def build_prompt(text, category_names):
    cat_list = ", ".join(category_names)
    return f"Classify this text into exactly one of these categories: {cat_list}\n\nText: {text}\n\nCategory:"


def match_prediction_to_category(pred_text, category_names):
    """Match LLM output text to a category index."""
    pred_lower = pred_text.strip().lower()
    # Exact match
    for k, cat in enumerate(category_names):
        if cat.lower() == pred_lower:
            return k
    # Substring match
    for k, cat in enumerate(category_names):
        if cat.lower() in pred_lower:
            return k
    # Reverse substring
    for k, cat in enumerate(category_names):
        if pred_lower in cat.lower():
            return k
    # Word overlap
    pred_words = set(pred_lower.split())
    best_k, best_score = 0, 0
    for k, cat in enumerate(category_names):
        cat_words = set(cat.lower().split())
        overlap = len(pred_words & cat_words)
        if overlap > best_score:
            best_score = overlap
            best_k = k
    if best_score > 0:
        return best_k
    return 0  # fallback


def generate_predictions_flan_t5(dataset, model_name, batch_size=64):
    """Batched Flan-T5 inference."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    
    hf_name = MODEL_CONFIGS[model_name]['hf_name']
    print(f"  Loading {hf_name}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(hf_name)
    model.eval()
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
        print(f"  Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print(f"  Using CPU")
    
    documents = dataset['documents']
    category_names = dataset['category_names']
    N = len(documents)
    
    labels = np.zeros(N, dtype=int)
    prompts = [build_prompt(doc['text'], category_names) for doc in documents]
    
    for start in tqdm(range(0, N, batch_size), desc=f"  {model_name}", unit="batch"):
        end = min(start + batch_size, N)
        batch_prompts = prompts[start:end]
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True,
                          truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=30)
        
        for j, out in enumerate(outputs):
            pred_text = tokenizer.decode(out, skip_special_tokens=True)
            labels[start + j] = match_prediction_to_category(pred_text, category_names)
    
    return labels


def generate_predictions_qwen(dataset, model_name, batch_size=32):
    """Batched Qwen inference."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    hf_name = MODEL_CONFIGS[model_name]['hf_name']
    print(f"  Loading {hf_name}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        hf_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    
    documents = dataset['documents']
    category_names = dataset['category_names']
    N = len(documents)
    
    labels = np.zeros(N, dtype=int)
    prompts = [build_prompt(doc['text'], category_names) for doc in documents]
    
    for start in tqdm(range(0, N, batch_size), desc=f"  {model_name}", unit="batch"):
        end = min(start + batch_size, N)
        batch_prompts = prompts[start:end]
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True,
                          truncation=True, max_length=512).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=30,
                                    pad_token_id=tokenizer.pad_token_id)
        
        for j in range(end - start):
            input_len = inputs['input_ids'][j].shape[0]
            new_tokens = outputs[j][input_len:]
            pred_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            labels[start + j] = match_prediction_to_category(pred_text, category_names)
    
    return labels


def generate_and_cache_predictions(dataset, model_name, batch_size=64):
    """Generate predictions and save cache."""
    cache_path = f"v4_llm_cache_{model_name}.pkl.gz"
    
    if os.path.exists(cache_path):
        print(f"  Loading existing cache: {cache_path}")
        with gzip.open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    config = MODEL_CONFIGS[model_name]
    if config['type'] == 'seq2seq':
        labels = generate_predictions_flan_t5(dataset, model_name, batch_size)
    else:
        labels = generate_predictions_qwen(dataset, model_name, batch_size)
    
    # Compute accuracy
    ys = dataset['ys']
    accuracy = float(np.mean(labels == ys))
    
    # Per-type accuracy
    documents = dataset['documents']
    llm_friendly_mask = np.array([d['is_llm_friendly'] for d in documents])
    acc_llm_type = float(np.mean(labels[llm_friendly_mask] == ys[llm_friendly_mask]))
    acc_linucb_type = float(np.mean(labels[~llm_friendly_mask] == ys[~llm_friendly_mask]))
    
    cache = {
        'labels': labels,
        'accuracy': accuracy,
        'acc_llm_friendly': acc_llm_type,
        'acc_linucb_friendly': acc_linucb_type,
        'model_name': model_name,
    }
    
    with gzip.open(cache_path, 'wb') as f:
        pickle.dump(cache, f)
    print(f"  Saved cache: {cache_path}")
    
    return cache


# ================================================================
# Algorithm 1: Confidence-Gated Routing
# ================================================================
def sm_update(V_inv, phi):
    Vp = V_inv @ phi
    denom = 1.0 + phi @ Vp
    if abs(denom) < 1e-12:
        return V_inv
    V_inv -= np.outer(Vp, Vp) / denom
    return V_inv


def run_algorithm1(dataset, llm_labels, T=8000, c=1.0, lam=0.1,
                   alpha=1.0, alpha_decay=0.9997, seed=42):
    """Run Algorithm 1 with given c parameter."""
    doc_features = dataset['doc_features'].astype(np.float64)
    arm_features = dataset['arm_features'].astype(np.float64)
    true_labels = dataset['ys']
    N, d = doc_features.shape
    K = arm_features.shape[0]

    T = min(T, N)
    np.random.seed(seed)
    perm = np.random.permutation(N)[:T]

    V_inv_warm = np.eye(d, dtype=np.float64) / lam
    b_warm = np.zeros(d, dtype=np.float64)
    V_inv_cold = np.eye(d, dtype=np.float64) / lam
    b_cold = np.zeros(d, dtype=np.float64)

    inv_sqrt_c = 1.0 / np.sqrt(c)
    cum_router, cum_llm, cum_cold = 0.0, 0.0, 0.0
    llm_count = 0

    history = {'step': [], 'router': [], 'llm': [], 'cold': [], 'llm_frac': []}

    for t_i in range(T):
        idx = perm[t_i]
        x = doc_features[idx]
        true_y = true_labels[idx]
        llm_choice = int(llm_labels[idx])
        current_alpha = max(0.05, alpha * (alpha_decay ** t_i))

        r_llm = 1.0 if llm_choice == true_y else 0.0
        cum_llm += r_llm

        # Cold LinUCB
        theta_cold = V_inv_cold @ b_cold
        phi_all = x[None, :] * arm_features
        scores_cold = phi_all @ theta_cold
        V_phi = phi_all @ V_inv_cold
        var_cold = np.sum(V_phi * phi_all, axis=1)
        ucb_cold = scores_cold + current_alpha * np.sqrt(np.maximum(var_cold, 0))
        cold_choice = int(np.argmax(ucb_cold))
        r_cold = 1.0 if cold_choice == true_y else 0.0
        cum_cold += r_cold
        phi_cold = phi_all[cold_choice]
        V_inv_cold = sm_update(V_inv_cold, phi_cold)
        b_cold += r_cold * phi_cold

        # Router (Algorithm 1)
        # First compute LinUCB's own best arm
        theta_warm = V_inv_warm @ b_warm
        scores_warm = phi_all @ theta_warm
        V_phi_w = phi_all @ V_inv_warm
        var_warm = np.sum(V_phi_w * phi_all, axis=1)
        ucb_warm = scores_warm + current_alpha * np.sqrt(np.maximum(var_warm, 0))
        linucb_choice = int(np.argmax(ucb_warm))

        # Uncertainty on each method's OWN chosen arm
        phi_llm = phi_all[llm_choice]
        phi_linucb = phi_all[linucb_choice]
        u_llm = norm(phi_llm) * inv_sqrt_c
        u_linucb = np.sqrt(max(0, phi_linucb @ V_inv_warm @ phi_linucb))

        if u_llm < u_linucb:
            router_choice = llm_choice
            llm_count += 1
        else:
            router_choice = linucb_choice

        r_router = 1.0 if router_choice == true_y else 0.0
        cum_router += r_router
        phi_router = phi_all[router_choice]
        V_inv_warm = sm_update(V_inv_warm, phi_router)
        b_warm += r_router * phi_router

        if (t_i + 1) % 500 == 0 or t_i == T - 1:
            history['step'].append(t_i + 1)
            history['router'].append(cum_router / (t_i + 1))
            history['llm'].append(cum_llm / (t_i + 1))
            history['cold'].append(cum_cold / (t_i + 1))
            history['llm_frac'].append(llm_count / (t_i + 1))

    return {
        'avg_router': cum_router / T,
        'avg_llm': cum_llm / T,
        'avg_cold': cum_cold / T,
        'llm_pct': llm_count / T * 100,
        'T': T,
        'history': history,
    }


def run_c_sweep(dataset, cache, T=8000, seed=42):
    """Run Algorithm 1 across multiple c values."""
    c_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    llm_labels = cache['labels']
    model_name = cache['model_name']
    
    print(f"\n  C-SWEEP for {model_name} (acc={cache['accuracy']*100:.2f}%)")
    print(f"  {'c':<8} {'Router':<8} {'LLM':<8} {'Cold':<8} {'Mult':<8} {'LLM%':<8} {'Note'}")
    print(f"  {'-'*56}")
    
    results = {}
    best_mult = 0
    best_c = None
    
    for c in tqdm(c_values, desc=f"  c-sweep", leave=False):
        res = run_algorithm1(dataset, llm_labels, T=T, c=c, seed=seed)
        mult = res['avg_router'] / max(res['avg_llm'], res['avg_cold'], 1e-8)
        note = "*** MULT > 1 ***" if mult > 1.0 else ""
        print(f"  {c:<8.3f} {res['avg_router']:<8.4f} {res['avg_llm']:<8.4f} "
              f"{res['avg_cold']:<8.4f} {mult:<8.4f} {res['llm_pct']:<8.1f} {note}")
        
        results[str(c)] = {
            'avg_router': res['avg_router'],
            'avg_llm': res['avg_llm'],
            'avg_cold': res['avg_cold'],
            'llm_pct': res['llm_pct'],
            'multiplication': float(mult),
            'history': res['history'],
        }
        
        if mult > best_mult:
            best_mult = mult
            best_c = c
    
    print(f"\n  BEST: c={best_c} → mult={best_mult:.4f}")
    return results, best_c, best_mult


# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--all_local', action='store_true')
    parser.add_argument('--all_cluster', action='store_true')
    parser.add_argument('--skip_cache', action='store_true')
    parser.add_argument('--sweep_only', action='store_true',
                        help='Only run c-sweep on existing caches')
    parser.add_argument('--T', type=int, default=8000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    dataset = load_dataset()
    
    if args.sweep_only:
        # Run c-sweep on all existing v4 caches
        cache_files = sorted([f for f in os.listdir('.') if f.startswith('v4_llm_cache_') and f.endswith('.pkl.gz')])
        if not cache_files:
            print("ERROR: No v4 caches found. Run with --model first.")
            return
        
        all_results = {}
        for cf in cache_files:
            with gzip.open(cf, 'rb') as f:
                cache = pickle.load(f)
            results, best_c, best_mult = run_c_sweep(dataset, cache, T=args.T, seed=args.seed)
            all_results[cache['model_name']] = results
        
        with open('v4_results_c_sweep.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved: v4_results_c_sweep.json")
        return

    if args.all_local:
        models = ['flan-t5-small', 'flan-t5-base']
    elif args.all_cluster:
        models = ['qwen-0.5b', 'qwen-3b', 'qwen-7b']
    elif args.model:
        models = [args.model]
    else:
        parser.error("Specify --model, --all_local, --all_cluster, or --sweep_only")

    all_results = {}
    
    for model_name in models:
        print(f"\n{'='*70}")
        print(f"MODEL: {model_name}")
        print(f"{'='*70}")
        
        if args.skip_cache:
            cache_path = f"v4_llm_cache_{model_name}.pkl.gz"
            if not os.path.exists(cache_path):
                print(f"  ERROR: Cache not found: {cache_path}")
                continue
            with gzip.open(cache_path, 'rb') as f:
                cache = pickle.load(f)
        else:
            cache = generate_and_cache_predictions(dataset, model_name, args.batch_size)
        
        print(f"  Accuracy: {cache['accuracy']*100:.2f}%")
        print(f"    LLM-friendly: {cache.get('acc_llm_friendly', 'N/A')}")
        print(f"    LinUCB-friendly: {cache.get('acc_linucb_friendly', 'N/A')}")
        
        # C-sweep
        results, best_c, best_mult = run_c_sweep(dataset, cache, T=args.T, seed=args.seed)
        all_results[model_name] = {
            'accuracy': cache['accuracy'],
            'c_sweep': results,
            'best_c': best_c,
            'best_mult': best_mult,
        }
    
    # Save
    out_path = 'v4_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for mk, res in all_results.items():
        print(f"  {mk:<16} acc={res['accuracy']*100:.1f}% best_c={res['best_c']} mult={res['best_mult']:.4f}")


if __name__ == '__main__':
    main()
