"""
run_experiment_v6.py — Full experiment on v6 (category-dependent complementarity) dataset.

Usage:
  python -u run_experiment_v6.py --model flan-t5-base
  python -u run_experiment_v6.py --all_local
  python -u run_experiment_v6.py --sweep_only
  python -u run_experiment_v6.py --model qwen-3b  # on cluster
"""

import argparse
import gzip
import json
import os
import pickle
import warnings

import numpy as np
from numpy.linalg import norm
from tqdm import tqdm

warnings.filterwarnings('ignore')

DATASET_PATH = 'synthetic_text_dataset_v6.pkl.gz'

MODEL_CONFIGS = {
    'flan-t5-small': {'hf_name': 'google/flan-t5-small', 'type': 'seq2seq'},
    'flan-t5-base': {'hf_name': 'google/flan-t5-base', 'type': 'seq2seq'},
    'flan-t5-large': {'hf_name': 'google/flan-t5-large', 'type': 'seq2seq'},
    'qwen-0.5b': {'hf_name': 'Qwen/Qwen2.5-0.5B-Instruct', 'type': 'causal'},
    'qwen-3b': {'hf_name': 'Qwen/Qwen2.5-3B-Instruct', 'type': 'causal'},
    'qwen-7b': {'hf_name': 'Qwen/Qwen2.5-7B-Instruct', 'type': 'causal'},
}


def load_dataset():
    print("Loading dataset...")
    with gzip.open(DATASET_PATH, 'rb') as f:
        dataset = pickle.load(f)
    N = dataset['doc_features'].shape[0]
    K = dataset['arm_features'].shape[0]
    d = dataset['doc_features'].shape[1]
    print(f"  {N} docs, {K} categories, d={d}")
    print(f"  LinUCB-group: {len(dataset['group_linucb_idx'])} cats, "
          f"LLM-group: {len(dataset['group_llm_idx'])} cats")
    return dataset


def build_prompt(text, category_names):
    cat_list = ", ".join(category_names)
    return f"Classify this text into exactly one of these categories: {cat_list}\n\nText: {text}\n\nCategory:"


def match_prediction_to_category(pred_text, category_names):
    pred_lower = pred_text.strip().lower()
    for k, cat in enumerate(category_names):
        if cat.lower() == pred_lower:
            return k
    for k, cat in enumerate(category_names):
        if cat.lower() in pred_lower:
            return k
    for k, cat in enumerate(category_names):
        if pred_lower in cat.lower():
            return k
    pred_words = set(pred_lower.split())
    best_k, best_score = 0, 0
    for k, cat in enumerate(category_names):
        cat_words = set(cat.lower().split())
        overlap = len(pred_words & cat_words)
        if overlap > best_score:
            best_score = overlap
            best_k = k
    return best_k if best_score > 0 else 0


def generate_predictions_flan_t5(dataset, model_name, batch_size=64):
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    hf_name = MODEL_CONFIGS[model_name]['hf_name']
    print(f"  Loading {hf_name}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(hf_name).eval()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        print(f"  Using CUDA: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        device = torch.device("cpu")
    print(f"  Device: {device}")

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
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    hf_name = MODEL_CONFIGS[model_name]['hf_name']
    # Reduce batch size for large models on limited GPU memory
    if 'qwen-7b' in model_name.lower() or '7B' in hf_name:
        batch_size = min(batch_size, 8)
        print(f"  Reduced batch_size to {batch_size} for 7B model")
    elif 'qwen-3b' in model_name.lower() or '3B' in hf_name:
        batch_size = min(batch_size, 16)
        print(f"  Reduced batch_size to {batch_size} for 3B model")
    print(f"  Loading {hf_name}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        hf_name, torch_dtype="auto", device_map="auto"
    ).eval()

    if torch.cuda.is_available():
        print(f"  GPU mem after load: {torch.cuda.memory_allocated()/1e9:.1f}GB / "
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

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


def generate_and_cache(dataset, model_name, batch_size=64):
    cache_path = f"v6_cache_{model_name}.pkl.gz"
    if os.path.exists(cache_path):
        print(f"  Loading existing cache: {cache_path}")
        with gzip.open(cache_path, 'rb') as f:
            return pickle.load(f)

    config = MODEL_CONFIGS[model_name]
    if config['type'] == 'seq2seq':
        labels = generate_predictions_flan_t5(dataset, model_name, batch_size)
    else:
        labels = generate_predictions_qwen(dataset, model_name, batch_size)

    ys = dataset['ys']
    llm_idx_set = set(dataset['group_llm_idx'])
    accuracy = float(np.mean(labels == ys))
    llm_grp_mask = np.array([y in llm_idx_set for y in ys])
    acc_llm_grp = float(np.mean(labels[llm_grp_mask] == ys[llm_grp_mask]))
    acc_linucb_grp = float(np.mean(labels[~llm_grp_mask] == ys[~llm_grp_mask]))

    cache = {
        'labels': labels,
        'accuracy': accuracy,
        'acc_llm_grp': acc_llm_grp,
        'acc_linucb_grp': acc_linucb_grp,
        'model_name': model_name,
    }

    with gzip.open(cache_path, 'wb') as f:
        pickle.dump(cache, f)
    print(f"  Saved: {cache_path}")
    print(f"  Overall: {accuracy*100:.2f}%  LLM-grp: {acc_llm_grp*100:.2f}%  "
          f"LinUCB-grp: {acc_linucb_grp*100:.2f}%")
    return cache


# ================================================================
# Algorithm 1 (with per-group tracking)
# ================================================================
def sm_update(V_inv, phi):
    Vp = V_inv @ phi
    denom = 1.0 + phi @ Vp
    if abs(denom) < 1e-12:
        return V_inv
    V_inv -= np.outer(Vp, Vp) / denom
    return V_inv


def run_algorithm1(dataset, llm_labels, T=5000, c=1.0, lam=1.0,
                   alpha=1.0, alpha_decay=0.9997, seed=42, ordering='random'):
    doc_features = dataset['doc_features'].astype(np.float64)
    arm_features = dataset['arm_features'].astype(np.float64)
    true_labels = dataset['ys']
    llm_idx_set = set(dataset['group_llm_idx'])
    N, d = doc_features.shape

    T = min(T, N)
    np.random.seed(seed)

    if ordering == 'random':
        perm = np.random.permutation(N)[:T]
    elif ordering == 'llm_first':
        # LLM-group docs first (maximize passive warm-start), then LinUCB-group
        llm_grp_mask = np.array([true_labels[i] in llm_idx_set for i in range(N)])
        llm_docs = np.where(llm_grp_mask)[0]
        linucb_docs = np.where(~llm_grp_mask)[0]
        np.random.shuffle(llm_docs)
        np.random.shuffle(linucb_docs)
        perm = np.concatenate([llm_docs, linucb_docs])[:T]
    elif ordering == 'interleaved':
        llm_grp_mask = np.array([true_labels[i] in llm_idx_set for i in range(N)])
        llm_docs = np.where(llm_grp_mask)[0]
        linucb_docs = np.where(~llm_grp_mask)[0]
        np.random.shuffle(llm_docs)
        np.random.shuffle(linucb_docs)
        interleaved = []
        li, ri = 0, 0
        while li < len(llm_docs) and ri < len(linucb_docs):
            interleaved.append(llm_docs[li]); li += 1
            interleaved.append(linucb_docs[ri]); ri += 1
        while li < len(llm_docs):
            interleaved.append(llm_docs[li]); li += 1
        while ri < len(linucb_docs):
            interleaved.append(linucb_docs[ri]); ri += 1
        perm = np.array(interleaved[:T])
    else:
        perm = np.random.permutation(N)[:T]

    V_inv_warm = np.eye(d, dtype=np.float64) / lam
    b_warm = np.zeros(d, dtype=np.float64)
    V_inv_cold = np.eye(d, dtype=np.float64) / lam
    b_cold = np.zeros(d, dtype=np.float64)

    inv_sqrt_c = 1.0 / np.sqrt(c)
    cum_router, cum_llm, cum_cold = 0.0, 0.0, 0.0
    # Per-group tracking
    cum_router_llmg, cum_router_linucbg = 0.0, 0.0
    n_llmg, n_linucbg = 0, 0
    llm_count = 0

    history = {'step': [], 'router': [], 'llm': [], 'cold': [],
               'llm_frac': [], 'router_llmg': [], 'router_linucbg': []}

    for t_i in range(T):
        idx = perm[t_i]
        x = doc_features[idx]
        true_y = true_labels[idx]
        llm_choice = int(llm_labels[idx])
        is_llm_grp = true_y in llm_idx_set
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
        # Compute LinUCB's own best arm via UCB
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

        # Per-group tracking
        if is_llm_grp:
            n_llmg += 1
            cum_router_llmg += r_router
        else:
            n_linucbg += 1
            cum_router_linucbg += r_router

        # Update V, b with chosen arm (passive for LLM rounds)
        phi_router = phi_all[router_choice]
        V_inv_warm = sm_update(V_inv_warm, phi_router)
        b_warm += r_router * phi_router

        if (t_i + 1) % 500 == 0 or t_i == T - 1:
            history['step'].append(t_i + 1)
            history['router'].append(cum_router / (t_i + 1))
            history['llm'].append(cum_llm / (t_i + 1))
            history['cold'].append(cum_cold / (t_i + 1))
            history['llm_frac'].append(llm_count / (t_i + 1))
            history['router_llmg'].append(cum_router_llmg / max(n_llmg, 1))
            history['router_linucbg'].append(cum_router_linucbg / max(n_linucbg, 1))

    return {
        'avg_router': cum_router / T,
        'avg_llm': cum_llm / T,
        'avg_cold': cum_cold / T,
        'llm_pct': llm_count / T * 100,
        'router_llmg': cum_router_llmg / max(n_llmg, 1),
        'router_linucbg': cum_router_linucbg / max(n_linucbg, 1),
        'T': T,
        'history': history,
    }


def run_c_sweep(dataset, cache, T=5000, seed=42, ordering='random', seeds=None):
    c_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    llm_labels = cache['labels']
    model_name = cache['model_name']

    # Multi-seed support
    if seeds is None:
        seeds = [seed]

    print(f"\n  C-SWEEP for {model_name} (acc={cache['accuracy']*100:.2f}%, "
          f"LLM-grp={cache.get('acc_llm_grp', 0)*100:.1f}%, "
          f"LinUCB-grp={cache.get('acc_linucb_grp', 0)*100:.1f}%) "
          f"ordering={ordering}, seeds={seeds}")
    print(f"  {'c':<8} {'Router':<9} {'LLM':<9} {'Cold':<9} {'Mult':<8} {'LLM%':<8} "
          f"{'R-LLMg':<9} {'R-Lg':<9} {'Std':<8} {'Note'}")
    print(f"  {'-'*95}")

    results = {}
    best_mult, best_c = 0, None

    for c in tqdm(c_values, desc=f"  c-sweep ({ordering})", leave=False):
        seed_results = []
        for s in seeds:
            res = run_algorithm1(dataset, llm_labels, T=T, c=c, seed=s, ordering=ordering)
            seed_results.append(res)

        # Average across seeds
        avg_router = np.mean([r['avg_router'] for r in seed_results])
        avg_llm = np.mean([r['avg_llm'] for r in seed_results])
        avg_cold = np.mean([r['avg_cold'] for r in seed_results])
        avg_llm_pct = np.mean([r['llm_pct'] for r in seed_results])
        avg_router_llmg = np.mean([r['router_llmg'] for r in seed_results])
        avg_router_linucbg = np.mean([r['router_linucbg'] for r in seed_results])
        std_router = np.std([r['avg_router'] for r in seed_results])

        mult = avg_router / max(avg_llm, avg_cold, 1e-8)
        note = f"*** MULT {mult:.2f} ***" if mult > 1.01 else ""
        print(f"  {c:<8.3f} {avg_router:<9.4f} {avg_llm:<9.4f} "
              f"{avg_cold:<9.4f} {mult:<8.4f} {avg_llm_pct:<8.1f} "
              f"{avg_router_llmg:<9.4f} {avg_router_linucbg:<9.4f} {std_router:<8.4f} {note}")
        results[str(c)] = {
            'avg_router': float(avg_router),
            'avg_llm': float(avg_llm),
            'avg_cold': float(avg_cold),
            'llm_pct': float(avg_llm_pct),
            'multiplication': float(mult),
            'router_llmg': float(avg_router_llmg),
            'router_linucbg': float(avg_router_linucbg),
            'std_router': float(std_router),
            'std_cold': float(np.std([r['avg_cold'] for r in seed_results])),
            'std_mult': float(np.std([r['avg_router'] / max(r['avg_llm'], r['avg_cold'], 1e-8)
                                       for r in seed_results])),
            'per_seed': [{
                'seed': s,
                'avg_router': r['avg_router'],
                'avg_llm': r['avg_llm'],
                'avg_cold': r['avg_cold'],
                'llm_pct': r['llm_pct'],
                'router_llmg': r['router_llmg'],
                'router_linucbg': r['router_linucbg'],
            } for s, r in zip(seeds, seed_results)],
        }
        if mult > best_mult:
            best_mult = mult
            best_c = c

    print(f"\n  BEST: c={best_c} → mult={best_mult:.4f}")
    return results, best_c, best_mult


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--all_local', action='store_true')
    parser.add_argument('--all_cluster', action='store_true')
    parser.add_argument('--skip_cache', action='store_true')
    parser.add_argument('--sweep_only', action='store_true')
    parser.add_argument('--T', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_seeds', type=int, default=1,
                       help='Number of seeds for c-sweep (uses seed, seed+1, ...)')
    parser.add_argument('--ordering', type=str, default='random',
                       choices=['random', 'llm_first', 'interleaved'])
    args = parser.parse_args()

    dataset = load_dataset()

    # Build seeds list
    seeds_list = [args.seed + i for i in range(args.num_seeds)]

    if args.sweep_only:
        cache_files = sorted([f for f in os.listdir('.')
                              if f.startswith('v6_cache_') and f.endswith('.pkl.gz')])
        if not cache_files:
            print("No v6 caches found.")
            return
        all_results = {}
        for cf in cache_files:
            with gzip.open(cf, 'rb') as f:
                cache = pickle.load(f)
            results, best_c, best_mult = run_c_sweep(
                dataset, cache, T=args.T, seed=args.seed, ordering=args.ordering,
                seeds=seeds_list)
            all_results[cache['model_name']] = {
                'accuracy': cache['accuracy'],
                'acc_llm_grp': cache.get('acc_llm_grp', 0),
                'acc_linucb_grp': cache.get('acc_linucb_grp', 0),
                'c_sweep': results,
                'best_c': best_c,
                'best_mult': best_mult,
                'num_seeds': args.num_seeds,
                'seeds': seeds_list,
            }
        out_path = f'v6_results_sweep_{args.ordering}.json'
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved: {out_path}")

        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        for mk, res in all_results.items():
            print(f"  {mk:<16} acc={res['accuracy']*100:.1f}% "
                  f"best_c={res['best_c']} mult={res['best_mult']:.4f}")
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
            cache_path = f"v6_cache_{model_name}.pkl.gz"
            if not os.path.exists(cache_path):
                print(f"  Cache not found: {cache_path}")
                continue
            with gzip.open(cache_path, 'rb') as f:
                cache = pickle.load(f)
        else:
            cache = generate_and_cache(dataset, model_name, args.batch_size)

        print(f"  Accuracy: {cache['accuracy']*100:.2f}% "
              f"(LLM-grp: {cache.get('acc_llm_grp', 0)*100:.1f}%, "
              f"LinUCB-grp: {cache.get('acc_linucb_grp', 0)*100:.1f}%)")

        results, best_c, best_mult = run_c_sweep(
            dataset, cache, T=args.T, seed=args.seed, ordering=args.ordering,
            seeds=seeds_list)
        all_results[model_name] = {
            'accuracy': cache['accuracy'],
            'acc_llm_grp': cache.get('acc_llm_grp', 0),
            'acc_linucb_grp': cache.get('acc_linucb_grp', 0),
            'c_sweep': results,
            'best_c': best_c,
            'best_mult': best_mult,
            'num_seeds': args.num_seeds,
            'seeds': seeds_list,
        }

    out_path = f'v6_results_{args.ordering}.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for mk, res in all_results.items():
        print(f"  {mk:<16} acc={res['accuracy']*100:.1f}% "
              f"best_c={res['best_c']} mult={res['best_mult']:.4f}")
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
