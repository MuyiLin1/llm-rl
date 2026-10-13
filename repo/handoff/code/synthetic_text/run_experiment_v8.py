"""
run_experiment_v8.py — LLM Inference + Algorithm 1 for v8 dataset (K=124)

Usage on SSH:
  # Step 1: Generate LLM predictions (one-time)
  python run_experiment_v8.py --mode predict --model flan-t5-large
  python run_experiment_v8.py --mode predict --model qwen-7b

  # Step 2: Run Algorithm 1 with c-sweep
  python run_experiment_v8.py --mode experiment --model flan-t5-large --T 10000 --c_sweep
  python run_experiment_v8.py --mode experiment --model flan-t5-large --T 10000 --c 1.0

  # Step 3: Run all models
  python run_experiment_v8.py --mode all --T 10000
"""

import numpy as np
import json
import gzip
import pickle
import os
import argparse
from numpy.linalg import norm
from tqdm import tqdm

DATASET_PATH = 'synthetic_text_dataset_v8.pkl.gz'

MODEL_CONFIGS = {
    'flan-t5-small': {'hf_name': 'google/flan-t5-small', 'type': 'seq2seq'},
    'flan-t5-base': {'hf_name': 'google/flan-t5-base', 'type': 'seq2seq'},
    'flan-t5-large': {'hf_name': 'google/flan-t5-large', 'type': 'seq2seq'},
    'qwen-0.5b': {'hf_name': 'Qwen/Qwen2.5-0.5B-Instruct', 'type': 'causal'},
    'qwen-3b': {'hf_name': 'Qwen/Qwen2.5-3B-Instruct', 'type': 'causal'},
    'qwen-7b': {'hf_name': 'Qwen/Qwen2.5-7B-Instruct', 'type': 'causal'},
}


# ================================================================
# Dataset Loading
# ================================================================
def load_dataset():
    print(f"Loading {DATASET_PATH}...")
    with gzip.open(DATASET_PATH, 'rb') as f:
        dataset = pickle.load(f)
    N = dataset['doc_features'].shape[0]
    K = dataset['arm_features'].shape[0]
    d = dataset['doc_features'].shape[1]
    print(f"  {N} docs, {K} categories, d={d}")
    print(f"  Diagnostics: α̂={dataset['diagnostics']['alpha_hat']:.4f}, "
          f"r̂={dataset['diagnostics']['r_hat']:.1f}")
    return dataset


# ================================================================
# LLM Prediction
# ================================================================
def build_prompt(text, category_names):
    cat_list = ", ".join(category_names)
    return (f"Classify this text into exactly one of these categories: {cat_list}\n\n"
            f"Text: {text}\n\nCategory:")


def match_prediction_to_category(pred_text, category_names):
    """Match LLM output text to the closest category name."""
    pred_lower = pred_text.strip().lower()
    # Exact match
    for k, cat in enumerate(category_names):
        if cat.lower() == pred_lower:
            return k
    # Category name in prediction
    for k, cat in enumerate(category_names):
        if cat.lower() in pred_lower:
            return k
    # Prediction in category name
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
    return best_k if best_score > 0 else np.random.randint(len(category_names))


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
        print(f"  CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        device = torch.device("cpu")
    print(f"  Device: {device}")

    texts = dataset['texts']
    category_names = dataset['category_names']
    N = len(texts)
    labels = np.zeros(N, dtype=int)

    for start in tqdm(range(0, N, batch_size), desc=f"  {model_name}"):
        end = min(start + batch_size, N)
        batch_prompts = [build_prompt(texts[i], category_names) for i in range(start, end)]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True,
                          truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=30)
        for j, out in enumerate(outputs):
            pred_text = tokenizer.decode(out, skip_special_tokens=True)
            labels[start + j] = match_prediction_to_category(pred_text, category_names)

    return labels


def generate_predictions_qwen(dataset, model_name, batch_size=16):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    hf_name = MODEL_CONFIGS[model_name]['hf_name']
    if '7b' in model_name.lower() or '7B' in hf_name:
        batch_size = min(batch_size, 4)
    elif '3b' in model_name.lower() or '3B' in hf_name:
        batch_size = min(batch_size, 8)
    print(f"  Loading {hf_name} (batch_size={batch_size})...")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        hf_name, torch_dtype="auto", device_map="auto"
    ).eval()

    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB used")

    texts = dataset['texts']
    category_names = dataset['category_names']
    N = len(texts)
    labels = np.zeros(N, dtype=int)

    for start in tqdm(range(0, N, batch_size), desc=f"  {model_name}"):
        end = min(start + batch_size, N)
        batch_prompts = [build_prompt(texts[i], category_names) for i in range(start, end)]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True,
                          truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=30,
                                    pad_token_id=tokenizer.pad_token_id)
        for j in range(end - start):
            input_len = inputs['input_ids'][j].shape[0]
            new_tokens = outputs[j][input_len:]
            pred_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            labels[start + j] = match_prediction_to_category(pred_text, category_names)

    return labels


def predict_and_cache(dataset, model_name, batch_size=64):
    """Generate LLM predictions and cache them."""
    cache_path = f"v8_cache_{model_name}.pkl.gz"
    if os.path.exists(cache_path):
        print(f"  Cache exists: {cache_path}")
        with gzip.open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        acc = np.mean(cache['labels'] == dataset['ys'])
        print(f"  LLM accuracy: {acc:.3f}")
        return cache

    config = MODEL_CONFIGS[model_name]
    if config['type'] == 'seq2seq':
        labels = generate_predictions_flan_t5(dataset, model_name, batch_size)
    else:
        labels = generate_predictions_qwen(dataset, model_name, batch_size)

    acc = np.mean(labels == dataset['ys'])
    print(f"  LLM accuracy: {acc:.3f} (random: {1/len(dataset['category_names']):.4f})")

    cache = {'labels': labels, 'accuracy': float(acc), 'model': model_name}
    with gzip.open(cache_path, 'wb') as f:
        pickle.dump(cache, f)
    print(f"  Saved to {cache_path}")
    return cache


# ================================================================
# Sherman-Morrison rank-1 inverse update
# ================================================================
def sm_update(V_inv, x):
    Vx = V_inv @ x
    denom = 1.0 + x @ Vx
    if abs(denom) < 1e-12:
        return V_inv
    return V_inv - np.outer(Vx, Vx) / denom


# ================================================================
# Algorithm 1: Confidence-Gated Router
# ================================================================
def run_algorithm1(dataset, llm_labels, T=10000, c=1.0, lam=0.1,
                   alpha=1.0, alpha_decay=0.9999, seed=42):
    """Run Algorithm 1 on the v8 dataset."""
    doc_features = dataset['doc_features'].astype(np.float64)
    arm_features = dataset['arm_features'].astype(np.float64)
    true_labels = dataset['ys']

    N, d = doc_features.shape
    K = arm_features.shape[0]
    T = min(T, N)

    np.random.seed(seed)
    perm = np.random.permutation(N)[:T]

    # Z = arm feature matrix (K, d)
    Z = arm_features

    # Warm LinUCB (Router)
    V_inv_warm = (1.0 / lam) * np.eye(d, dtype=np.float64)
    b_warm = np.zeros(d, dtype=np.float64)

    # Cold LinUCB
    V_inv_cold = (1.0 / lam) * np.eye(d, dtype=np.float64)
    b_cold = np.zeros(d, dtype=np.float64)

    inv_sqrt_c = 1.0 / np.sqrt(c)

    cum_router = 0
    cum_llm = 0
    cum_cold = 0
    llm_rounds = 0

    history = {'step': [], 'router': [], 'llm': [], 'cold': [], 'llm_frac': []}
    log_interval = max(1, T // 50)

    for t in range(T):
        idx = perm[t]
        cx = doc_features[idx].astype(np.float64)
        y_true = true_labels[idx]
        y_llm = int(llm_labels[idx])

        current_alpha = max(0.1, alpha * (alpha_decay ** t))

        # --- LLM baseline ---
        r_llm = 1.0 if y_llm == y_true else 0.0
        cum_llm += r_llm

        # --- Cold LinUCB ---
        theta_cold = V_inv_cold @ b_cold
        weighted_cold = cx * theta_cold
        mu_cold = Z @ weighted_cold

        Cx_cold = (V_inv_cold * cx[:, None]) * cx[None, :]
        ZCx_cold = Z @ Cx_cold
        var_cold = np.sum(ZCx_cold * Z, axis=1)
        ucb_cold = mu_cold + current_alpha * np.sqrt(np.maximum(var_cold, 0))
        cold_choice = int(np.argmax(ucb_cold))

        r_cold = 1.0 if cold_choice == y_true else 0.0
        cum_cold += r_cold

        phi_cold = cx * Z[cold_choice]
        V_inv_cold = sm_update(V_inv_cold, phi_cold)
        b_cold += r_cold * phi_cold

        # --- Router (Algorithm 1) ---
        theta_warm = V_inv_warm @ b_warm
        weighted_warm = cx * theta_warm
        mu_warm = Z @ weighted_warm

        Cx_warm = (V_inv_warm * cx[:, None]) * cx[None, :]
        ZCx_warm = Z @ Cx_warm
        var_warm = np.sum(ZCx_warm * Z, axis=1)
        ucb_warm = mu_warm + current_alpha * np.sqrt(np.maximum(var_warm, 0))
        linucb_choice = int(np.argmax(ucb_warm))

        phi_llm = cx * Z[y_llm]
        phi_linucb = cx * Z[linucb_choice]

        u_llm = inv_sqrt_c * norm(phi_llm)
        u_linucb = np.sqrt(max(0.0, phi_linucb @ V_inv_warm @ phi_linucb))

        if u_llm < u_linucb:
            router_choice = y_llm
            r_router = 1.0 if router_choice == y_true else 0.0
            llm_rounds += 1
            V_inv_warm = sm_update(V_inv_warm, phi_llm)
            b_warm += r_router * phi_llm
        else:
            router_choice = linucb_choice
            r_router = 1.0 if router_choice == y_true else 0.0
            V_inv_warm = sm_update(V_inv_warm, phi_linucb)
            b_warm += r_router * phi_linucb

        cum_router += r_router

        if (t + 1) % log_interval == 0 or t == T - 1:
            history['step'].append(t + 1)
            history['router'].append(cum_router / (t + 1))
            history['llm'].append(cum_llm / (t + 1))
            history['cold'].append(cum_cold / (t + 1))
            history['llm_frac'].append(llm_rounds / (t + 1))

    avg_router = cum_router / T
    avg_llm = cum_llm / T
    avg_cold = cum_cold / T
    llm_pct = llm_rounds / T
    mult = avg_router / max(avg_llm, avg_cold, 1e-8)

    return {
        'avg_router': avg_router,
        'avg_llm': avg_llm,
        'avg_cold': avg_cold,
        'llm_pct': llm_pct,
        'multiplication': mult,
        'T': T,
        'c': c,
        'seed': seed,
        'history': history,
    }


# ================================================================
# C-sweep
# ================================================================
def run_c_sweep(dataset, llm_labels, T=10000, seeds=None, c_values=None):
    if seeds is None:
        seeds = [42, 43, 44]
    if c_values is None:
        c_values = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]

    results = {}
    best_mult = 0
    best_c = None

    for c in c_values:
        seed_results = []
        for s in seeds:
            r = run_algorithm1(dataset, llm_labels, T=T, c=c, seed=s)
            seed_results.append(r)
        
        avg_router = np.mean([r['avg_router'] for r in seed_results])
        avg_cold = np.mean([r['avg_cold'] for r in seed_results])
        avg_llm = np.mean([r['avg_llm'] for r in seed_results])
        llm_pct = np.mean([r['llm_pct'] for r in seed_results])
        mult = avg_router / max(avg_llm, avg_cold, 1e-8)

        print(f"  c={c:8.2f}: router={avg_router:.4f} cold={avg_cold:.4f} "
              f"llm={avg_llm:.4f} mult={mult:.3f} llm%={llm_pct:.1%}")

        results[str(c)] = {
            'avg_router': float(avg_router),
            'avg_cold': float(avg_cold),
            'avg_llm': float(avg_llm),
            'llm_pct': float(llm_pct),
            'multiplication': float(mult),
            'std_router': float(np.std([r['avg_router'] for r in seed_results])),
            'std_cold': float(np.std([r['avg_cold'] for r in seed_results])),
        }

        if mult > best_mult:
            best_mult = mult
            best_c = c

    return results, best_c, best_mult


# ================================================================
# Main
# ================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['predict', 'experiment', 'all'],
                        default='experiment')
    parser.add_argument('--model', type=str, default='flan-t5-large')
    parser.add_argument('--T', type=int, default=10000)
    parser.add_argument('--c', type=float, default=1.0)
    parser.add_argument('--c_sweep', action='store_true')
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    dataset = load_dataset()

    if args.mode == 'predict':
        print(f"\n--- Generating predictions: {args.model} ---")
        predict_and_cache(dataset, args.model, args.batch_size)

    elif args.mode == 'experiment':
        print(f"\n--- Loading LLM cache: {args.model} ---")
        cache = predict_and_cache(dataset, args.model, args.batch_size)
        llm_labels = cache['labels']

        seeds_list = [42 + i for i in range(args.seeds)]

        if args.c_sweep:
            print(f"\n--- C-Sweep: {args.model}, T={args.T} ---")
            results, best_c, best_mult = run_c_sweep(
                dataset, llm_labels, T=args.T, seeds=seeds_list
            )
            print(f"\nBest: c={best_c}, mult={best_mult:.3f}")

            out_path = f"v8_results_{args.model}_T{args.T}.json"
            output = {
                'model': args.model,
                'T': args.T,
                'llm_accuracy': float(cache['accuracy']),
                'diagnostics': dataset['diagnostics'],
                'c_sweep': results,
                'best_c': best_c,
                'best_mult': best_mult,
            }
            with open(out_path, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"Saved to {out_path}")
        else:
            print(f"\n--- Single run: {args.model}, T={args.T}, c={args.c} ---")
            for s in seeds_list:
                r = run_algorithm1(dataset, llm_labels, T=args.T, c=args.c, seed=s)
                print(f"  seed={s}: router={r['avg_router']:.4f} "
                      f"cold={r['avg_cold']:.4f} llm={r['avg_llm']:.4f} "
                      f"mult={r['multiplication']:.3f} llm%={r['llm_pct']:.1%}")

    elif args.mode == 'all':
        print("\n--- Running ALL models ---")
        all_results = {}
        models = ['flan-t5-small', 'flan-t5-base', 'flan-t5-large',
                  'qwen-0.5b', 'qwen-3b', 'qwen-7b']

        seeds_list = [42 + i for i in range(args.seeds)]

        for model_name in models:
            print(f"\n{'='*60}")
            print(f"  MODEL: {model_name}")
            print(f"{'='*60}")
            cache = predict_and_cache(dataset, model_name, args.batch_size)
            llm_labels = cache['labels']

            results, best_c, best_mult = run_c_sweep(
                dataset, llm_labels, T=args.T, seeds=seeds_list
            )
            all_results[model_name] = {
                'llm_accuracy': float(cache['accuracy']),
                'c_sweep': results,
                'best_c': best_c,
                'best_mult': best_mult,
            }

        out_path = f"v8_results_all_T{args.T}.json"
        output = {
            'T': args.T,
            'diagnostics': dataset['diagnostics'],
            'models': all_results,
        }
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nAll results saved to {out_path}")
