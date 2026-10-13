"""
run_text_experiment.py — LLM Cache + Algorithm 1 on Synthetic Text Dataset

Pipeline:
  1. Load synthetic_text_dataset_v3.pkl.gz (5200 docs, 52 categories, d=150)
  2. Run LLM on raw text → classify into 52 categories → cache predictions
  3. Compute Phase 2 diagnostic (subspace mismatch)
  4. Run Algorithm 1 (confidence-gated routing)

LLM Models:
  - Flan-T5: small (80M), base (250M), large (780M)  — run locally on Mac
  - Qwen2.5: 0.5B, 3B, 7B                            — run on GPU cluster

Usage:
  # Local Mac (all Flan-T5 models, sequentially):
  python -u run_text_experiment.py --all_local

  # Single model:
  python -u run_text_experiment.py --model flan-t5-small

  # GPU cluster (Qwen):
  python -u run_text_experiment.py --model qwen-0.5b
  python -u run_text_experiment.py --model qwen-3b
  python -u run_text_experiment.py --model qwen-7b

  # All Qwen on cluster:
  python -u run_text_experiment.py --all_cluster

  # Experiment only (if cache exists):
  python -u run_text_experiment.py --model flan-t5-small --skip_cache
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
    'flan-t5-small': {
        'hf_name': 'google/flan-t5-small',
        'type': 'seq2seq',
        'family': 'flan-t5',
    },
    'flan-t5-base': {
        'hf_name': 'google/flan-t5-base',
        'type': 'seq2seq',
        'family': 'flan-t5',
    },
    'flan-t5-large': {
        'hf_name': 'google/flan-t5-large',
        'type': 'seq2seq',
        'family': 'flan-t5',
    },
    'qwen-0.5b': {
        'hf_name': 'Qwen/Qwen2.5-0.5B-Instruct',
        'type': 'causal',
        'family': 'qwen',
    },
    'qwen-3b': {
        'hf_name': 'Qwen/Qwen2.5-3B-Instruct',
        'type': 'causal',
        'family': 'qwen',
    },
    'qwen-7b': {
        'hf_name': 'Qwen/Qwen2.5-7B-Instruct',
        'type': 'causal',
        'family': 'qwen',
    },
}


# ================================================================
# Dataset loading
# ================================================================
def load_dataset(path='synthetic_text_dataset_v3.pkl.gz'):
    print(f"Loading dataset from {path}...")
    with gzip.open(path, 'rb') as f:
        dataset = pickle.load(f)
    print(f"  Documents: {dataset['n_documents']}, Categories: {dataset['n_categories']}, "
          f"d={dataset['doc_features'].shape[1]}")
    return dataset


# ================================================================
# LLM prediction generation
# ================================================================
def build_prompt(text, category_names):
    cat_list = "\n".join(f"  {i+1}. {name}" for i, name in enumerate(category_names))
    return (
        f"Classify the following text into one of these {len(category_names)} categories.\n"
        f"Categories:\n{cat_list}\n\n"
        f"Text: \"{text}\"\n\n"
        f"Answer with ONLY the category name, nothing else."
    )


def generate_predictions_flan_t5(documents, category_names, model_name, batch_size=64):
    """Flan-T5: batched seq2seq, MPS-accelerated on Mac."""
    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    print(f"Loading {model_name}...")
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=True)
    model = T5ForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float32  # MPS needs float32
    )

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    print(f"  Device: {device}, batch_size: {batch_size}")

    # Pre-build all prompts
    all_prompts = [build_prompt(doc['text'], category_names) for doc in documents]
    predictions = []
    N = len(documents)

    pbar = tqdm(total=N, desc=f"  {model_name.split('/')[-1]}", unit="doc")
    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        batch_texts = all_prompts[batch_start:batch_end]

        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True,
                           truncation=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                num_beams=1,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)
        pbar.update(batch_end - batch_start)
    pbar.close()

    del model, tokenizer
    import gc; gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()

    return predictions


def generate_predictions_qwen(documents, category_names, model_name, batch_size=32):
    """Qwen: batched causal LM with left-padding for GPU throughput."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Enable left-padding for batched generation
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')

    load_kwargs = dict(dtype=dtype, trust_remote_code=True)
    if device.type == 'cuda':
        load_kwargs['device_map'] = 'auto'
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if device.type != 'cuda':
        model = model.to(device)
    model.eval()
    print(f"  Device: {device}, dtype: {dtype}, batch_size: {batch_size}")

    if device.type == 'cuda':
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU memory: {mem:.1f} GB")

    # Pre-build all prompts with chat template
    all_formatted = []
    for doc in documents:
        prompt_text = build_prompt(doc['text'], category_names)
        messages = [{"role": "user", "content": prompt_text}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        all_formatted.append(text)

    predictions = []
    N = len(documents)

    pbar = tqdm(total=N, desc=f"  {model_name.split('/')[-1]}", unit="doc")
    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        batch_texts = all_formatted[batch_start:batch_end]

        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True,
                           truncation=True, max_length=1024).to(device)
        prompt_len = inputs['input_ids'].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j in range(outputs.shape[0]):
            generated = outputs[j][prompt_len:]
            decoded = tokenizer.decode(generated, skip_special_tokens=True)
            predictions.append(decoded)

        pbar.update(batch_end - batch_start)
    pbar.close()

    del model, tokenizer
    import gc; gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return predictions


def match_prediction_to_category(pred_text, category_names):
    """Match LLM output text to the closest category name."""
    pred_lower = pred_text.strip().lower()

    # Exact match
    for i, name in enumerate(category_names):
        if name.lower() == pred_lower:
            return i
    # Category in prediction
    for i, name in enumerate(category_names):
        if name.lower() in pred_lower:
            return i
    # Prediction in category
    for i, name in enumerate(category_names):
        if pred_lower in name.lower():
            return i
    # Word overlap
    pred_words = set(pred_lower.split())
    best_score, best_idx = 0, 0
    for i, name in enumerate(category_names):
        overlap = len(pred_words & set(name.lower().split()))
        if overlap > best_score:
            best_score, best_idx = overlap, i
    return best_idx


def generate_and_cache_predictions(dataset, model_key, batch_size=64):
    """Generate LLM predictions and cache them."""
    config = MODEL_CONFIGS[model_key]
    documents = dataset['documents']
    category_names = dataset['category_names']

    cache_path = f"llm_cache_{model_key}.pkl.gz"
    if os.path.exists(cache_path):
        print(f"Cache exists: {cache_path}, loading...")
        with gzip.open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        print(f"  Accuracy: {cache['accuracy']:.4f}, Time: {cache['generation_time']:.1f}s")
        return cache

    print(f"\n{'='*60}")
    print(f"Generating predictions: {model_key} ({config['hf_name']})")
    print(f"  {len(documents)} documents -> {len(category_names)} categories")
    print(f"{'='*60}")

    t0 = time.time()
    if config['type'] == 'seq2seq':
        raw_predictions = generate_predictions_flan_t5(
            documents, category_names, config['hf_name'], batch_size=batch_size
        )
    else:
        raw_predictions = generate_predictions_qwen(
            documents, category_names, config['hf_name'], batch_size=batch_size
        )
    gen_time = time.time() - t0
    print(f"  Generation done: {gen_time:.1f}s ({len(documents)/gen_time:.1f} docs/s)")

    # Match raw text to category indices
    labels = np.array([match_prediction_to_category(p, category_names)
                       for p in raw_predictions])

    true_labels = dataset['ys']
    accuracy = np.mean(labels == true_labels)
    print(f"  LLM accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Samples
    for i in range(min(5, len(documents))):
        true_cat = category_names[true_labels[i]]
        pred_cat = category_names[labels[i]]
        m = "+" if labels[i] == true_labels[i] else "-"
        print(f"    {m} True: {true_cat:<30s} Pred: {pred_cat:<30s} Raw: '{raw_predictions[i][:40]}'")

    cache = {
        'model_key': model_key,
        'model_config': config,
        'labels': labels,
        'raw_predictions': raw_predictions,
        'accuracy': accuracy,
        'generation_time': gen_time,
        'n_samples': len(documents),
    }
    with gzip.open(cache_path, 'wb') as f:
        pickle.dump(cache, f)
    print(f"  Cache saved: {cache_path}")
    return cache


# ================================================================
# Diagnostic 4: Subspace Mismatch
# ================================================================
def compute_diagnostic4(dataset, llm_labels):
    doc_features = dataset['doc_features']
    arm_features = dataset['arm_features']
    d = doc_features.shape[1]
    N = doc_features.shape[0]

    cov_ctx = (doc_features.T @ doc_features) / N
    llm_features = doc_features * arm_features[llm_labels]
    cov_llm = (llm_features.T @ llm_features) / N

    eigvals_ctx = np.linalg.eigvalsh(cov_ctx)
    eigvals_ctx = eigvals_ctx[eigvals_ctx > 1e-10]
    r_hat = eigvals_ctx.sum() / eigvals_ctx.max()
    r = max(1, min(int(round(r_hat)), d - 1))

    _, _, Vh_llm = svd(cov_llm)
    _, _, Vh_ctx = svd(cov_ctx)
    U_llm = Vh_llm[:r, :].T
    U_ctx = Vh_ctx[:r, :].T

    angles_rad = subspace_angles(U_llm, U_ctx)
    mean_angle = float(np.mean(np.degrees(angles_rad)))
    return mean_angle, r


# ================================================================
# Algorithm 1: Confidence-Gated Routing (vectorized)
# ================================================================
def sm_update(V_inv, x):
    Vx = V_inv @ x
    denom = 1.0 + x @ Vx
    if abs(denom) < 1e-12:
        return V_inv
    return V_inv - np.outer(Vx, Vx) / denom


def run_algorithm1(dataset, llm_labels, T=2000, c=100.0, lam=0.1,
                   alpha=1.0, alpha_decay=0.9999, seed=42):
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

    pbar = tqdm(range(T), desc="  Algorithm 1", unit="step")
    for t_i in pbar:
        idx = perm[t_i]
        x = doc_features[idx]
        true_y = true_labels[idx]
        llm_choice = int(llm_labels[idx])
        current_alpha = max(0.1, alpha * (alpha_decay ** t_i))

        # LLM reward
        r_llm = 1.0 if llm_choice == true_y else 0.0
        cum_llm += r_llm

        # Cold LinUCB — vectorized over arms
        theta_cold = V_inv_cold @ b_cold
        phi_all = x[None, :] * arm_features  # (K, d)
        scores_cold = phi_all @ theta_cold    # (K,)
        # UCB: sqrt(phi_a^T V_inv phi_a) for each arm
        V_phi = phi_all @ V_inv_cold          # (K, d)
        var_cold = np.sum(V_phi * phi_all, axis=1)  # (K,)
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

        # Progress bar update
        if (t_i + 1) % 100 == 0:
            pbar.set_postfix({
                'R': f"{cum_router/(t_i+1):.3f}",
                'L': f"{cum_llm/(t_i+1):.3f}",
                'C': f"{cum_cold/(t_i+1):.3f}",
                'L%': f"{llm_count/(t_i+1)*100:.0f}%",
            })

        if (t_i + 1) % 200 == 0 or t_i == T - 1:
            history['step'].append(t_i + 1)
            history['router'].append(cum_router / (t_i + 1))
            history['llm'].append(cum_llm / (t_i + 1))
            history['cold'].append(cum_cold / (t_i + 1))
            history['llm_frac'].append(llm_count / (t_i + 1))
    pbar.close()

    return {
        'avg_router': cum_router / T,
        'avg_llm': cum_llm / T,
        'avg_cold': cum_cold / T,
        'llm_pct': llm_count / T * 100,
        'T': T,
        'history': history,
    }


# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser(description='Synthetic Text LLM Experiment')
    parser.add_argument('--model', type=str, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--all_local', action='store_true',
                        help='Run all Flan-T5 models (for Mac)')
    parser.add_argument('--all_cluster', action='store_true',
                        help='Run all Qwen models (for GPU cluster)')
    parser.add_argument('--skip_cache', action='store_true')
    parser.add_argument('--T', type=int, default=2000)
    parser.add_argument('--c', type=float, default=100.0)
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (64 for Flan-T5 on Mac, 32 for Qwen on GPU)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.all_local:
        models_to_run = ['flan-t5-small', 'flan-t5-base', 'flan-t5-large']
    elif args.all_cluster:
        models_to_run = ['qwen-0.5b', 'qwen-3b', 'qwen-7b']
    elif args.model:
        models_to_run = [args.model]
    else:
        parser.error("Specify --model, --all_local, or --all_cluster")

    dataset = load_dataset()

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {models_to_run}")
    print(f"  T={args.T}, c={args.c}, batch_size={args.batch_size}, seed={args.seed}")
    print(f"{'='*70}")

    all_results = {}

    for mi, model_key in enumerate(models_to_run):
        print(f"\n{'#'*70}")
        print(f"# [{mi+1}/{len(models_to_run)}] {model_key}")
        print(f"{'#'*70}")

        # Step 1: Cache
        if args.skip_cache:
            cache_path = f"llm_cache_{model_key}.pkl.gz"
            if not os.path.exists(cache_path):
                print(f"ERROR: Cache not found: {cache_path}")
                continue
            with gzip.open(cache_path, 'rb') as f:
                cache = pickle.load(f)
        else:
            cache = generate_and_cache_predictions(
                dataset, model_key, batch_size=args.batch_size
            )

        llm_labels = cache['labels']
        accuracy = cache['accuracy']
        print(f"\n  LLM accuracy: {accuracy*100:.2f}%")

        # Step 2: Diagnostic 4
        mismatch_angle, r = compute_diagnostic4(dataset, llm_labels)
        print(f"  Diagnostic 4: mismatch={mismatch_angle:.2f} deg (r={r})")

        # Step 3: Algorithm 1
        print(f"  Running Algorithm 1 (T={args.T})...")
        results = run_algorithm1(dataset, llm_labels, T=args.T, c=args.c, seed=args.seed)

        mult = results['avg_router'] / max(results['avg_llm'], results['avg_cold'], 1e-8)
        print(f"\n  {'='*50}")
        print(f"  RESULTS: {model_key}")
        print(f"  {'='*50}")
        print(f"    Router:    {results['avg_router']:.4f}")
        print(f"    LLM-only:  {results['avg_llm']:.4f}")
        print(f"    Cold:      {results['avg_cold']:.4f}")
        print(f"    LLM%:      {results['llm_pct']:.1f}%")
        print(f"    Multiply:  {mult:.4f} {'***' if mult > 1.0 else ''}")

        all_results[model_key] = {
            'accuracy': float(accuracy),
            'mismatch_angle': float(mismatch_angle),
            'subspace_dim_r': int(r),
            'avg_router': results['avg_router'],
            'avg_llm': results['avg_llm'],
            'avg_cold': results['avg_cold'],
            'llm_pct': results['llm_pct'],
            'multiplication': float(mult),
            'history': results['history'],
        }

        # Save incrementally (don't lose results if later model crashes)
        out_path = f"results_synthetic_text_c{args.c}.json"
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2)

    # Final summary
    print(f"\n\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<16} {'Acc%':<8} {'Angle':<8} {'Router':<8} {'LLM':<8} {'Cold':<8} {'Mult':<6}")
    print("-" * 62)
    for mk, res in all_results.items():
        print(f"{mk:<16} {res['accuracy']*100:<8.2f} {res['mismatch_angle']:<8.1f} "
              f"{res['avg_router']:<8.4f} {res['avg_llm']:<8.4f} {res['avg_cold']:<8.4f} "
              f"{res['multiplication']:<6.3f}")

    out_path = f"results_synthetic_text_c{args.c}.json"
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
