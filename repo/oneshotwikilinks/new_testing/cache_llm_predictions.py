"""
cache_llm_predictions.py

Pre-compute and cache LLM predictions for ALL dataset samples.
Run once per model; then experiment scripts load the cache
instead of running the LLM live.

This makes the bandit experiment pure numpy — no GPU needed,
finishes in minutes instead of hours. You can sweep hyperparameters
(c, alpha, lambda, passive_full) instantly.

Usage:
  # v1: Flan-T5 + bert-base-nli-mean-tokens (HuggingFace — T5 is encoder-decoder, no vLLM)
  python -u cache_llm_predictions.py --model_family v1 --llm_type small
  python -u cache_llm_predictions.py --model_family v1 --llm_type base
  python -u cache_llm_predictions.py --model_family v1 --llm_type large

  # v3: Qwen2.5 + bge-large-en-v1.5 (vLLM by default — much faster)
  python -u cache_llm_predictions.py --model_family v3 --llm_type small
  python -u cache_llm_predictions.py --model_family v3 --llm_type base
  python -u cache_llm_predictions.py --model_family v3 --llm_type large

  # v3 without vLLM (HuggingFace fallback)
  python -u cache_llm_predictions.py --model_family v3 --llm_type small --no_vllm

Output:
  llm_cache_v1_small.pkl.gz   (contains labels + sims for all N samples)
  llm_cache_v3_small.pkl.gz
  etc.
"""

import torch
import argparse
import os
import numpy as np
import gzip
import pickle
import torch.nn as nn
from sentence_transformers import SentenceTransformer


# ================================================================
# Model configurations
# ================================================================
V1_MODELS = {
    'small': 'google/flan-t5-small',
    'base':  'google/flan-t5-base',
    'large': 'google/flan-t5-large',
}
V1_ST_MODEL = 'bert-base-nli-mean-tokens'

V3_MODELS = {
    'small': 'Qwen/Qwen2.5-0.5B-Instruct',
    'base':  'Qwen/Qwen2.5-3B-Instruct',
    'large': 'Qwen/Qwen2.5-7B-Instruct',
}
V3_ST_MODEL = 'BAAI/bge-large-en-v1.5'


# ================================================================
# Dataset helpers (same as experiment scripts)
# ================================================================
class MyDataset(torch.utils.data.Dataset):
    def __init__(self): pass
    def __len__(self): return self.Xs.shape[0]
    def __getitem__(self, i):
        return self.Xs[i], self.ys[i], self.pre_texts[i], self.post_texts[i], self.text_entities[i]


def loadMyDataset(fname):
    print(f'Loading dataset: {fname}')
    with gzip.open(fname, 'rb') as f:
        return pickle.load(f)


def get_embd(dataset):
    d = {}
    for k in dataset.labelfeats:
        id_, embd = dataset.labelfeats[k]
        d[id_] = torch.FloatTensor(embd)
    return torch.stack([d[k] for k in sorted(d)])


# ================================================================
# v1: Flan-T5 predictions (HuggingFace only — T5 is encoder-decoder)
# ================================================================
def generate_v1_predictions(dataset, llm_type, device, batch_size=128):
    from transformers import T5Tokenizer, T5ForConditionalGeneration

    model_name = V1_MODELS[llm_type]
    print(f"Loading {model_name}...")
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()

    N = len(dataset)
    all_predictions = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        pre_texts = dataset.pre_texts[start:end]
        post_texts = dataset.post_texts[start:end]

        sentences = [pre + ' [MASK]. ' + post for pre, post in zip(pre_texts, post_texts)]
        texts = [f"question: {s.replace('[MASK]', '<extra_id_0>')}" for s in sentences]

        inp = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            out = model.generate(
                inp.input_ids.to(device),
                attention_mask=inp.attention_mask.to(device),
                max_new_tokens=20
            )

        preds = [
            tokenizer.decode(ids, skip_special_tokens=True).split('<extra_id_0>')[-1].strip()
            for ids in out
        ]
        all_predictions.extend(preds)

        batch_num = start // batch_size
        if batch_num % 100 == 0:
            print(f"  Processed {end}/{N} samples ({end/N*100:.1f}%)")

    del model
    torch.cuda.empty_cache()
    return all_predictions


# ================================================================
# v3: Qwen predictions via vLLM (fast)
# ================================================================
def generate_v3_predictions_vllm(dataset, llm_type):
    # vLLM uses multiprocessing internally. On Linux, Python defaults to 'fork'
    # which fails if CUDA is already initialized in the parent process.
    # Force 'spawn' to avoid this.
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    model_name = V3_MODELS[llm_type]
    print(f"Loading {model_name} with vLLM...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    llm = LLM(
        model=model_name,
        dtype="float16",
        trust_remote_code=True,
        gpu_memory_utilization=0.45,
        max_num_seqs=32,
        enforce_eager=True,  # skip CUDA graph warmup to save memory
    )
    params = SamplingParams(max_tokens=20, temperature=0)

    N = len(dataset)
    print(f"Building {N} prompts...")

    prompts = []
    for i in range(N):
        pre = dataset.pre_texts[i]
        post = dataset.post_texts[i]
        prompt = (
            f"What entity or person is being referred to in the blank? "
            f"Context: '{pre} ____ {post}' "
            f"Reply with ONLY the entity name, nothing else."
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer with only the entity name, no explanation."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text)

    print(f"Running vLLM inference on {N} prompts...")
    outputs = llm.generate(prompts, params)

    predictions = []
    for o in outputs:
        pred = o.outputs[0].text.strip()
        pred = pred.split('\n')[0].strip().strip('"').strip("'").rstrip('.')
        predictions.append(pred)

    del llm
    torch.cuda.empty_cache()
    return predictions


# ================================================================
# v3: Qwen predictions via HuggingFace (fallback, slower)
# ================================================================
def generate_v3_predictions_hf(dataset, llm_type, device, batch_size=128):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_name = V3_MODELS[llm_type]
    print(f"Loading {model_name} with HuggingFace...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, trust_remote_code=True
    ).to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    N = len(dataset)
    all_predictions = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        all_texts = []
        for i in range(start, end):
            pre = dataset.pre_texts[i]
            post = dataset.post_texts[i]
            prompt = (
                f"What entity or person is being referred to in the blank? "
                f"Context: '{pre} ____ {post}' "
                f"Reply with ONLY the entity name, nothing else."
            )
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Answer with only the entity name, no explanation."},
                {"role": "user", "content": prompt},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            all_texts.append(text)

        inputs = tokenizer(all_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        prompt_len = inputs['input_ids'].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j in range(output_ids.shape[0]):
            new_tokens = output_ids[j, prompt_len:]
            pred = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            pred = pred.split('\n')[0].strip().strip('"').strip("'").rstrip('.')
            all_predictions.append(pred)

        batch_num = start // batch_size
        if batch_num % 100 == 0:
            print(f"  Processed {end}/{N} samples ({end/N*100:.1f}%)")

    del model
    torch.cuda.empty_cache()
    return all_predictions


# ================================================================
# Encode predictions and compute cosine similarities
# ================================================================
def compute_similarities(predictions, entity_embd, st_model_name, device):
    print(f"Loading SentenceTransformer: {st_model_name}")
    st_model = SentenceTransformer(st_model_name, device=device)

    N = len(predictions)
    num_entities = entity_embd.shape[0]

    print(f"Encoding {N} predictions...")
    pred_embd = st_model.encode(predictions, batch_size=512, show_progress_bar=True)
    pred_embd = torch.FloatTensor(pred_embd).to(device)  # (N, d)

    entity_embd_dev = entity_embd.to(device)  # (num_entities, d)

    print(f"Computing cosine similarities ({N} x {num_entities})...")
    # Normalize for cosine similarity (matches nn.CosineSimilarity behavior)
    pred_norm = pred_embd / (pred_embd.norm(dim=1, keepdim=True) + 1e-6)
    ent_norm = entity_embd_dev / (entity_embd_dev.norm(dim=1, keepdim=True) + 1e-6)

    # Compute in chunks to avoid OOM on GPU
    chunk_size = 10000
    sims_list = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk_sims = pred_norm[start:end] @ ent_norm.T  # (chunk, num_entities)
        sims_list.append(chunk_sims.cpu().numpy())

    sims = np.concatenate(sims_list, axis=0).astype(np.float32)  # (N, num_entities)
    labels = np.argmax(sims, axis=1).astype(np.int64)            # (N,)

    del st_model, pred_embd, entity_embd_dev
    torch.cuda.empty_cache()

    return labels, sims


# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Cache LLM predictions for bandit experiments'
    )
    parser.add_argument('--model_family', type=str, required=True, choices=['v1', 'v3'])
    parser.add_argument('--llm_type', type=str, required=True, choices=['small', 'base', 'large'])
    parser.add_argument('--threshold', type=int, default=2000)
    parser.add_argument('--no_vllm', action='store_true',
                        help='Use HuggingFace instead of vLLM for v3 (slower)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for HuggingFace generation (lower = less GPU memory). '
                             'Try 8 or 16 for 7B models on tight GPUs.')
    parser.add_argument('--cuda', type=int, default=1)
    args = parser.parse_args()

    # Device
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load dataset
    if args.model_family == 'v1':
        dataset_fname = f'mydataset.{args.threshold}.pickle.gz'
        st_model_name = V1_ST_MODEL
    else:
        dataset_fname = f'mydataset.{args.threshold}.v3.pickle.gz'
        st_model_name = V3_ST_MODEL

    dataset = loadMyDataset(dataset_fname)
    print(f"Dataset size: {len(dataset)} samples")

    entity_embd = get_embd(dataset)
    print(f"Entity embeddings: {entity_embd.shape}")

    # Generate LLM predictions
    if args.model_family == 'v1':
        predictions = generate_v1_predictions(dataset, args.llm_type, device, batch_size=args.batch_size)
    elif not args.no_vllm:
        predictions = generate_v3_predictions_vllm(dataset, args.llm_type)
    else:
        predictions = generate_v3_predictions_hf(dataset, args.llm_type, device, batch_size=args.batch_size)

    print(f"Generated {len(predictions)} predictions")
    print(f"Sample predictions: {predictions[:5]}")

    # Compute cosine similarities against all entities
    labels, sims = compute_similarities(predictions, entity_embd, st_model_name, device)

    # Save cache
    cache = {
        'labels': labels,         # (N,) int64 — argmax entity per sample
        'sims': sims,             # (N, num_entities) float32 — cosine similarities
        'config': {
            'model_family': args.model_family,
            'llm_type': args.llm_type,
            'llm_model': (V1_MODELS if args.model_family == 'v1' else V3_MODELS)[args.llm_type],
            'st_model': st_model_name,
            'num_samples': len(predictions),
            'num_entities': entity_embd.shape[0],
        }
    }

    out_fname = f"llm_cache_{args.model_family}_{args.llm_type}.pkl.gz"
    print(f"Saving cache to {out_fname}...")
    with gzip.open(out_fname, 'wb') as f:
        pickle.dump(cache, f)

    # Quick stats
    print(f"Done! Cache saved: {out_fname}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Sims shape:   {sims.shape}")


if __name__ == '__main__':
    main()
