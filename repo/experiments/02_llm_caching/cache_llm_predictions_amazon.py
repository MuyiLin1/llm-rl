"""
cache_llm_predictions_amazon.py

Pre-compute and cache Flan-T5 predictions for ALL AmazonCat-13K samples.
Same idea as cache_llm_predictions.py for WikiLinks, but adapted for
AmazonCat's prompt format and dataset structure.

The AmazonCat LLM approach (from corral_clip.py / Chen et al.):
  1. Prompt: "Title: {title}\nContent: {content}\nTask: Predict the associated label."
  2. Flan-T5 generates a text label
  3. Encode the predicted label with SentenceTransformer
  4. Cosine similarity against all 13,330 entity embeddings
  5. Argmax = predicted entity

Output:
  llm_cache_amazon_small.pkl.gz  — labels (1.18M,) + sims (1.18M, 13330)

NOTE: The sims matrix will be ~60GB in float32 (1.18M x 13330 x 4 bytes).
      We store sims in float16 to halve this (~30GB), which is still large.
      If memory is tight, we can store only top-K similarities instead.

Usage:
  # Run all 3 Flan-T5 sizes sequentially (recommended):
  python -u cache_llm_predictions_amazon.py --llm_type all

  # Run a single size:
  python -u cache_llm_predictions_amazon.py --llm_type small

  # Store only top-K sims per sample (default=100, saves memory):
  python -u cache_llm_predictions_amazon.py --llm_type all --top_k 100
"""

import torch
import argparse
import os
import sys
import numpy as np
import gzip
import pickle
import gc
import time
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
# Dataset loading
# ================================================================
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs): pass
    def __len__(self): return self.Xs.shape[0]
    def __getitem__(self, index):
        return self.Xs[index], self.text_titles[index], self.text_contents[index], self.ys[index]

sys.modules['__main__'].MyDataset = MyDataset


def loadMyDataset(model_family='v1'):
    if model_family == 'v3':
        fname = 'mydataset.0.v3.pickle.gz'
    else:
        fname = 'mydataset.0.pickle.gz'
    print(f'Loading dataset: {fname}')
    with gzip.open(fname, 'rb') as f:
        return pickle.load(f)


def get_entity_embeddings(dataset):
    """Extract entity embedding matrix sorted by ordinal."""
    d = {}
    for k in dataset.labelfeats:
        idx, embd = dataset.labelfeats[k]
        d[idx] = torch.FloatTensor(embd)
    return torch.stack([d[i] for i in sorted(d.keys())])


# ================================================================
# v1: Flan-T5 prediction (matches corral_clip.py prompt format exactly)
# ================================================================
def generate_v1_predictions(dataset, llm_type, device, batch_size=32):
    from transformers import T5Tokenizer, T5ForConditionalGeneration

    model_name = V1_MODELS[llm_type]
    print(f"Loading {model_name}...")
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()

    N = len(dataset)
    all_predictions = []
    t0 = time.time()

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        titles = dataset.text_titles[start:end]
        contents = dataset.text_contents[start:end]

        # Same prompt as corral_clip.py predict_labels()
        input_texts = [
            f"Title: {title}\nContent: {content}\nTask: Predict the associated label."
            for title, content in zip(titles, contents)
        ]

        inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            output_ids = model.generate(
                inputs.input_ids.to(device),
                attention_mask=inputs.attention_mask.to(device),
            )

        preds = [
            tokenizer.decode(ids, skip_special_tokens=True)
            for ids in output_ids
        ]
        all_predictions.extend(preds)

        batch_num = start // batch_size
        if batch_num % 500 == 0:
            elapsed = time.time() - t0
            rate = end / max(elapsed, 1)
            eta = (N - end) / max(rate, 1)
            print(f"  [{end:>8d}/{N}] ({end/N*100:.1f}%) | {elapsed:.0f}s elapsed | ETA {eta:.0f}s")

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()

    return all_predictions


# ================================================================
# v3: Qwen prediction via vLLM (fast, needs GPU)
# ================================================================
def generate_v3_predictions_vllm(dataset, llm_type):
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
        enforce_eager=True,
    )
    params = SamplingParams(max_tokens=20, temperature=0)

    N = len(dataset)
    print(f"Building {N} prompts...")

    prompts = []
    for i in range(N):
        title = dataset.text_titles[i]
        content = dataset.text_contents[i]
        prompt = (
            f"Title: {title}\n"
            f"Content: {content}\n"
            f"Task: Predict the associated label. "
            f"Reply with ONLY the label name, nothing else."
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer with only the label name, no explanation."},
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
# v3: Qwen prediction via HuggingFace (fallback, slower)
# ================================================================
def generate_v3_predictions_hf(dataset, llm_type, device, batch_size=32):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_name = V3_MODELS[llm_type]
    print(f"Loading {model_name} with HuggingFace...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    N = len(dataset)
    all_predictions = []
    t0 = time.time()

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        all_texts = []
        for i in range(start, end):
            title = dataset.text_titles[i]
            content = dataset.text_contents[i]
            prompt = (
                f"Title: {title}\n"
                f"Content: {content}\n"
                f"Task: Predict the associated label. "
                f"Reply with ONLY the label name, nothing else."
            )
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Answer with only the label name, no explanation."},
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
        if batch_num % 500 == 0:
            elapsed = time.time() - t0
            rate = end / max(elapsed, 1)
            eta = (N - end) / max(rate, 1)
            print(f"  [{end:>8d}/{N}] ({end/N*100:.1f}%) | {elapsed:.0f}s elapsed | ETA {eta:.0f}s")

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()

    return all_predictions


# ================================================================
# Encode predictions and compute cosine similarities
# ================================================================
def compute_similarities(predictions, entity_embd, st_model_name, device, top_k=None):
    """
    Encode LLM text predictions with SentenceTransformer, then compute
    cosine similarity against all entity embeddings.

    If top_k is set, only store the top-K similarities and their indices
    per sample (saves massive memory for 13K entities).
    """
    print(f"Loading SentenceTransformer: {st_model_name}")
    st_model = SentenceTransformer(st_model_name, device=str(device))

    N = len(predictions)
    num_entities = entity_embd.shape[0]

    # Encode predictions in batches
    print(f"Encoding {N} predictions...")
    pred_embd = st_model.encode(predictions, batch_size=512, show_progress_bar=True)
    pred_embd = torch.FloatTensor(pred_embd)  # keep on CPU to save GPU mem

    # Normalize
    pred_norm = pred_embd / (pred_embd.norm(dim=1, keepdim=True) + 1e-6)
    ent_norm = entity_embd / (entity_embd.norm(dim=1, keepdim=True) + 1e-6)

    del st_model
    gc.collect()

    # Compute similarities in chunks (entity matrix is large: 13330 x 768)
    chunk_size = 5000
    print(f"Computing cosine similarities ({N} x {num_entities})...")

    if top_k is not None:
        # Store only top-K per sample
        print(f"  Storing top-{top_k} similarities per sample")
        all_labels = np.zeros(N, dtype=np.int64)
        all_top_sims = np.zeros((N, top_k), dtype=np.float16)
        all_top_indices = np.zeros((N, top_k), dtype=np.int32)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk_sims = pred_norm[start:end] @ ent_norm.T  # (chunk, num_entities)

            # Argmax for labels
            all_labels[start:end] = chunk_sims.argmax(dim=1).numpy()

            # Top-K
            topk_vals, topk_idx = torch.topk(chunk_sims, top_k, dim=1)
            all_top_sims[start:end] = topk_vals.numpy().astype(np.float16)
            all_top_indices[start:end] = topk_idx.numpy().astype(np.int32)

            if (start // chunk_size) % 50 == 0:
                print(f"    [{end:>8d}/{N}]")

        return all_labels, all_top_sims, all_top_indices
    else:
        # Store full similarity matrix (WARNING: ~30GB for 1.18M x 13330 in float16)
        all_sims = np.zeros((N, num_entities), dtype=np.float16)
        all_labels = np.zeros(N, dtype=np.int64)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk_sims = pred_norm[start:end] @ ent_norm.T  # (chunk, num_entities)
            all_sims[start:end] = chunk_sims.numpy().astype(np.float16)
            all_labels[start:end] = chunk_sims.argmax(dim=1).numpy()

            if (start // chunk_size) % 50 == 0:
                print(f"    [{end:>8d}/{N}]")

        return all_labels, all_sims, None


# ================================================================
# Run one LLM type
# ================================================================
def run_one(dataset, entity_embd, model_family, llm_type, device,
            batch_size=32, top_k=100, no_vllm=False):
    """Generate predictions and cache for a single LLM size."""
    if model_family == 'v1':
        models_dict = V1_MODELS
        st_model_name = V1_ST_MODEL
        predictions = generate_v1_predictions(dataset, llm_type, device,
                                              batch_size=batch_size)
    else:
        models_dict = V3_MODELS
        st_model_name = V3_ST_MODEL
        if no_vllm:
            predictions = generate_v3_predictions_hf(dataset, llm_type, device,
                                                     batch_size=batch_size)
        else:
            predictions = generate_v3_predictions_vllm(dataset, llm_type)

    print(f"Generated {len(predictions)} predictions")
    print(f"Sample predictions: {predictions[:5]}")

    # Compute cosine similarities
    labels, sims_or_top_sims, top_indices = compute_similarities(
        predictions, entity_embd, st_model_name, device, top_k=top_k
    )

    # Build cache
    family_tag = model_family  # 'v1' or 'v3'
    if top_k is not None:
        cache = {
            'labels': labels,
            'top_sims': sims_or_top_sims,      # (N, top_k) float16
            'top_indices': top_indices,          # (N, top_k) int32
            'top_k': top_k,
            'config': {
                'dataset': 'amazoncat13k',
                'model_family': model_family,
                'llm_type': llm_type,
                'llm_model': models_dict[llm_type],
                'st_model': st_model_name,
                'num_samples': len(predictions),
                'num_entities': entity_embd.shape[0],
                'storage': f'top-{top_k}',
            }
        }
    else:
        cache = {
            'labels': labels,
            'sims': sims_or_top_sims,           # (N, num_entities) float16
            'config': {
                'dataset': 'amazoncat13k',
                'model_family': model_family,
                'llm_type': llm_type,
                'llm_model': models_dict[llm_type],
                'st_model': st_model_name,
                'num_samples': len(predictions),
                'num_entities': entity_embd.shape[0],
                'storage': 'full',
            }
        }

    out_fname = f"llm_cache_amazon_{family_tag}_{llm_type}.pkl.gz"
    print(f"Saving cache to {out_fname}...")
    with gzip.open(out_fname, 'wb') as f:
        pickle.dump(cache, f)

    print(f"Done! Cache saved: {out_fname}")
    print(f"  Labels shape: {labels.shape}")
    if top_k is not None:
        print(f"  Top sims shape: {sims_or_top_sims.shape}")
        print(f"  Top indices shape: {top_indices.shape}")
    else:
        print(f"  Sims shape: {sims_or_top_sims.shape}")

    # Quick accuracy check
    correct = 0
    for i in range(len(labels)):
        if labels[i] in dataset.ys[i]:
            correct += 1
    print(f"  LLM accuracy (Precision@1): {correct/len(labels)*100:.2f}%")


# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Cache LLM predictions for AmazonCat-13K'
    )
    parser.add_argument('--model_family', type=str, required=True,
                        choices=['v1', 'v3'],
                        help='v1 = Flan-T5 + bert-base-nli-mean-tokens, '
                             'v3 = Qwen + bge-large-en-v1.5')
    parser.add_argument('--llm_type', type=str, default='all',
                        choices=['small', 'base', 'large', 'all'],
                        help='Which size to cache. "all" runs small, base, large sequentially.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for generation')
    parser.add_argument('--top_k', type=int, default=100,
                        help='Store only top-K sims per sample (saves memory). '
                             'Set to 0 to store full similarity matrix.')
    parser.add_argument('--no_vllm', action='store_true',
                        help='Use HuggingFace instead of vLLM for v3 (slower)')
    parser.add_argument('--cuda', type=int, default=1)
    args = parser.parse_args()

    if args.top_k == 0:
        args.top_k = None

    # Device
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load dataset (shared across all LLM sizes)
    dataset = loadMyDataset(args.model_family)
    print(f"Dataset size: {len(dataset)} samples")

    entity_embd = get_entity_embeddings(dataset)
    print(f"Entity embeddings: {entity_embd.shape}")

    family_label = 'Flan-T5' if args.model_family == 'v1' else 'Qwen'

    # Determine which LLM sizes to run
    if args.llm_type == 'all':
        llm_types = ['small', 'base', 'large']
    else:
        llm_types = [args.llm_type]

    for llm_type in llm_types:
        print(f"\n{'='*70}")
        print(f"  CACHING: {family_label}-{llm_type}")
        print(f"{'='*70}\n")

        run_one(dataset, entity_embd, args.model_family, llm_type, device,
                batch_size=args.batch_size, top_k=args.top_k,
                no_vllm=args.no_vllm)

        # Free GPU memory between model sizes
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()

    print("\nAll done!")


if __name__ == '__main__':
    main()
