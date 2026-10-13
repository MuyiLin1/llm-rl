"""
experiment_algorithm1_v2.py

Same as experiment_algorithm1.py but with upgraded SentenceTransformer.
Uses 'all-mpnet-base-v2' (768-dim) instead of 'bert-base-nli-mean-tokens'.

Key changes from v1:
  1. Loads mydataset.2000.v2.pickle.gz, which was built with all-mpnet-base-v2
     for BOTH entity embeddings AND context embeddings (Xs).
  2. Entity embeddings come from the v2 pickle via get_embd() — same space as Xs.
  3. LLM prediction text is encoded at runtime with all-mpnet-base-v2 for
     cosine-similarity matching against the v2 entity embeddings.
  4. Output files are tagged with '_v2' to distinguish from v1 results.
"""

import torch
import argparse
import numpy as np
import json
import gzip
import pickle
import gc
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer


# New sentence transformer model
ST_MODEL = 'all-mpnet-base-v2'  # 768-dim, much better than bert-base-nli-mean-tokens


# ================================================================
# Sherman-Morrison rank-1 inverse update: O(d^2)
# ================================================================
def sm_update(V_inv, x):
    Vx = V_inv @ x
    return V_inv - np.outer(Vx, Vx) / (1.0 + x @ Vx)


# ================================================================
# Helpers
# ================================================================
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


class MyDataset(torch.utils.data.Dataset):
    def __init__(self): pass
    def __len__(self): return self.Xs.shape[0]
    def __getitem__(self, i):
        return self.Xs[i], self.ys[i], self.pre_texts[i], self.post_texts[i], self.text_entities[i]


def loadMyDataset(threshold):
    fname = f'mydataset.{threshold}.v2.pickle.gz'
    print(f'Loading dataset: {fname}')
    with gzip.open(fname, 'rb') as f:
        return pickle.load(f)


def load_model_and_tokenizer(model_name, device):
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    return model, tokenizer


def generate_mask_fillings(sentences, model, tokenizer, device):
    texts = [f"question: {s.replace('[MASK]', '<extra_id_0>')}" for s in sentences]
    inp = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    out = model.generate(
        inp.input_ids.to(device),
        attention_mask=inp.attention_mask.to(device),
        max_new_tokens=20
    )
    return [
        tokenizer.decode(ids, skip_special_tokens=True).split('<extra_id_0>')[-1].strip()
        for ids in out
    ]


def language_model_scores(pre_texts, post_texts, t5_tokenizer, t5_model,
                           entity_model, entity_embd, num_entities, cos, device):
    """
    Returns:
      labels : (batch_size,) — LLM's argmax predictions
      sims   : (batch_size, num_entities) — cosine similarities
    """
    sentences = [pre + ' [MASK]. ' + post for pre, post in zip(pre_texts, post_texts)]
    predicted = generate_mask_fillings(sentences, t5_model, t5_tokenizer, device)
    pred_embd = torch.FloatTensor(entity_model.encode(predicted)).to(device)
    pred_embd = pred_embd.unsqueeze(1).repeat(1, num_entities, 1)
    ent_exp = entity_embd.unsqueeze(0).repeat(len(pre_texts), 1, 1).to(device)
    sims = cos(pred_embd, ent_exp)                           # (batch, num_entities)
    labels = torch.argmax(sims, dim=-1).detach()             # (batch,)
    return labels, sims.detach().cpu().numpy()


def get_embd(dataset):
    """Extract entity embeddings from the pickle, sorted by id."""
    d = {}
    for k in dataset.labelfeats:
        id_, embd = dataset.labelfeats[k]
        d[id_] = torch.FloatTensor(embd)
    return torch.stack([d[k] for k in sorted(d)])


# ================================================================
# Main experiment loop
# ================================================================
def learnOnline(dataset, batch_size, cuda, seed, llm_type,
                linucb_alpha=1.0, lam=0.1, alpha_decay=0.9985,
                llm_confidence=100.0, passive_full=False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Entity embeddings from the v2 pickle (already all-mpnet-base-v2)
    entity_embd = get_embd(dataset)
    num_entities = entity_embd.shape[0]

    if cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"LLM: flan-t5-{llm_type} | alpha={linucb_alpha} | lam={lam}")
    print(f"LLM confidence (c): {llm_confidence}")
    print(f"Passive update mode: {'full (V + b)' if passive_full else 'covariance only (V)'}")
    print(f"Num actions: {num_entities}")
    print(f"SentenceTransformer: {ST_MODEL}")

    t5_model, t5_tokenizer = load_model_and_tokenizer(f"google/flan-t5-{llm_type}", device)

    # Use the new ST model only for encoding LLM prediction text
    entity_model = SentenceTransformer(ST_MODEL, device=device)
    cos = nn.CosineSimilarity(dim=2, eps=1e-6)

    print(f"Entity embeddings shape: {entity_embd.shape}")

    # Dedicated generator so shuffle order is identical across LLM sizes
    data_gen = torch.Generator()
    data_gen.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
        generator=data_gen
    )

    action_features = entity_embd.cpu().numpy()  # (num_actions, 768)
    d = action_features.shape[1]                 # 768

    # mpnet produces L2-normalized embeddings (||e||=1), so element-wise
    # products have ||phi|| ≈ 1/sqrt(d) ≈ 0.036.  This is too small for
    # lam=0.1 regularization — V_inv barely shrinks.  Scale features so
    # ||phi|| ≈ 1, matching the scale of bert-nli embeddings in v1.
    feat_scale = np.sqrt(d)  # sqrt(768) ≈ 27.7
    action_features = action_features * feat_scale
    print(f"Feature dim: {d}  (scaled by sqrt(d) = {feat_scale:.1f})")

    # --- Warm LinUCB (receives passive covariance updates from LLM rounds) ---
    V_inv_warm = (1.0 / lam) * np.eye(d)
    b_warm = np.zeros(d)

    # --- Cold LinUCB (independent baseline, no help from LLM) ---
    V_inv_cold = (1.0 / lam) * np.eye(d)
    b_cold = np.zeros(d)

    inv_sqrt_c = 1.0 / np.sqrt(llm_confidence)

    # --- Accumulators ---
    acc_router = EasyAcc()
    acc_llm    = EasyAcc()
    acc_cold   = EasyAcc()

    # --- Routing diagnostics ---
    llm_rounds   = 0
    total_rounds = 0

    history = {
        'batch': [], 'hybrid': [], 'llm': [], 'linucb_cold': [],
        'llm_fraction': [],
        'config': {
            'algorithm': 'confidence_gated_routing_v2',
            'features': 'elementwise_768d',
            'sentence_transformer': ST_MODEL,
            'linucb_alpha': linucb_alpha, 'lam': lam,
            'llm_type': llm_type, 'alpha_decay': alpha_decay,
            'llm_confidence': llm_confidence,
            'passive_full': passive_full,
        }
    }

    tag = f"algo1v2_{llm_type}_c{llm_confidence}"
    if passive_full:
        tag += "_pfull"
    fname = f"results_{tag}.json"
    print(f"Results file: {fname}")
    print(f"Total batches: {len(loader)}")
    print("=" * 90)

    for bno, (Xs, ys, pre_texts, post_texts, text_entities) in enumerate(loader):

        current_alpha = max(0.1, linucb_alpha * (alpha_decay ** bno))
        Xs, ys = Xs.to(device), ys.to(device)

        with torch.no_grad():
            llm_labels, llm_sims = language_model_scores(
                pre_texts, post_texts, t5_tokenizer, t5_model,
                entity_model, entity_embd, num_entities, cos, device
            )

        for i in range(len(Xs)):
            context_x   = Xs[i].cpu().numpy()
            d_ctx       = context_x.shape[0] // 2
            cx          = context_x[:d_ctx]            # pre-text embedding (768-dim)
            true_label  = ys[i].item()
            llm_choice  = llm_labels[i].item()

            # Base features: element-wise product (K, 768)
            joint_base = action_features * cx

            total_rounds += 1

            # ===========================================================
            # TRACK 1: LLM only (static baseline)
            # ===========================================================
            r_llm = 1.0 if llm_choice == true_label else 0.0
            acc_llm += r_llm

            # ===========================================================
            # TRACK 2: Cold LinUCB (no warm-start, acts every round)
            # ===========================================================
            theta_cold = V_inv_cold @ b_cold
            Vf_cold    = joint_base @ V_inv_cold
            vars_cold  = np.sum(Vf_cold * joint_base, axis=1)
            ucb_cold   = joint_base @ theta_cold + current_alpha * np.sqrt(np.maximum(vars_cold, 0))
            cold_choice = int(np.argmax(ucb_cold))
            r_cold = 1.0 if cold_choice == true_label else 0.0
            acc_cold += r_cold

            x_cold = joint_base[cold_choice]
            V_inv_cold = sm_update(V_inv_cold, x_cold)
            b_cold += r_cold * x_cold

            # ===========================================================
            # TRACK 3: Confidence-Gated Router (Algorithm 1)
            # ===========================================================
            theta_warm = V_inv_warm @ b_warm
            Vf_warm    = joint_base @ V_inv_warm
            vars_warm  = np.sum(Vf_warm * joint_base, axis=1)
            ucb_warm   = joint_base @ theta_warm + current_alpha * np.sqrt(np.maximum(vars_warm, 0))
            linucb_choice = int(np.argmax(ucb_warm))

            x_llm    = joint_base[llm_choice]
            x_linucb = joint_base[linucb_choice]

            u_llm = inv_sqrt_c * np.linalg.norm(x_llm)
            u_linucb = np.sqrt(max(0.0, x_linucb @ V_inv_warm @ x_linucb))

            if u_llm < u_linucb:
                router_choice = llm_choice
                r_router = 1.0 if router_choice == true_label else 0.0
                llm_rounds += 1

                V_inv_warm = sm_update(V_inv_warm, x_llm)
                if passive_full:
                    b_warm += r_router * x_llm
            else:
                router_choice = linucb_choice
                r_router = 1.0 if router_choice == true_label else 0.0

                V_inv_warm = sm_update(V_inv_warm, x_linucb)
                b_warm += r_router * x_linucb

            acc_router += r_router

        # ===========================================================
        # Logging every 10 batches
        # ===========================================================
        if bno % 10 == 0:
            llm_frac = llm_rounds / max(total_rounds, 1)
            history['batch'].append(bno)
            history['hybrid'].append(acc_router.mean())
            history['llm'].append(acc_llm.mean())
            history['linucb_cold'].append(acc_cold.mean())
            history['llm_fraction'].append(llm_frac)

            print(
                f"Batch {bno:4d} | "
                f"Router: {acc_router.mean():.4f} | "
                f"LLM only: {acc_llm.mean():.4f} | "
                f"Cold LinUCB: {acc_cold.mean():.4f} | "
                f"LLM routing: {llm_frac:.1%}"
            )

            with open(fname, "w") as f:
                json.dump(history, f)

        del Xs, ys, pre_texts, post_texts, text_entities, llm_labels, llm_sims
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()

    print("=" * 90)
    print(
        f"FINAL | Router: {acc_router.mean():.4f} | "
        f"LLM only: {acc_llm.mean():.4f} | "
        f"Cold LinUCB: {acc_cold.mean():.4f}"
    )
    print(f"LLM routing: {llm_rounds}/{total_rounds} = {llm_rounds/max(total_rounds,1):.1%}")

    with open(fname, "w") as f:
        json.dump(history, f)
    print(f"Results saved to: {fname}")
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Algorithm 1 v2: With upgraded SentenceTransformer'
    )
    parser.add_argument('--seed',           type=int,   default=1)
    parser.add_argument('--llm_type',       type=str,   default='small',
                        choices=['small', 'base', 'large'],
                        help='LLM size.')
    parser.add_argument('--linucb_alpha',   type=float, default=1.0)
    parser.add_argument('--lam',            type=float, default=0.1)
    parser.add_argument('--llm_confidence', type=float, default=100.0)
    parser.add_argument('--passive_full',   action='store_true')
    args = parser.parse_args()

    mydata = loadMyDataset(2000)
    learnOnline(
        mydata, batch_size=128, cuda=True, seed=args.seed,
        llm_type=args.llm_type,
        linucb_alpha=args.linucb_alpha, lam=args.lam,
        llm_confidence=args.llm_confidence,
        passive_full=args.passive_full,
    )
