"""
experiment_algorithm1_v4.py  —  SVD-reduced features

Identical to experiment_algorithm1.py except:
  1. Action embeddings are SVD-reduced:  (311, 768) → (311, rank)
  2. Context embeddings are projected to match:  (768,) → (rank,)
     via  z_x = U_rank^T @ x_pre   (project into action SVD basis)
  3. Joint feature is element-wise product in the reduced space:
       φ(x, a) = z_x ⊙ z_a  ∈ ℝ^rank
  4. LinUCB operates on rank-d features (default 50) instead of 768-d.
  5. Default batch_size=32 (matching Chen et al.) for more frequent updates.

The theoretical guarantees (Theorems 1–2) apply unchanged — they are
parameterised by the feature dimension d, which is now `rank`.

The SVD projection is computed once at startup from the action embedding
matrix.  No information from contexts or rewards leaks into the projection.
"""

import torch
import argparse
import warnings
import numpy as np
import json
import gzip
import pickle
import gc
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from scipy.linalg import eigvalsh

warnings.filterwarnings('ignore', message='.*encountered in matmul.*', category=RuntimeWarning)


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
        return self.Xs[i], self.ys[i], self.pre_texts[i], self.post_texts[i], self.text_entities[i], i


def loadMyDataset(threshold):
    with gzip.open(f'mydataset.{threshold}.pickle.gz', 'rb') as f:
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
    sentences = [pre + ' [MASK]. ' + post for pre, post in zip(pre_texts, post_texts)]
    predicted = generate_mask_fillings(sentences, t5_model, t5_tokenizer, device)
    pred_embd = torch.FloatTensor(entity_model.encode(predicted)).to(device)
    pred_embd = pred_embd.unsqueeze(1).repeat(1, num_entities, 1)
    ent_exp = entity_embd.unsqueeze(0).repeat(len(pre_texts), 1, 1).to(device)
    sims = cos(pred_embd, ent_exp)
    labels = torch.argmax(sims, dim=-1).detach()
    return labels, sims.detach().cpu().numpy()


def get_embd(dataset):
    d = {}
    for k in dataset.labelfeats:
        id_, embd = dataset.labelfeats[k]
        d[id_] = torch.FloatTensor(embd)
    return torch.stack([d[k] for k in sorted(d)])


# ================================================================
# SVD projection helper
# ================================================================
def compute_svd_projection(action_features, rank):
    """
    Given action_features (K, D), compute truncated SVD and return:
      - action_reduced: (K, rank)  — SVD-reduced action embeddings
      - U_rank: (D, rank)          — projection matrix for contexts
      - S_rank: (rank,)            — singular values (for diagnostics)

    The projection for a context vector x is:  z_x = U_rank^T @ x
    """
    # action_features: (K, D) — rows are action embeddings
    # SVD: A = U S V^T, where A is (K, D)
    # We want the top-rank right singular vectors (the D-dimensional basis)
    # np.linalg.svd on A^T gives us what we want efficiently
    # Or equivalently: U_A, S_A, Vt_A = svd(A) where Vt_A rows are (D,)
    U_A, S_A, Vt_A = np.linalg.svd(action_features, full_matrices=False)
    # Vt_A: (min(K,D), D) — rows are right singular vectors
    # U_rank: (D, rank) — the projection matrix
    U_rank = Vt_A[:rank, :].T  # (D, rank)
    S_rank = S_A[:rank]

    # Reduced action features: project each action into the SVD basis
    # z_a = U_rank^T @ a = a @ U_rank, shape (K, rank)
    action_reduced = action_features @ U_rank  # (K, rank)

    return action_reduced, U_rank, S_rank


# ================================================================
# Main experiment loop
# ================================================================
def learnOnline(dataset, batch_size, cuda, seed, llm_type,
                linucb_alpha=1.0, lam=0.1, alpha_decay=0.9985,
                llm_confidence=100.0, passive_full=False,
                llm_cache_path=None, svd_rank=50):
    """
    SVD-reduced version.  New parameter:
      svd_rank : int — target feature dimension after SVD projection.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

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

    # --- Load LLM cache or live models ---
    cached_labels = None
    if llm_cache_path:
        print(f"Loading LLM cache: {llm_cache_path}")
        with gzip.open(llm_cache_path, 'rb') as f:
            cache = pickle.load(f)
        cached_labels = cache['labels']
        print(f"  Cache: {cached_labels.shape[0]} samples, config: {cache['config']}")
        t5_model = t5_tokenizer = entity_model = cos = None
    else:
        t5_model, t5_tokenizer = load_model_and_tokenizer(f"google/flan-t5-{llm_type}", device)
        entity_model = SentenceTransformer('bert-base-nli-mean-tokens', device=device)
        cos = nn.CosineSimilarity(dim=2, eps=1e-6)

    data_gen = torch.Generator()
    data_gen.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
        generator=data_gen
    )

    # --- SVD projection ---
    action_features_raw = entity_embd.cpu().numpy()  # (K, 768)
    D_orig = action_features_raw.shape[1]

    # Only use the first half (pre-text embedding dim) for context projection
    # In the original code, context Xs are (1536,) = concat(pre_768, post_768)
    # and action_features are 768-d (entity embeddings from bert-base-nli-mean-tokens).
    # The element-wise product uses cx = Xs[:768] (the pre-text half).
    # For SVD, we compute SVD of the action matrix (311, 768) and project both.

    action_reduced, U_rank, S_rank = compute_svd_projection(action_features_raw, svd_rank)
    d = svd_rank  # feature dimension for LinUCB

    print(f"SVD projection: {D_orig}d → {d}d")
    print(f"Top singular values: {S_rank[:5]}")
    print(f"Variance captured: {np.sum(S_rank**2) / np.sum(np.linalg.svd(action_features_raw, compute_uv=False)**2):.3f}")
    print(f"Feature dim: {d}")

    # --- Warm LinUCB ---
    V_inv_warm = (1.0 / lam) * np.eye(d)
    b_warm = np.zeros(d)

    # --- Cold LinUCB ---
    V_inv_cold = (1.0 / lam) * np.eye(d)
    b_cold = np.zeros(d)

    inv_sqrt_c = 1.0 / np.sqrt(llm_confidence)

    # --- Accumulators ---
    acc_router = EasyAcc()
    acc_llm    = EasyAcc()
    acc_cold   = EasyAcc()

    llm_rounds   = 0
    total_rounds = 0

    TAU_THRESHOLD = 0.05
    tau_detected = False
    tau_round = 0
    tau_batch = 0
    tail_gap_sum = 0.0
    tail_gap_count = 0
    llm_correct_post_tau = 0
    llm_total_post_tau = 0
    max_feat_norm = 0.0

    history = {
        'batch': [], 'hybrid': [], 'llm': [], 'linucb_cold': [],
        'llm_fraction': [],
        'eigenvalues_warm_bottom10': [],
        'eigenvalues_cold_bottom10': [],
        'tail_gap_cumulative': [],
        'tau_round': None,
        'tau_batch': None,
        'max_feat_norm': [],
        'config': {
            'algorithm': 'confidence_gated_routing_svd',
            'features': f'elementwise_svd{d}d',
            'svd_rank': svd_rank,
            'original_dim': D_orig,
            'linucb_alpha': linucb_alpha, 'lam': lam,
            'llm_type': llm_type, 'alpha_decay': alpha_decay,
            'llm_confidence': llm_confidence,
            'passive_full': passive_full,
            'batch_size': batch_size,
        }
    }

    tag = f"algo1v4_{llm_type}_c{llm_confidence}_r{svd_rank}"
    if passive_full:
        tag += "_pfull"
    tag += f"_bs{batch_size}"
    if seed != 1:
        tag += f"_s{seed}"
    fname = f"results_{tag}.json"
    print(f"Results file: {fname}")
    print(f"Total batches: {len(loader)}")
    print("=" * 90)

    for bno, (Xs, ys, pre_texts, post_texts, text_entities, indices) in enumerate(loader):

        current_alpha = max(0.1, linucb_alpha * (alpha_decay ** bno))
        Xs, ys = Xs.to(device), ys.to(device)

        if cached_labels is not None:
            batch_idx = indices.numpy()
            llm_labels = torch.tensor(cached_labels[batch_idx])
            llm_sims = None
        else:
            with torch.no_grad():
                llm_labels, llm_sims = language_model_scores(
                    pre_texts, post_texts, t5_tokenizer, t5_model,
                    entity_model, entity_embd, num_entities, cos, device
                )

        for i in range(len(Xs)):
            context_x   = Xs[i].cpu().numpy()
            d_ctx       = context_x.shape[0] // 2
            cx_raw      = context_x[:d_ctx]            # pre-text embedding (768-dim)
            true_label  = ys[i].item()
            llm_choice  = llm_labels[i].item()

            # --- SVD-project the context ---
            cx_reduced = cx_raw @ U_rank  # (rank,)

            # --- Joint features in reduced space: element-wise product (K, rank) ---
            joint_base = action_reduced * cx_reduced  # (K, rank)

            # Normalize each action's joint feature to unit L2 norm
            _norms = np.linalg.norm(joint_base, axis=1, keepdims=True)
            _norms = np.maximum(_norms, 1e-8)  # avoid div-by-zero
            joint_base = joint_base / _norms

            total_rounds += 1

            feat_norms = np.linalg.norm(joint_base, axis=1)
            round_max_norm = float(np.max(feat_norms))
            if round_max_norm > max_feat_norm:
                max_feat_norm = round_max_norm

            # ===========================================================
            # TRACK 1: LLM only
            # ===========================================================
            r_llm = 1.0 if llm_choice == true_label else 0.0
            acc_llm += r_llm

            if tau_detected:
                tail_gap_sum += (1.0 - r_llm)
                tail_gap_count += 1
                llm_total_post_tau += 1
                llm_correct_post_tau += int(r_llm)

            # ===========================================================
            # TRACK 2: Cold LinUCB
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
            # TRACK 3: Confidence-Gated Router
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
        if not tau_detected:
            llm_frac_now = llm_rounds / max(total_rounds, 1)
            if total_rounds > batch_size * 10 and llm_frac_now < TAU_THRESHOLD:
                tau_detected = True
                tau_round = total_rounds
                tau_batch = bno
                history['tau_round'] = tau_round
                history['tau_batch'] = tau_batch
                print(f"  >> TAU DETECTED at batch {bno}, round {total_rounds} (LLM frac = {llm_frac_now:.3f})")

        if bno % 10 == 0:
            llm_frac = llm_rounds / max(total_rounds, 1)
            history['batch'].append(bno)
            history['hybrid'].append(acc_router.mean())
            history['llm'].append(acc_llm.mean())
            history['linucb_cold'].append(acc_cold.mean())
            history['llm_fraction'].append(llm_frac)

            try:
                eigs_warm_vinv = eigvalsh(V_inv_warm)
                eigs_warm_v = 1.0 / eigs_warm_vinv[-10:][::-1]
                history['eigenvalues_warm_bottom10'].append(eigs_warm_v.tolist())

                eigs_cold_vinv = eigvalsh(V_inv_cold)
                eigs_cold_v = 1.0 / eigs_cold_vinv[-10:][::-1]
                history['eigenvalues_cold_bottom10'].append(eigs_cold_v.tolist())
            except Exception as e:
                print(f"  eigenvalue computation failed: {e}")
                history['eigenvalues_warm_bottom10'].append([])
                history['eigenvalues_cold_bottom10'].append([])

            eps_tail = (tail_gap_sum / max(tail_gap_count, 1)) if tau_detected else None
            history['tail_gap_cumulative'].append(eps_tail)
            history['max_feat_norm'].append(max_feat_norm)

            print(
                f"Batch {bno:4d} | "
                f"Router: {acc_router.mean():.4f} | "
                f"LLM only: {acc_llm.mean():.4f} | "
                f"Cold LinUCB: {acc_cold.mean():.4f} | "
                f"LLM routing: {llm_frac:.1%}"
            )

            with open(fname, "w") as f:
                json.dump(history, f)

        del Xs, ys, pre_texts, post_texts, text_entities, indices, llm_labels, llm_sims
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
    if tau_detected:
        eps_tail_final = tail_gap_sum / max(tail_gap_count, 1)
        print(f"Tau detected at round {tau_round} (batch {tau_batch})")
        print(f"Epsilon_tail (avg LLM gap post-tau): {eps_tail_final:.4f}")
        print(f"LLM accuracy post-tau: {llm_correct_post_tau}/{llm_total_post_tau} = {llm_correct_post_tau/max(llm_total_post_tau,1):.4f}")
    else:
        print("Tau NOT detected (LLM routing never dropped below threshold)")
    print(f"Max feature norm L: {max_feat_norm:.4f}")

    with open(fname, "w") as f:
        json.dump(history, f)
    print(f"Results saved to: {fname}")
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Algorithm 1 v4: SVD-Reduced Features with Confidence-Gated Routing'
    )
    parser.add_argument('--seed',           type=int,   default=1)
    parser.add_argument('--llm_type',       type=str,   default='small',
                        choices=['small', 'base', 'large'])
    parser.add_argument('--linucb_alpha',   type=float, default=1.0)
    parser.add_argument('--lam',            type=float, default=0.1)
    parser.add_argument('--llm_confidence', type=float, default=100.0)
    parser.add_argument('--passive_full',   action='store_true')
    parser.add_argument('--llm_cache',      type=str, default=None)
    parser.add_argument('--svd_rank',       type=int,   default=50,
                        help='SVD rank for feature reduction (default: 50)')
    parser.add_argument('--batch_size',     type=int,   default=32,
                        help='Batch size (default: 32, matching Chen et al.)')
    args = parser.parse_args()

    mydata = loadMyDataset(2000)
    learnOnline(
        mydata, batch_size=args.batch_size, cuda=True, seed=args.seed,
        llm_type=args.llm_type,
        linucb_alpha=args.linucb_alpha, lam=args.lam,
        llm_confidence=args.llm_confidence,
        passive_full=args.passive_full,
        llm_cache_path=args.llm_cache,
        svd_rank=args.svd_rank,
    )
