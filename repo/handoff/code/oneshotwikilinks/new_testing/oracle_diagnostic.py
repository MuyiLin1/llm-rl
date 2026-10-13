"""
Quick diagnostic: measure the oracle ceiling for Flan-T5 and Qwen models.
Oracle = fraction of samples where LLM OR LinUCB (cold) is correct.
This tells us the theoretical max the router could achieve.

Uses LLM caches to avoid loading models.
Runs the full dataset but only needs Cold LinUCB + cached LLM labels.
"""
import json, gzip, pickle, numpy as np, torch, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from experiment_algorithm1 import loadMyDataset, get_embd, sm_update, MyDataset
# Pickle needs MyDataset in __main__
import __main__
__main__.MyDataset = MyDataset

def measure_oracle(llm_cache_path, batch_size=128, seed=1):
    """Run Cold LinUCB and count oracle overlap with cached LLM labels."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"Loading dataset...")
    dataset = loadMyDataset(2000)
    entity_embd = get_embd(dataset)
    action_features = entity_embd.cpu().numpy()  # (K, 768)
    num_entities = action_features.shape[0]
    d = action_features.shape[1]  # 768

    print(f"Loading LLM cache: {llm_cache_path}")
    with gzip.open(llm_cache_path, 'rb') as f:
        cache = pickle.load(f)
    cached_labels = cache['labels']
    print(f"  Cache: {cached_labels.shape[0]} samples, config: {cache['config']}")

    data_gen = torch.Generator()
    data_gen.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=True, generator=data_gen
    )

    # Cold LinUCB parameters (match v1 defaults)
    lam = 0.1
    linucb_alpha = 1.0
    alpha_decay = 0.9985
    V_inv_cold = (1.0 / lam) * np.eye(d)
    b_cold = np.zeros(d)

    # Counters
    total = 0
    llm_correct = 0
    cold_correct = 0
    both_correct = 0
    either_correct = 0  # oracle
    neither_correct = 0

    for bno, (Xs, ys, pre_texts, post_texts, text_entities, indices) in enumerate(loader):
        batch_idx = indices.numpy()
        llm_labels = cached_labels[batch_idx]
        current_alpha = max(0.1, linucb_alpha * (alpha_decay ** bno))

        for i in range(len(Xs)):
            context_x = Xs[i].cpu().numpy()
            d_ctx = context_x.shape[0] // 2
            cx = context_x[:d_ctx]
            true_label = ys[i].item()
            llm_choice = int(llm_labels[i])

            joint_base = action_features * cx
            theta_cold = V_inv_cold @ b_cold
            Vf_cold = joint_base @ V_inv_cold
            vars_cold = np.sum(Vf_cold * joint_base, axis=1)
            ucb_cold = joint_base @ theta_cold + current_alpha * np.sqrt(np.maximum(vars_cold, 0))
            cold_choice = int(np.argmax(ucb_cold))

            r_llm = (llm_choice == true_label)
            r_cold = (cold_choice == true_label)

            total += 1
            llm_correct += int(r_llm)
            cold_correct += int(r_cold)
            both_correct += int(r_llm and r_cold)
            either_correct += int(r_llm or r_cold)
            neither_correct += int(not r_llm and not r_cold)

            # Update Cold LinUCB
            x_cold = joint_base[cold_choice]
            V_inv_cold = sm_update(V_inv_cold, x_cold)
            b_cold += float(r_cold) * x_cold

        if bno % 100 == 0 and bno > 0:
            print(f"  batch {bno}/{len(loader)}: "
                  f"LLM={llm_correct/total*100:.2f}% "
                  f"Cold={cold_correct/total*100:.2f}% "
                  f"Oracle={either_correct/total*100:.2f}% "
                  f"Both={both_correct/total*100:.2f}% "
                  f"Neither={neither_correct/total*100:.2f}%")

    print(f"\n{'='*70}")
    print(f"FINAL ({total} samples):")
    print(f"  LLM accuracy:      {llm_correct/total*100:.2f}%")
    print(f"  Cold LinUCB:       {cold_correct/total*100:.2f}%")
    print(f"  Oracle (either):   {either_correct/total*100:.2f}%")
    print(f"  Both correct:      {both_correct/total*100:.2f}%")
    print(f"  Neither correct:   {neither_correct/total*100:.2f}%")
    print(f"  Overlap ratio:     {both_correct/max(1,min(llm_correct,cold_correct))*100:.1f}% "
          f"(of the weaker system's correct answers)")
    print(f"  Oracle lift over LLM:  +{(either_correct-llm_correct)/total*100:.2f}pp")
    print(f"  Oracle lift over Cold: +{(either_correct-cold_correct)/total*100:.2f}pp")

    return {
        'total': total,
        'llm_correct': llm_correct,
        'cold_correct': cold_correct,
        'both_correct': both_correct,
        'either_correct': either_correct,
        'neither_correct': neither_correct,
    }

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--llm_cache', required=True)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--seed', type=int, default=1)
    args = p.parse_args()
    measure_oracle(args.llm_cache, args.batch_size, args.seed)
