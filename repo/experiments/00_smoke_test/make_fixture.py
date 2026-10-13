"""Build a tiny synthetic dataset in the same format as the real slim .npz files.

Purpose: let anyone exercise the exact code path of the online experiments in
seconds, without the 228 MB-2.6 GB data files. The fixture is written in the
same schema the runner expects:

    contexts  (n, d) float32     context embeddings
    Z         (K, d) float32     action embeddings
    y_flat    (nnz,) int32       correct action ids, concatenated
    y_offsets (n+1,) int64       ragged offsets into y_flat
    n_keep    ()     int64       how many contexts form the online stream
    llm_small / llm_base / llm_large  (n,) int64   each expert's cached proposal

The three synthetic "experts" are built to have near-equal accuracy but
different error structure, so the smoke test also exercises the A/V/B
decomposition with non-degenerate numbers.
"""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='fixture_slim.npz')
    ap.add_argument('--n', type=int, default=1200, help='contexts (stream + held-out pool)')
    ap.add_argument('--n_keep', type=int, default=1000, help='online stream length')
    ap.add_argument('--actions', type=int, default=40)
    ap.add_argument('--dim', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n, K, d = args.n, args.actions, args.dim

    contexts = rng.standard_normal((n, d)).astype(np.float32)
    contexts /= np.linalg.norm(contexts, axis=1, keepdims=True)
    Z = rng.standard_normal((K, d)).astype(np.float32)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)

    # A planted linear model decides the correct action, so the task is learnable.
    theta = rng.standard_normal(d).astype(np.float32)
    scores = (contexts * theta) @ Z.T                      # (n, K)
    best = scores.argmax(1)

    # Multi-label: the best action plus a random runner-up, like AmazonCat.
    second = scores.argsort(1)[:, -2]
    y_flat, y_offsets = [], [0]
    for i in range(n):
        labs = {int(best[i]), int(second[i])}
        y_flat.extend(sorted(labs)); y_offsets.append(len(y_flat))

    # Three experts with similar accuracy but different mistakes.
    def expert(p_correct, bias_to):
        out = np.empty(n, dtype=np.int64)
        for i in range(n):
            if rng.random() < p_correct:
                out[i] = best[i]
            else:                       # wrong, but concentrated on a subset
                out[i] = int(rng.integers(bias_to[0], bias_to[1]))
        return out

    z = dict(contexts=contexts, Z=Z,
             y_flat=np.array(y_flat, dtype=np.int32),
             y_offsets=np.array(y_offsets, dtype=np.int64),
             n_keep=np.int64(args.n_keep),
             llm_small=expert(0.20, (0, K // 2)),
             llm_base=expert(0.20, (K // 2, K)),
             llm_large=expert(0.22, (0, K)))
    np.savez_compressed(args.out, **z)
    acc = {k: float((z[k][:args.n_keep] == best[:args.n_keep]).mean())
           for k in ('llm_small', 'llm_base', 'llm_large')}
    print(f"wrote {args.out}: {n} contexts, {K} actions, d={d}, stream={args.n_keep}")
    print("expert top-1 accuracies on the stream:",
          {k: round(v, 3) for k, v in acc.items()})


if __name__ == '__main__':
    main()
