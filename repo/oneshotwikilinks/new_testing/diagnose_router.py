"""
Diagnose why the Router lift over Cold LinUCB is negligible on WikiLinks.
Examines the dynamics: when does the advantage peak? how fast does the LLM
get phased out? what's the windowed accuracy?
"""
import json
import numpy as np
import os

RESULTS_DIR = "v2"

def load(fname):
    path = os.path.join(RESULTS_DIR, fname)
    with open(path) as f:
        return json.load(f)

def analyze(data, label):
    batches = np.array(data['batch'])
    router = np.array(data['hybrid'])
    cold = np.array(data['linucb_cold'])
    llm = np.array(data['llm'])
    llm_frac = np.array(data['llm_fraction'])
    
    lift = router - cold  # Router advantage over Cold
    
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"Config: passive_full={data['config'].get('passive_full', 'N/A')}, c={data['config']['llm_confidence']}")
    print(f"Total batches: {len(batches)}")
    
    # Final values
    print(f"\nFinal values (last batch {batches[-1]}):")
    print(f"  Router:     {router[-1]*100:.4f}%")
    print(f"  Cold:       {cold[-1]*100:.4f}%")
    print(f"  LLM:        {llm[-1]*100:.4f}%")
    print(f"  Lift:       {lift[-1]*100:+.4f} pp")
    print(f"  LLM frac:   {llm_frac[-1]*100:.4f}%")
    
    # Peak lift
    peak_idx = np.argmax(lift)
    print(f"\nPeak Router-Cold lift:")
    print(f"  Batch {batches[peak_idx]}: {lift[peak_idx]*100:+.4f} pp")
    print(f"  Router={router[peak_idx]*100:.2f}%, Cold={cold[peak_idx]*100:.2f}%")
    print(f"  LLM frac at peak: {llm_frac[peak_idx]*100:.2f}%")
    
    # When does LLM routing drop below thresholds?
    for thresh in [0.50, 0.10, 0.05, 0.01]:
        mask = llm_frac < thresh
        if mask.any():
            idx = np.argmax(mask)
            print(f"  LLM frac < {thresh*100:.0f}% at batch {batches[idx]} ({batches[idx]/batches[-1]*100:.1f}% of horizon)")
    
    # Windowed analysis: what's the lift at different horizon cutoffs?
    print(f"\nLift at horizon cutoffs:")
    for cutoff_batch in [100, 200, 500, 1000, 2000, 3000, 4000, 4860]:
        if cutoff_batch <= batches[-1]:
            idx = np.searchsorted(batches, cutoff_batch)
            if idx < len(batches):
                print(f"  Batch {cutoff_batch:5d}: Router={router[idx]*100:.3f}%, Cold={cold[idx]*100:.3f}%, Lift={lift[idx]*100:+.4f} pp, LLM%={llm_frac[idx]*100:.2f}%")
    
    # Compute INSTANTANEOUS accuracy in windows (not cumulative)
    # Approximate instantaneous acc as: d(cumulative * n_rounds) / d(n_rounds)
    print(f"\nInstantaneous accuracy in batch windows:")
    batch_size = 128
    for start_b, end_b in [(0, 100), (100, 300), (300, 1000), (1000, 2000), (2000, 4860)]:
        si = np.searchsorted(batches, start_b)
        ei = np.searchsorted(batches, end_b)
        if ei >= len(batches):
            ei = len(batches) - 1
        
        # cumulative reward at boundary = acc * n_rounds_so_far
        n_start = batches[si] * batch_size
        n_end = batches[ei] * batch_size
        if n_end > n_start:
            r_router_window = (router[ei] * n_end - router[si] * n_start) / (n_end - n_start)
            r_cold_window = (cold[ei] * n_end - cold[si] * n_start) / (n_end - n_start)
            window_lift = r_router_window - r_cold_window
            print(f"  Batch [{start_b:5d}, {end_b:5d}): Router≈{r_router_window*100:.2f}%, Cold≈{r_cold_window*100:.2f}%, Lift≈{window_lift*100:+.3f} pp")


# Load all seed-1 results
for size in ['small', 'base', 'large']:
    fname = f"results_algo1_{size}_c100.0_pfull.json"
    if os.path.exists(os.path.join(RESULTS_DIR, fname)):
        data = load(fname)
        analyze(data, f"Flan-T5-{size} (c=100, passive_full)")

# Also check v3 if available
for size in ['small', 'base', 'large']:
    fname = f"results_algo1v3_{size}_c100.0_pfull.json"
    if os.path.exists(os.path.join(RESULTS_DIR, fname)):
        data = load(fname)
        analyze(data, f"Qwen-{size} (c=100, passive_full)")

# Check for v2 results
for size in ['small', 'base', 'large']:
    fname = f"results_algo1v2_{size}_c100.0_pfull.json"
    if os.path.exists(os.path.join(RESULTS_DIR, fname)):
        data = load(fname)
        analyze(data, f"v2-{size} (c=100, passive_full)")
