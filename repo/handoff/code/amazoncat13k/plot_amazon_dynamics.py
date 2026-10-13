"""Generate learning dynamics figure for AmazonCat-13K (BERT-768d, c=100)."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load all three configs
configs = {}
for size in ['small', 'base', 'large']:
    fname = f'results_algo1_amazon_v1_{size}_c100.0_pfull.json'
    try:
        with open(fname) as f:
            configs[size] = json.load(f)
    except FileNotFoundError:
        print(f'Missing: {fname}')

fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5))

# Plot Flan-T5-large (the star result)
d = configs['large']
steps = [s/1000 for s in d['step']]
ax.plot(steps, [100*v for v in d['hybrid']], 'b-', linewidth=2, label='Router (Flan-T5-large)')
ax.plot(steps, [100*v for v in d['linucb_cold']], 'r--', linewidth=2, label='Cold LinUCB')
ax.plot(steps, [100*v for v in d['llm']], 'g:', linewidth=1.5, label='LLM only (large)')

# Also show small Router for contrast
if 'small' in configs:
    d2 = configs['small']
    steps2 = [s/1000 for s in d2['step']]
    ax.plot(steps2, [100*v for v in d2['hybrid']], color='blue', linestyle='-',
            linewidth=0.8, alpha=0.4, label='Router (Flan-T5-small)')

ax.set_xlabel('Round (thousands)', fontsize=10)
ax.set_ylabel('Cumulative accuracy (%)', fontsize=10)
ax.set_title('AmazonCat-13K, BERT-768d features, c=100', fontsize=10)
ax.legend(fontsize=8, loc='upper left')
ax.set_xlim([0, 50])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../figures_to_upload/amazon_dynamics.pdf', dpi=150, bbox_inches='tight')
plt.savefig('../figures_to_upload/amazon_dynamics.png', dpi=150, bbox_inches='tight')
print('Saved to figures_to_upload/amazon_dynamics.{pdf,png}')

# Print key values for verification
print(f"Router final: {100*d['hybrid'][-1]:.2f}%")
print(f"Cold final: {100*d['linucb_cold'][-1]:.2f}%")
print(f"LLM final: {100*d['llm'][-1]:.2f}%")
print(f"Mult: {d['hybrid'][-1]/max(d['llm'][-1], d['linucb_cold'][-1]):.2f}x")
