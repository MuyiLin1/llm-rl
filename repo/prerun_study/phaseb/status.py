"""Phase B progress.  On the cluster:  python3 ~/DMwithLLM/phaseb/status.py
From the Mac:  bash /Users/linmuyi/code/DMwithLLM/prerun_study/phaseb/status.sh"""
import os, json, glob, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__)); LOG = os.environ.get('PB_LOG', f'{HERE}/logs')
alive = subprocess.run(['pgrep', '-f', os.environ.get('PB_PROC', 'phaseb.py')], capture_output=True, text=True).stdout.split()
try: st = json.load(open(f'{LOG}/status.json'))
except Exception: print('no status yet'); raise SystemExit
print(f"Phase B driver running: {'yes' if alive else 'NO -- restart with: bash ~/DMwithLLM/phaseb/start.sh'}   (status updated {st['updated']})\n")
for c in st['chains']:
    tot = c['specs_total']; d = c['specs_done']
    bar = f"{d}/{tot} runs" if tot else ''
    print(f"  {c['name']:14}  {c['stage']:58} {bar:>12}   running batches: {c['batches_running']}, queued: {c['batches_pending']}")
progs = []
for p in glob.glob(f'{LOG}/*.prog'):
    try: progs.append(json.load(open(p)))
    except Exception: pass
if progs:
    rate = sum(p['rounds_per_sec']*p['K'] for p in progs)
    print(f"\n  {len(progs)} batches in progress; typical batch {sorted(p['t']/p['T']*100 for p in progs)[len(progs)//2]:.0f}% done; "
          f"total speed {rate:.0f} learner-rounds/s")
# Rough time left: planned runs per group (upper estimates) x recent minutes per run in that group / workers.
import re, statistics
OUT = os.environ.get('PB_OUT', LOG.replace('logs', 'out'))
PLAN = {'bert_prod': 520, 'bge_prod': 360, 'bert_prodarm': 180, 'bge_prodarm': 180,
        'bertcap30_prod': 40, 'bertcap20_prod': 40, 'bertcap10_prod': 40, 'bertcap05_prod': 40}
WEIGHT = {'bert_prod': 1.0, 'bge_prod': 1.6, 'bert_prodarm': 3.6, 'bge_prodarm': 7.0,
          'bertcap30_prod': 0.28, 'bertcap20_prod': 0.28, 'bertcap10_prod': 0.28, 'bertcap05_prod': 0.28}
SKIPPED = {'BGE Tier 2': 'bge_prodarm', 'BERT Tier 2': 'bert_prodarm'}
if os.path.exists(f'{LOG}/skip.txt'):
    for l in open(f'{LOG}/skip.txt'):
        if l.strip() in SKIPPED: PLAN.pop(SKIPPED[l.strip()], None)
mins = {}
try:
    for line in open(f'{LOG}/driver.log').readlines():
        m = re.search(r'\] (\w+?_prod(?:arm)?)_.* done: .*\((\d+) min\)', line)
        if m: mins.setdefault(m.group(1), []).append(int(m.group(2)))
except FileNotFoundError: pass
if mins.get('bert_prod'):
    base = statistics.median(mins['bert_prod'][-30:]); left_h = 0; parts = []
    for g, n in PLAN.items():
        done_n = len([p for p in glob.glob(f'{OUT}/{g}_*.npz') if '/ckpt_' not in p and '.tmp' not in p])
        m = statistics.median(mins[g][-30:]) if mins.get(g) else base*WEIGHT[g]
        h = max(0, n - done_n)*m/max(1, st['workers'])/60; left_h += h
        parts.append(f"{g}: {done_n}/{n} runs, ~{h:.0f} h")
    print(f"\n  Rough time left at the current speed: ~{left_h:.0f} h (~{left_h/24:.1f} days)\n    " + '\n    '.join(parts))
sel = st.get('selections', {})
if sel:
    print('\nSelections so far (tuning seeds 11-20):')
    for fam in ('bert', 'bge'):
        if f'{fam}_S1_rank' in sel:
            r = sel[f'{fam}_S1_rank']
            print(f"  {fam.upper()}: exploration grid, best 3: " + ', '.join(f"{x[0]}/floor {x[1]:g}: {x[2]*100:.2f}%" for x in r[:3]) +
                  f"   (default fast/floor 0.1: {[x for x in r if x[0]=='fast' and x[1]==0.1][0][2]*100:.2f}%)")
        for k, lab in ((f'{fam}_cold_best', 'best cold setting (schedule, floor, lambda)'), (f'{fam}_router_c_at_best', 'Router c at best setting'),
                       (f'{fam}_arm_best', 'best category-term cold setting'), (f'{fam}_A3_c', 'category-term Router c scores')):
            if k in sel: print(f"  {fam.upper()}: {lab}: {sel[k]}")
print(f"\nLog: {LOG}/driver.log")
