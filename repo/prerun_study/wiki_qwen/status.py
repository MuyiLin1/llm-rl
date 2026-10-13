"""Progress of the WikiLinks runs: Flan-T5 c=1e4 evaluation (runs first), then the Qwen sweep and evaluation.
Usage: python3 /Users/linmuyi/code/DMwithLLM/prerun_study/wiki_qwen/status.py"""
import os, glob, json, subprocess, sys, time
Q = os.path.dirname(os.path.abspath(__file__)); F = os.path.join(os.path.dirname(Q), 'wiki_c10000')
sys.path.insert(0, Q)
def procs(pat):
    out = subprocess.run(['ps', '-Ao', 'command'], capture_output=True, text=True).stdout.splitlines()
    return [l for l in out if pat in l and not l.startswith('caffeinate') and 'grep' not in l]
src = open(os.path.join(Q, 'run.py')).read().split("if __name__ == '__main__':")[0].replace('from sim import Data, sm_update', '')
ns = {'__file__': os.path.join(Q, 'run.py')}; exec(compile(src, 'run.py', 'exec'), ns)

flan_done = len([p for p in glob.glob(f'{F}/out/*_s*.npz') if 'ckpt' not in p])
orch = procs('wiki_qwen/orchestrate.sh'); fw = procs('wiki_c10000/run.py'); qw = procs('wiki_qwen/run.py')
print(f"Chain running: {'yes' if orch or fw or qw else 'NO -- restart with: bash ' + Q + '/start.sh'}")
print(f"\n1) Flan-T5 c=1e4 evaluation: {flan_done}/30 finished, {len(fw)} workers running")
if flan_done < 30: print(f"   details: python3 {F}/status.py")

sw = ns['sweep_jobs'](); sd = sum(ns['done'](j) for j in sw)
print(f"\n2) Qwen sweep (tuning seeds 11-20): {sd}/{len(sw)} finished, {len(qw)} Qwen workers running")
cj = os.path.join(Q, 'out', 'chosen_c.json')
if os.path.exists(cj):
    z = json.load(open(cj)); ev = ns['eval_jobs'](z['c']); ed = sum(ns['done'](j) for j in ev)
    print("   mean Router accuracy by c: " + ", ".join(f"c={k}: {v*100:.2f}%" for k, v in z['mean_router_acc'].items()))
    print(f"   selected c = {z['c']:g}")
    print(f"\n3) Qwen evaluation (seeds 1-10) at c={z['c']:g}: {ed}/{len(ev)} finished")
else:
    print(f"\n3) Qwen evaluation (seeds 1-10): starts when the sweep is done")
running = []
for p in sorted(glob.glob(f'{Q}/logs/*.progress')):
    try: z = json.load(open(p))
    except Exception: continue
    if not os.path.exists(f"{Q}/out/{z['job']}.npz"): running.append(z)
if running:
    print("\n   in progress:")
    for z in running:
        print(f"   {z['job']:18} {z['t']/z['T']*100:5.1f}%  acc so far {z['acc']*100:5.2f}%  routed {z['routed']*100:5.1f}%  ~{z['eta_hours']:.1f} h left  (updated {z['updated']})")
done_rates = [json.load(open(p))['rounds_per_sec'] for p in glob.glob(f'{Q}/logs/*.progress')]
left = (len(sw) - sd) + (len(ns['eval_jobs'](1)) if not os.path.exists(cj) else len(ev) - ed)
if done_rates and left:
    r = sorted(done_rates)[len(done_rates)//2]; hrs = left * 622000 / r / 3600 / 3
    print(f"\n   rough time left for Qwen: ~{hrs:.0f} h ({left} runs, 3 at a time)")
print(f"\nLogs: {Q}/logs/  (orchestrator.log, worker0-2.log)")
