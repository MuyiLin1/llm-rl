import json, glob, os, csv, sys
D=os.path.expanduser('~/DMwithLLM/amazoncat13k/study')
rows=[]
for p in sorted(glob.glob(os.path.join(D,'results_algo1_*.json'))):
    try: d=json.load(open(p))
    except Exception: continue
    c=d.get('config') or {}
    if not d.get('step'): continue
    rows.append(dict(
        run_tag=os.path.basename(p).replace('results_algo1_','').replace('.json',''),
        dataset=c.get('dataset'), model_family=c.get('model_family'),
        features=c.get('features'), llm_type=c.get('llm_type'), seed=c.get('seed'),
        router=c.get('router') or 'gate', llm_confidence=c.get('llm_confidence'),
        alpha_decay=c.get('alpha_decay'), lam=c.get('lam'),
        linucb_alpha=c.get('linucb_alpha'), passive_full=c.get('passive_full'),
        offline_tau=c.get('offline_tau'), corral_eta=c.get('corral_eta'),
        corral_pmin=c.get('corral_pmin'), num_actions=c.get('num_actions'),
        T_completed=d['step'][-1], n_checkpoints=len(d['step']),
        router_acc=round(d['hybrid'][-1],6), cold_acc=round(d['linucb_cold'][-1],6),
        llm_acc=round(d['llm'][-1],6), routed_frac=round(d['llm_fraction'][-1],6)))
cols=list(rows[0].keys())
w=csv.DictWriter(sys.stdout, fieldnames=cols); w.writeheader()
for r in sorted(rows, key=lambda r:(str(r['router']),str(r['dataset']),str(r['llm_type']),r['seed'] or 0)): w.writerow(r)
