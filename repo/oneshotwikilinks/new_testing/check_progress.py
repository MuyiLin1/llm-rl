import json, os
for f in sorted(os.listdir('.')):
    if f.startswith('results_algo1v4_') and f.endswith('.json'):
        with open(f) as fp:
            d = json.load(fp)
        n = len(d['batch'])
        if n == 0: continue
        last = d['batch'][-1]
        total = 19437
        pct = last/total*100
        status = 'DONE' if last >= 19400 else f'batch {last}/{total} ({pct:.0f}%)'
        r = d['hybrid'][-1]; cold = d['linucb_cold'][-1]; llm = d['llm'][-1]; lf = d['llm_fraction'][-1]
        mn = d['max_feat_norm'][-1] if 'max_feat_norm' in d and d['max_feat_norm'] else -1
        print(f'{f}')
        print(f'  {status}  Router={r*100:.2f}% Cold={cold*100:.2f}% LLM={llm*100:.2f}% LLMfrac={lf*100:.1f}% MaxNorm={mn:.3f}')
        # trajectory
        for idx in [0, n//4, n//2, 3*n//4, n-1]:
            b = d['batch'][idx]
            print(f'  batch {b}: Cold={d["linucb_cold"][idx]*100:.2f}% Router={d["hybrid"][idx]*100:.2f}%')
        print()
