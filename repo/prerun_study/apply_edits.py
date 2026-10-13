import sys, json
path='/Users/linmuyi/code/DMwithLLM/v6.tex'
edits=json.load(open(sys.argv[1]))
s=open(path).read()
for i,(old,new) in enumerate(edits):
    n=s.count(old)
    if n!=1: sys.exit(f"edit {i}: found {n} times, aborting. OLD starts: {old[:80]!r}")
    s=s.replace(old,new)
open(path,'w').write(s); print(f"applied {len(edits)} edits")
