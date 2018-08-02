#!/usr/bin/env python
import glob
from collections import defaultdict
import os.path

for t in ['auto', 'auto_pos', 'gates', 'manual']:
    hp = defaultdict(list)
    for d in glob.glob("results/1-factor/{}/*/*.txt".format(t)):
        avg_test_acc  = float(open(d).readlines()[1].split(" ")[-1])
        hp[os.path.dirname(d)].append(avg_test_acc)
    for x in hp:
        hp[x] = sum(hp[x]) / len(hp[x])
    best = None
    for x in hp:
        if best is None or hp[x] > hp[best]:
            best = x
    print(t, best)
        

