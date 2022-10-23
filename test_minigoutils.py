from collections import Counter

import numpy as np

from utils import soft_pick


def test_soft_pick():
    pi = np.array([.1, .5, .2, .1, .1])
    print()
    for temp in (1.0, 2.0):
        picks = [soft_pick(pi.copy(), temperature=temp) for x in range(100)]
        c1 = Counter(picks)
        print(f'temp={temp}', c1.most_common())


def test_soft_pick_topN():
    pi = np.array([.1, .5, .15, .15, .1])
    print()
    for cutoff in range(3):
        picks = [soft_pick(pi.copy(), softpick_topn_cutoff=cutoff) for x in range(100)]
        c1 = Counter(picks)
        print(f'cutoff={cutoff}', c1.most_common())