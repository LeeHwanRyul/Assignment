import numpy as np

def LP_Relaxation(p, l, c, n):
    # p: value
    # l: weight
    # c: max weight
    # n: item count

    n = len(p)
    ratio = p / l
    idx = np.argsort(-ratio) 
    lp_x = np.zeros(n, dtype=float)
    remaining = c
    lp_cost = 0.0

    for i in idx:
        if l[i] <= remaining:
            lp_x[i] = 1.0
            remaining -= l[i]
            lp_cost += p[i]
        else:
            lp_x[i] = remaining / l[i]
            lp_cost += p[i] * lp_x[i]
            break

    total_weight = 0
    for i in range(len(lp_x)):
        if lp_x[i] == 1:
            total_weight += l[i]

    return lp_cost, lp_x, total_weight