import numpy as np
from utils import logistic, iterate

feigenbuam_delta = 4.4669


def cal_fei_const(fn, l_low, l_high, step, spstable_fn, threshold):
    n_cycle = 1
    alpha = []
    beta = []
    prev_l = l_low
    cur_l = l_low
    while cur_l < l_high:
        if n_cycle > threshold:
            break
        x_0 = spstable_fn(cur_l)
        if abs(x_0 - iterate(fn, cur_l, x_0, 2 ** (n_cycle - 1))) < 1e-6:
            print(n_cycle)
            n_cycle += 1
            alpha.append(cur_l)
            if n_cycle > 1:
                beta.append(iterate(fn, cur_l, x_0, 2 ** (n_cycle - 2)) - x_0)
            cur_l += (cur_l - prev_l) / feigenbuam_delta / 2
            step /= feigenbuam_delta
        cur_l += step

    d_alpha = [alpha[i] - alpha[i - 1] for i in range(1, len(alpha))]
    ratio_alpha = [d_alpha[i - 1] / d_alpha[i] for i in range(1, len(d_alpha))]
    ratio_beta = [beta[i - 1] / beta[i] for i in range(1, len(beta))]

    print(f"cur_l: {cur_l}")
    print("Alpha:")
    print(alpha)
    print("Beta")
    print(beta)


cal_fei_const(logistic, 0, 1, 0.000001, lambda a: 0.5, 5)
