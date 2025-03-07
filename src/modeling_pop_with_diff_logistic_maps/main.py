import numpy as np
import matplotlib.pyplot as plt


def logistic_map(l, x):
    return 4 * l * x * (1 - x)


def iterate_logistic_map(l, x, n):
    ret = []
    ret.append(x)
    for i in range(n):
        x = logistic_map(l, x)
        ret.append(x)
    return ret


row = 3
col = 2
lambda_vals = [0.1, 0.25, 0.6, 0.8, 0.9, 1]
fig, ax = plt.subplots(row, col, figsize=(12, 14))

for i in range(row):
    t_row = i
    for j in range(col):
        t_col = j
        lambda_v = lambda_vals[t_row * col + t_col]
        ax[t_row][t_col].set_title(f"$\lambda$ = {lambda_v}", fontsize=30)
        ax[t_row][t_col].plot(
            iterate_logistic_map(lambda_v, 0.1, 30), "-o", label="$x_0$ = 0.1"
        )
        ax[t_row][t_col].plot(
            iterate_logistic_map(lambda_v, 0.3, 30), "-o", label="$x_0$ = 0.2"
        )
        ax[t_row][t_col].plot(
            iterate_logistic_map(lambda_v, 0.7, 30), "-o", label="$x_0$ = 0.7"
        )
        ax[t_row][t_col].plot(
            iterate_logistic_map(lambda_v, 0.87, 30), "-o", label="$x_0$ = 0.9"
        )
        ax[t_row][t_col].legend(fontsize=24)
        # set ticks size
        ax[t_row][t_col].tick_params(axis="both", labelsize=14)
        ax[t_row][t_col].set_xlabel("Time", fontsize=20)
        ax[t_row][t_col].set_ylabel("Population", fontsize=20)

frame = fig.add_subplot(111, frameon=False)
frame.set_xticks([])
frame.set_yticks([])
# frame.set_ylabel('Population', fontsize=30, labelpad=40)
# frame.set_xlabel('Time', fontsize=30, labelpad=40)
plt.subplots_adjust(hspace=0.33)
plt.tight_layout()


fig.savefig("logistic.png", dpi=300)
