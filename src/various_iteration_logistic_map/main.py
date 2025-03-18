import matplotlib.pyplot as plt
from utils import logistic

lambda_s = [0.1, 0.25, 0.6, 0.8, 0.9, 1]
x_0 = [0.1, 0.2, 0.7, 0.9]

plot_times = [i for i in range(29)]

row = 2
col = 3
fig, axs = plt.subplots(row, col, figsize=(col*5, row*5))
for i in range(row):
    for j in range(col):
        for k in x_0:
            y_vals = [k]
            for l_val in plot_times:
                y_vals.append(logistic(lambda_s[col*i + j], y_vals[l_val]))
            axs[i][j].plot(y_vals, '-o', label=f"$x_0 = {k}$")
            axs[i][j].set_ylim((0, 1))
            axs[i][j].legend(fontsize=16)
            if i == 0:
                axs[i][j].set_xticks([])
            if j != 0:
                axs[i][j].set_yticks([])
            if j == 0:
                axs[i][j].set_ylabel("Population", fontsize=2)
            if i != 0:
                axs[i][j].set_xlabel("Time", fontsize=22)

            axs[i][j].set_title(
                rf"$\lambda = {lambda_s[col*i + j]}$", fontsize=24)
            axs[i][j].tick_params(axis='both', which='major', labelsize=20)
            axs[i][j].tick_params(axis='both', which='minor', labelsize=20)


plt.tight_layout()
plt.savefig("logistic_line_fig.png")
plt.show()
