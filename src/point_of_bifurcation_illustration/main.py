import numpy as np
import matplotlib.pyplot as plt

from utils import logistic, iterate


def graph_bifurcation_comparison_plot(
    iter_times, lambda_1, lambda_2, x_low, x_high, fig_name
):
    I = np.linspace(x_low, x_high, 1000)
    x_ticks = np.linspace(x_low, x_high, 5)
    for i in range(len(x_ticks)):
        x_ticks[i] = round(x_ticks[i], 2)

    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    # set common label
    # frame = fig.add_subplot(111, frameon=False)
    # frame.set_ylabel('Population', fontsize=20, labelpad=40)

    #####
    # Second row, first column
    #####
    ax[0].plot(
        I,
        [iterate(logistic, lambda_1, x, iter_times) for x in I],
        label="$L_{\lambda}^%d$" % iter_times,
    )
    ax[0].plot(
        I,
        [iterate(logistic, lambda_1, x, 2 * iter_times) for x in I],
        label="$L_{\lambda}^%d$" % (2 * iter_times),
    )
    ax[0].plot(I, I, label="$x$")
    # a0[0].set_xlabel('x', fontsize=18)
    ax[0].tick_params(axis="both", labelsize=18)
    # ax[0].set_yticks([])
    ax[0].set_xticks(x_ticks)
    ax[0].legend(fontsize=18)
    ax[0].set_title("$\lambda = %.4f$" % lambda_1, fontsize=20, y=-0.15)

    #####
    # Second row, second column
    #####

    ax[1].plot(
        I,
        [iterate(logistic, lambda_2, x, iter_times) for x in I],
        label="$L_{\lambda}^%d$" % iter_times,
    )
    ax[1].plot(
        I,
        [iterate(logistic, lambda_2, x, 2 * iter_times) for x in I],
        label="$L_{\lambda}^%d$" % (2 * iter_times),
    )
    # remove x ticks
    ax[1].plot(I, I, label="$x$")
    # a1[0].set_xlabel('x', fontsize=18)
    ax[1].tick_params(axis="both", labelsize=18)
    ax[1].set_xticks(x_ticks)
    ax[1].set_yticks([])
    ax[1].ylim = (0.4, 1)
    # incse ticks spacing
    ax[1].legend(fontsize=18)
    ax[1].set_title("$\lambda = %.4f$" % lambda_2, fontsize=20, y=-0.15)

    plt.tight_layout()
    # plt.show()

    plt.savefig(fig_name)


x_low = 2 / 3 - 1 / 6
x_high = 2 / 3 + 1 / 6
lambda_1 = 0.75
lambda_2 = 0.78
fig_name = "first_bifurcation.png"

graph_bifurcation_comparison_plot(1, lambda_1, lambda_2, x_low, x_high, fig_name)

# x_low = 0.38
# x_high = 0.5
# lambda_1 = 0.863
# lambda_2 = 0.868
# fig_name = 'second_bifurcation.png'
#
# graph_bifurcation_comparison_plot(
#     2, lambda_1, lambda_2, x_low, x_high, fig_name)
