import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def x_tr(x):
    return -np.log(x - 0.45)


def x_tr_inv(x):
    return np.exp(-x) + 0.45


def exp_tr(x, upper):
    return upper - np.exp(-x)


def exp_inv(x, upper):
    return -np.log(upper - x)


def logistic(x, r):
    return r * 4 * x * (1 - x)


def iterate_r(func, x_0, r, prep_times, plot_times):
    """
    Iterate the logistic function with initial value x_0 for r = r
    x_1 = func(x_0,r), etc.

    func shall have signature func(x, r) -> x, simliar to logistic function

    x_0 to x_{prep_times-1} are ignored.
    x_{prep_times} to x_{prep_times+plot_times-1} are recorded in res
    and returned
    """
    res = []
    for i in range(12):
        val = np.random.uniform(0, 1)
        for _ in range(prep_times):
            if abs(val) > 1e6:
                return []
            val = func(val, r)

        # ignore x_500, recording values from x_501
        for _ in range(plot_times):
            if abs(val) > 1e6:
                return []
            val = func(val, r)
            res.append(val)

    return res


#################
# Global variables
#################
# Lower bound of r
r_low = 0
# Upper bound of r
r_high = 0.892512

upper = 0.892518
n_of_r = 6500  # Number of r values to try

x_log_low = 2.7
x_log_high = 3.3

# init x
x_0 = 0.5

prep_times = 100  # Number of iterations to ignore
medium_prep_times = 5 * prep_times  # Number of iterations to ignore
extra_prep_times = 15 * prep_times  # Number of iterations to ignore
plot_times = 100  # Number of iterations to plot
alpha_tmp = 0.7

# Start of plotting

fig, ax = plt.subplots()
plt.grid(color="gray", linestyle=":", linewidth=0.5)

exp_low = exp_inv(r_low, upper)
exp_high = exp_inv(r_high, upper)
exp_in_r = np.linspace(exp_low, exp_high, n_of_r)

for r in exp_in_r:
    # variable line size as for r <3.5 there are very few paths
    if r > 4:
        alpha_tmp = 0.5
        prep_times = medium_prep_times
    if r > 4.8:
        prep_times = extra_prep_times
        alpha_tmp = 0.2
    x_0 = np.random.uniform(0, 1)
    x = iterate_r(logistic, x_0, exp_tr(r, upper), prep_times, plot_times)
    # l_size = 0.02
    x = [x_tr(x) for x in x]

    # plt.plot([r]*len(x), x, ',b')
    r_dummy = np.linspace(r, r, len(x))
    ax.scatter(
        r_dummy,
        x,
        c="tab:blue",
        s=0.5,
        alpha=alpha_tmp,
        edgecolors="none",
        facecolors="tab:blue",
        marker="o",
    )

# correct the labels
x_ticks = np.linspace(exp_low, exp_high, 5)
plt.xticks(x_ticks, [f"{exp_tr(x, upper):.6f}" for x in x_ticks])
y_ticks = np.linspace(x_log_low, x_log_high, 5)
plt.yticks(y_ticks, [f"{x_tr_inv(y):.6f}" for y in y_ticks])

# plot a line a y = 0.5
ax.plot(
    [exp_low, exp_high],
    [x_tr(0.5), x_tr(0.5)],
    color="r",
    linestyle="--",
    linewidth=0.8,
    alpha=0.5,
    label="x=0.5",
)
ax.legend(loc="upper left", fontsize=10)

plt.ylim(x_log_low, x_log_high)
plt.xlabel("$\lambda$", fontsize=15)
plt.ylabel("x", fontsize=15)

# plt.xlim(r_low, r_high)
plt.tight_layout()
# plt.show()
# plt.axis('off')
# plt.gca().set_position([0, 0, 1, 1])

# dpi = 2000 -> 28 MB image
# dpi = 1000 -> 7 MB image
plt.savefig("log_bifurcation.png", dpi=200)
