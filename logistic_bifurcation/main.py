import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def logistic(x, r):
    return r * 4 * x*(1-x)


def iterate_r(func, x_0, r, prep_times, plot_times):
    """
    Iterate the logistic function with initial value x_0 for r = r
    x_1 = func(x_0,r), etc.

    func shall have signature func(x, r) -> x, simliar to logistic function

    x_0 to x_{prep_times-1} are ignored.
    x_{prep_times} to x_{prep_times+plot_times-1} are recorded in res
    and returned
    """
    val = x_0
    for _ in range(prep_times):
        val = func(val, r)

    res = []
    # ignore x_500, recording values from x_501
    for _ in range(plot_times):
        val = func(val, r)
        res.append(val)

    return res


#################
# Global variables
#################
# Lower bound of r
r_low = 0.91
# Upper bound of r
r_high = 0.97
n_of_r = 2500  # Number of r values to try

# init x
x_0 = 0.5

prep_times = 200  # Number of iterations to ignore
plot_times = 2000  # Number of iterations to plot

# Start of plotting

fig, ax = plt.subplots()
plt.grid(color='gray', linestyle=':', linewidth=0.5)

r_vals = np.linspace(r_low, r_high, n_of_r)

for r in r_vals:
    # variable line size as for r <3.5 there are very few paths
    l_size = 0.05
    if r < 3.5/4:  # before 3.5 the system is rather stable
        x = iterate_r(logistic, x_0, r, prep_times, 100)
        l_size = 0.1
    else:
        x = iterate_r(logistic, x_0, r, prep_times, plot_times)
        l_size = 0.02

    # plt.plot([r]*len(x), x, ',b')
    r_dummy = np.linspace(r, r, len(x))
    ax.scatter(r_dummy, x, c='tab:blue', s=l_size,
               alpha=0.5, edgecolors='none', facecolors='tab:blue',
               marker='o')

# plot a line a y = 0.5
ax.plot([r_low, r_high], [0.5, 0.5], color='r',
        linestyle='--', linewidth=0.8, alpha=0.5, label='x=0.5')
ax.legend(loc='upper left', fontsize=10)

# Custom legend
# legend_elements = [
# Line2D([0], [0], color='r', linestyle='--', linewidth=0.8, label='1-1/r'),
# Line2D([0], [0], marker='o', color='w', label='x values',
# markerfacecolor='b', markersize=3),
# ]


# ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

plt.xlabel('$\lambda$', fontsize=15)
plt.ylabel('x', fontsize=15)

plt.xlim(r_low, r_high)
plt.tight_layout()
# plt.show()
# plt.axis('off')
# plt.gca().set_position([0, 0, 1, 1])

# dpi = 2000 -> 28 MB image
# dpi = 1000 -> 7 MB image
plt.savefig("bifurcation.png", dpi=200)
