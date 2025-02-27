import numpy as np
import matplotlib.pyplot as plt


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
r_high = 0.89

upper = 0.892518
n_of_r = 2500  # Number of r values to try

x_low = 0
x_high = 1

# init x
x_0 = 0.5

prep_times = 600  # Number of iterations to ignore
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
    x_0 = np.random.uniform(0, 1)
    x = iterate_r(logistic, x_0, exp_tr(r, upper), prep_times, plot_times)

    r_dummy = np.linspace(r, r, len(x))
    ax.scatter(
        r_dummy,
        x,
        c="tab:blue",
        s=0.3,
        alpha=0.8,
        edgecolors="none",
        facecolors="tab:blue",
        marker="o",
    )

# Label the A_values
A_List = [2, 3.23606797749979, 3.4985479775008543, 3.554637435512696]
A_List_lambda = [x / 4 for x in A_List]
A_List = [exp_inv(x / 4, upper) for x in A_List]

x_ticks = np.linspace(exp_low, exp_high, 3).tolist()
x_ticks_label = [f"{exp_tr(x, upper):.2f}" for x in x_ticks]
x_ticks += A_List
x_ticks_label += [f"$A_{i}$" for i in range(len(A_List))]
plt.xticks(x_ticks, x_ticks_label, fontsize=16)

# Graph the d values
closest_stable_pos = []
for i in range(1, len(A_List)):
    x_0 = 0.5
    for j in range(2**(i - 1)):
        x_0 = logistic(x_0, A_List_lambda[i])
    closest_stable_pos.append(x_0)

for id, ele in enumerate(closest_stable_pos):
    plt.plot([A_List[id+1], A_List[id + 1]], [0.5, ele], 'm:')
    plt.text(A_List[id+1] + 0.1, (ele + 0.5)/2, f"$d_{id+1}$",
             horizontalalignment='left',
             verticalalignment='center',
             fontsize=18,)

# plot a line a y = 0.5
ax.plot(
    [exp_low, exp_high],
    [0.5, 0.5],
    color="r",
    linestyle="--",
    linewidth=0.8,
    alpha=0.5,
    label="x=0.5",
)
ax.legend(loc="upper left", fontsize=10)

plt.xlabel(r"$\lambda$", fontsize=18)
plt.ylabel("x", fontsize=18)

# plt.xlim(r_low, r_high)
plt.tight_layout()
# plt.show()
# plt.axis('off')
# plt.gca().set_position([0, 0, 1, 1])

# dpi = 2000 -> 28 MB image
# dpi = 1000 -> 7 MB image
plt.savefig("demonstration of feigenbuam constants", dpi=200)
