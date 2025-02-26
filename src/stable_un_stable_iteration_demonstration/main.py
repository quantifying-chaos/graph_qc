import numpy as np
import matplotlib.pyplot as plt


def set_ax(ax):
    ax.set_xticks([0])
    ax.set_yticks([0])
    #  tick size
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel('$x$', fontsize=20)
    ax.set_ylabel('$y$', fontsize=20)
    # w x, y axis
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.legend(fontsize=20)


def f(x):
    return 0.4 * np.exp(x) - 0.4


def g(x):
    return 1.3 * np.exp(x) - 1.3


interval = (0, 1)
to_graph_x = []
to_graph_y = []
distinct_y = []
start_x = 0.9
n = 4

interval_g = (0, 1)
to_graph_x_g = []
to_graph_y_g = []
distinct_y_g = []
start_x_g = 0.16
n_g = 4

for i in range(n):
    if i == 0:
        x = start_x
    else:
        x = to_graph_y[-1]
    to_graph_x.append(x)
    to_graph_y.append(x)
    to_graph_x.append(x)
    to_graph_y.append(f(x))
    distinct_y.append(f(x))


for i in range(n_g):
    if i == 0:
        x = start_x_g
    else:
        x = to_graph_y_g[-1]
    to_graph_x_g.append(x)
    to_graph_y_g.append(x)
    to_graph_x_g.append(x)
    to_graph_y_g.append(g(x))
    distinct_y_g.append(g(x))


graphing_interval = np.linspace(interval[0], interval[1], 100)
graphing_interval_g = np.linspace(interval[0], interval[1], 100)
fig, ax = plt.subplots(1, 2, figsize=(14, 7))

ax[0].plot(graphing_interval, f(graphing_interval),
           label='$f(x)$', linewidth=2)
ax[0].plot(graphing_interval, graphing_interval, label='$x$', linewidth=2)
ax[0].plot(to_graph_x, to_graph_y, 'o', color='red')
ax[0].annotate('$a$', (start_x, start_x), fontsize=20,
               textcoords='offset points',  # Use offset points for xytext
               ha='right',  # Horizontal alignment of the text
               va='bottom',  # Vertical alignment of the text
               )
for i in range(len(distinct_y)-1):
    ax[0].annotate(f'$f^{i+1}(a)$', (distinct_y[i], distinct_y[i]), fontsize=20,
                   textcoords='offset points',  # Use offset points for xytext
                   ha='right',  # Horizontal alignment of the text
                   va='bottom',  # Vertical alignment of the text
                   )

for i in range(len(to_graph_x_g) - 1):
    dx = to_graph_x[i + 1] - to_graph_x[i]  # Change in x
    dy = to_graph_y[i + 1] - to_graph_y[i]  # Change in y
    ax[0].quiver(to_graph_x[i], to_graph_y[i], dx, dy, angles='xy',
                 scale_units='xy', scale=1, color='blue', width=0.006)

set_ax(ax[0])

ax[1].plot(graphing_interval_g, g(graphing_interval_g),
           label='$g(x)$', linewidth=2)
ax[1].plot(graphing_interval_g, graphing_interval_g, label='$x$', linewidth=2)
ax[1].plot(to_graph_x_g, to_graph_y_g, 'o', color='red')

ax[1].annotate('$a$', (start_x_g, start_x_g), fontsize=20,
               textcoords='offset points',  # Use offset points for xytext
               xytext=(2, -2),
               ha='left',  # Horizontal alignment of the text
               va='top',  # Vertical alignment of the text
               )

# The arrows
for i in range(len(to_graph_x_g) - 1):
    dx = to_graph_x_g[i + 1] - to_graph_x_g[i]  # Change in x
    dy = to_graph_y_g[i + 1] - to_graph_y_g[i]  # Change in y
    ax[1].quiver(to_graph_x_g[i], to_graph_y_g[i], dx, dy, angles='xy',
                 scale_units='xy', scale=1, color='blue', width=0.006)

for i in range(len(distinct_y)-1):
    ax[1].annotate(f'$g^{i+1}(a)$', (distinct_y_g[i], distinct_y_g[i]), fontsize=20,
                   textcoords='offset points',  # Use offset points for xytext
                   xytext=(2, -2),
                   ha='left',  # Horizontal alignment of the text
                   va='top',  # Vertical alignment of the text
                   )

set_ax(ax[1])
plt.savefig("stable_and_unstable_fixed_point.png")
plt.tight_layout()
