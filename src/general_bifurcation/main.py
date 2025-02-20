import sys
import numpy as np
import matplotlib.pyplot as plt

from utils import iterate_and_record_all_x_0, logistic


def graph_bifurcation(
    func,
    ax,
    r_low=0,
    r_high=0,
    n_of_r=400,
    prep_times=100,
    plot_times=100,
    plot_name="bifurcation.png",
    graph_line=None,
    line_attr=None,  # dictionary
    random_x_0=True,
    x_low=0,
    x_high=1,
    repetitions=5,
    alpha=0.5,
    dot_size=0.3,
    finer_tune=False,
    stage_steps=[0.3, 0.4],
    alpha_list=[0.5, 0.3, 0.2],
    prep_times_list=[100, 200, 300],
    plot_times_list=[100, 200, 250],
    repetitions_list=[5, 10, 10],
    dot_size_list=[0.2, 0.1, 0.1],
):
    """
    Produce the graph of bifurcation for a given function in the open interval.
    Producing such a graph involves rather tedious process of
    fine tuning various parameters, all of which are included in the
    function parameters

    If the finer_tune is true, the function will use the lists of parameters
    provided in alpha_list, prep_times_list, plot_times_list, repetitions_list
    after r increase reaches the values in stage_steps.

    For example
    if stage_steps = [0.3, 0.4]
    alpha_list = [0.5, 0.3, 0.2] , etc
    at r  = 0.1,
        stage = 0
        r = 0.5
    at r = 0.35
        stage = 1
        r = 0.3
    at r = 0.45
        stage = 2
        r = 0.45


    """
    # Error checking
    if finer_tune:
        assert len(stage_steps) + 1 == len(alpha_list)
        assert len(stage_steps) + 1 == len(prep_times_list)
        assert len(stage_steps) + 1 == len(plot_times_list)
        assert len(stage_steps) + 1 == len(repetitions_list)
        assert len(stage_steps) + 1 == len(dot_size_list)

    use_line_attr = {"color": "r", "linestyle": "--",
                     "linewidth": 0.5, "alpha": 0.5,
                     "label": "line"}
    if line_attr is not None:
        for i in line_attr.keys():
            use_line_attr[i] = line_attr[i]

    def get_stage(list_r, val):
        """
        list_r: increasing list of r
        r: shall be smaller than or equal to the largest element in list_r

        return:
            an index of list_r such that list_r[index] <= val < list_r[index+1]

        list_r = [1,2,40]
        r = 0.1 -> return 0
        r = 1.1 -> return 1
        r= 2.1 -> return 2
        r = 40.1 -> return 3
        """
        for i in range(len(list_r)):
            if list_r[i] > val:
                return i
        return len(list_r)

    r_vals = np.linspace(r_low, r_high, n_of_r)

    cur_stage = -1
    for r in r_vals:
        # variable line size as for r <3.5 there are very few paths
        if finer_tune:
            stage = get_stage(stage_steps, r)
            if stage != cur_stage:
                print(f"stage changed to {stage}!")
                cur_stage = stage
                prep_times = prep_times_list[stage]
                plot_times = plot_times_list[stage]
                repetitions = repetitions_list[stage]
                dot_size = dot_size_list[stage]
                alpha = alpha_list[stage]
                print(f"current stage: {stage}")
                print(f"prep_times: {prep_times}")
                print(f"plot_times: {plot_times}")
                print(f"repetitions: {repetitions}")
                print(f"dot_size: {dot_size}")
                print(f"alpha: {alpha}")

        x = iterate_and_record_all_x_0(
            func,
            r,
            prep_times,
            plot_times,
            random_x_0=random_x_0,
            x_low=x_low,
            x_high=x_high,
            repetitions=repetitions,
        )

        r_dummy = np.linspace(r, r, len(x))
        ax.scatter(
            r_dummy,
            x,
            c="tab:blue",
            s=dot_size,
            alpha=alpha,
            edgecolors="none",
            facecolors="tab:blue",
            marker="o",
        )

    if graph_line is not None:
        x_vals = np.linspace(x_low, x_high, 1000)
        y_vals = [graph_line(x) for x in x_vals]
        ax.plot(x_vals, y_vals, color=use_line_attr["color"],
                linestyle=use_line_attr["linestyle"],
                linewidth=use_line_attr["linewidth"],
                alpha=use_line_attr["alpha"],
                label=use_line_attr["label"],
                )
    ax.legend()

    ax.set_xlabel("$\lambda$", fontsize=15)
    ax.set_ylabel("x", fontsize=15)

    ax.set_xlim(r_low, r_high)
    # ax.tight_layout()
    # plt.show()
    # plt.axis('off')
    # plt.gca().set_position([0, 0, 1, 1])

    return ax


def sin_bi(la, x):
    return la * np.sin(2 * np.pi * x)


fig, ax = plt.subplots(1, 1, figsize=(10, 10))

graph_bifurcation(
    sin_bi,
    ax,
    r_low=0,
    r_high=1,
    n_of_r=100,
    graph_line=lambda x: 0.25,
    line_attr={"label": "$y = 0.25$"},
    plot_name="sin_bifurcation.png",
    finer_tune=True,
    stage_steps=[0.2, 0.4],
    alpha_list=[0.8, 0.5, 0.3],
    prep_times_list=[200, 80, 100],
    plot_times_list=[30, 50, 150],
    repetitions_list=[1, 4, 7],
    dot_size_list=[10.2, 0.12, 0.1],
)

plt.show()
