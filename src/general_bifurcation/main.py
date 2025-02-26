import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os

from utils import iterate_and_record_all_x_0, get_stage, timeit, logistic, tent


@timeit
def cal_bifurcation_data(
    func=None,
    r_low=0,
    r_high=0,
    n_of_r=400,
    prep_times=100,
    plot_times=100,
    random_x_0=True,
    x_low_bound=0,
    x_high_bound=1,
    repetitions=5,
    save_to_file=None,
    finer_tune=False,
    stage_steps=None,
    prep_times_list=None,
    plot_times_list=None,
    repetitions_list=None,
    config=None,
):
    if config is not None:
        func = config.get("func")
        r_low = config.get("r_low")
        r_high = config.get("r_high")
        n_of_r = config.get("n_of_r")
        prep_times = config.get("prep_times")
        plot_times = config.get("plot_times")
        random_x_0 = config.get("random_x_0")
        x_low_bound = config.get("x_low_bound")
        x_high_bound = config.get("x_high_bound")
        repetitions = config.get("repetitions")
        save_to_file = config.get("file_name") + ".pkl"
        finer_tune = config.get("finer_tune")
        stage_steps = config.get("stage_steps")
        prep_times_list = config.get("prep_times_list")
        plot_times_list = config.get("plot_times_list")
        repetitions_list = config.get("repetitions_list")

    if finer_tune:
        if stage_steps is None:
            print("stage_steps must be provided in finer_tune mode")
            sys.exit(1)
        assert len(stage_steps) + 1 == len(prep_times_list)
        assert len(stage_steps) + 1 == len(plot_times_list)
        assert len(stage_steps) + 1 == len(repetitions_list)

    r_vals = np.linspace(r_low, r_high, n_of_r)

    y_vals = []
    cur_stage = -1

    desired_len = 0
    if finer_tune is None or not finer_tune:
        desired_len = repetitions * plot_times
    else:
        desired_len = max(repetitions_list) * max(plot_times_list)

    for r in r_vals:
        if finer_tune:
            stage = get_stage(stage_steps, r)
            if stage != cur_stage:
                cur_stage = stage
                prep_times = prep_times_list[stage]
                plot_times = plot_times_list[stage]
                repetitions = repetitions_list[stage]

        padded = iterate_and_record_all_x_0(
            func,
            r,
            prep_times,
            plot_times,
            random_x_0=random_x_0,
            x_low=x_low_bound,
            x_high=x_high_bound,
            repetitions=repetitions,
        )
        padded = padded + [np.nan] * (desired_len - len(padded))

        y_vals.append(padded)

    dict_t = {key: val for key, val in zip(r_vals, y_vals)}
    df = pd.DataFrame(dict_t)

    if save_to_file is not None:
        df.to_pickle(save_to_file)

    return df


@timeit
def graph_bifurcation(
    ax,
    func=None,
    read_from_file=None,
    r_low=0,
    r_high=0,
    n_of_r=400,
    prep_times=100,
    plot_times=100,
    graph_line=None,
    line_attr=None,  # dictionary
    random_x_0=True,
    x_low_bound=0,
    x_high_bound=1,
    repetitions=5,
    alpha=0.5,
    dot_size=0.3,
    finer_tune=False,
    stage_steps=None,
    alpha_list=None,
    prep_times_list=None,
    plot_times_list=None,
    repetitions_list=None,
    dot_size_list=None,
    no_vertical_label=False,
    config=None,  # a dictionary overwrites all the above parameters
):
    """
    Produce the graph of bifurcation for a given function in the open interval.
    Producing such a graph involves rather tedious process of
    fine tuning various parameters, all of which are included in the
    function parameters

    The provided function must have the
    signature f(parameter, x) like logistic in utils.py

    If no function is provided,
    will read the data from the file provided in read_from_file

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

    if config is not None:
        func = config.get("func")
        r_low = config.get("r_low")
        r_high = config.get("r_high")
        n_of_r = config.get("n_of_r")
        prep_times = config.get("prep_times")
        plot_times = config.get("plot_times")
        random_x_0 = config.get("random_x_0")
        x_low_bound = config.get("x_low_bound")
        x_high_bound = config.get("x_high_bound")
        repetitions = config.get("repetitions")
        read_from_file = config.get("file_name") + ".pkl"
        alpha = config.get("alpha")
        dot_size = config.get("dot_size")
        finer_tune = config.get("finer_tune")
        stage_steps = config.get("stage_steps")
        alpha_list = config.get("alpha_list")
        prep_times_list = config.get("prep_times_list")
        plot_times_list = config.get("plot_times_list")
        repetitions_list = config.get("repetitions_list")
        dot_size_list = config.get("dot_size_list")
        graph_line = config.get("graph_line")
        line_attr = config.get("line_attr")
        no_vertical_label = config.get("no_vertical_label")

    if func is None and read_from_file is None:
        print(
            "Either func or read_from_file must be provided in function graph_bifurcation"
        )
        sys.exit(1)

    # Error checking
    if finer_tune:
        assert len(stage_steps) + 1 == len(alpha_list)
        if prep_times_list is not None:
            assert len(stage_steps) + 1 == len(prep_times_list)
        if plot_times_list is not None:
            assert len(stage_steps) + 1 == len(plot_times_list)
        if repetitions_list is not None:
            assert len(stage_steps) + 1 == len(repetitions_list)
        assert len(stage_steps) + 1 == len(dot_size_list)

    use_line_attr = {
        "color": "r",
        "linestyle": "--",
        "linewidth": 1,
        "alpha": 0.5,
        "label": "line",
    }
    if line_attr is not None:
        for i in line_attr.keys():
            use_line_attr[i] = line_attr[i]

    if read_from_file is None:
        df = cal_bifurcation_data(
            func,
            r_low=r_low,
            r_high=r_high,
            n_of_r=n_of_r,
            prep_times=prep_times,
            plot_times=plot_times,
            random_x_0=random_x_0,
            x_low_bound=x_low_bound,
            x_high_bound=x_high_bound,
            repetitions=repetitions,
        )
    else:
        # check if file exists
        if not os.path.exists(read_from_file):
            print(f"File {read_from_file} does not exist")
            print("Have you run the cal_bifurcation_data function?")
            sys.exit(1)

        df = pd.read_pickle(read_from_file)

    r_vals = df.columns.tolist()
    r_low = min(r_vals)
    r_high = max(r_vals)
    cur_stage = -1
    for r in r_vals:
        # variable line size as for r <3.5 there are very few paths
        if finer_tune:
            stage = get_stage(stage_steps, r)
            if stage != cur_stage:
                cur_stage = stage
                dot_size = dot_size_list[stage]
                alpha = alpha_list[stage]

        x = df[r].values.tolist()
        # filtered out nan
        x = [i for i in x if not np.isnan(i)]

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
        x_vals = np.linspace(r_low, r_high, 1000)
        y_vals = [graph_line(x) for x in x_vals]
        ax.plot(
            x_vals,
            y_vals,
            color=use_line_attr["color"],
            linestyle=use_line_attr["linestyle"],
            linewidth=use_line_attr["linewidth"],
            alpha=use_line_attr["alpha"],
            label=use_line_attr["label"],
        )
        # only the line needs the legend
        ax.legend(fontsize=18, loc="upper left")

    ax.set_xlabel(r"$\lambda$", fontsize=18)

    if not no_vertical_label:
        ax.set_ylabel("$x$", fontsize=18)

    ax.set_xlim(r_low, r_high)
    ax.tick_params(axis="both", which="major", labelsize=18)

    return ax


###############################
# START OF GRAPHING
###############################

#######
# normal sin(x) bifurcation
#######
def sin_bi(la, x):
    return la * np.sin(2 * np.pi * x)


sin_whole = {
    "func": sin_bi,
    "file_name": "sin_bifurcation_whole",
    "r_low": 0,
    "r_high": 1,
    "n_of_r": 1500,
    "prep_times": None,
    "plot_times": None,
    "random_x_0": True,
    "x_low_bound": 0,
    "x_high_bound": 1,
    "repetitions": None,
    "alpha": None,
    "dot_size": None,
    "finer_tune": True,
    "stage_steps": [0.4],
    "prep_times_list": [200, 100],
    "plot_times_list": [15, 150],
    "repetitions_list": [4, 7],
    "alpha_list": [0.7, 0.4],
    "dot_size_list": [0.8, 0.07],
    "graph_line": lambda x: 0.25,
    "line_attr": {"label": "$x = 0.25$"},
}

sin_details = {
    "func": sin_bi,
    "r_low": 0.3,
    "r_high": 0.47,
    "n_of_r": 1500,
    "prep_times": None,
    "plot_times": None,
    "random_x_0": True,
    "x_low_bound": 0,
    "x_high_bound": 1,
    "repetitions": None,
    "alpha": None,
    "dot_size": None,
    "finer_tune": True,
    "stage_steps": [0.4],
    "prep_times_list": [300, 100],
    "plot_times_list": [10, 150],
    "repetitions_list": [4, 7],
    "file_name": "sin_bifurcation_03-047",
    "alpha_list": [0.6, 0.4],
    "dot_size_list": [0.6, 0.07],
    "graph_line": lambda x: 0.25,
    "line_attr": {"label": "$x = 0.25$"},
    "no_vertical_label": True,
}

# cal_bifurcation_data(
#     config=sin_02_05,
# )
#
# cal_bifurcation_data(
#     config=sin_whole,
# )
#
# fig, ax = plt.subplots(1, 2, figsize=(14, 7))
#
# graph_bifurcation(
#     ax[0],
#     config=sin_whole,
# )
# graph_bifurcation(
#     ax[1],
#     config=sin_details,
# )
# plt.tight_layout()
#
# plt.savefig("sin_bifurcation_combined.png")


def sin_c(c, x):
    return np.sin(2 * np.pi * x) + c


sin_c_whole_config = {
    "func": sin_c,
    "file_name": "sin_c_bifurcation_whole",
    "r_low": -1,
    "r_high": 1,
    "n_of_r": 1500,
    "dot_size": None,
    "random_x_0": True,
    "x_low_bound": -1,
    "x_high_bound": 1,
    "finer_tune": True,
    "stage_steps": [0.4],
    "prep_times_list": [100, 100],
    "plot_times_list": [150, 150],
    "repetitions_list": [7, 7],
    "alpha_list": [0.3, 0.4],
    "dot_size_list": [0.07, 0.07],
    "graph_line": lambda x: 0.25,
    "line_attr": {"label": "$x = 0.25$"},
    "prep_times": None,
    "plot_times": None,
    "repetitions": None,
    "alpha": None,
}

sin_c_detail_config = {
    "func": sin_c,
    "file_name": "sin_c_bifurcation_detail",
    "r_low": 0.25,
    "r_high": 0.35,
    "n_of_r": 2500,
    "dot_size": None,
    "random_x_0": True,
    "x_low_bound": -1,
    "x_high_bound": 1,
    "finer_tune": True,
    "stage_steps": [0.3],
    "prep_times_list": [250, 100],
    "plot_times_list": [20, 150],
    "repetitions_list": [4, 10],
    "alpha_list": [0.8, 0.6],
    "dot_size_list": [0.80, 0.02],
    "graph_line": lambda x: 1.25,
    "line_attr": {"label": "$x = 1.25$"},
    "prep_times": None,
    "plot_times": None,
    "repetitions": None,
    "alpha": None,
    "no_vertical_label": True,
}

# cal_bifurcation_data(
#     config=sin_c_whole_config,
# )
#
# cal_bifurcation_data(
#     config=sin_c_detail_config,
# )
#
# fig, ax = plt.subplots(1, 2, figsize=(14, 7))
#
# graph_bifurcation(
#     ax[0],
#     config=sin_c_whole_config,
# )
# graph_bifurcation(
#     ax[1],
#     config=sin_c_detail_config,
# )
# plt.tight_layout()
#
# plt.savefig("sin_c_try.png")


def l_exp(c, x):
    return c * np.exp(x)


l_exp_config = {
    "func": l_exp,
    "file_name": "l_exp",
    "r_low": -20,
    "r_high": 1,
    "n_of_r": 1500,
    "dot_size": 1,
    "random_x_0": True,
    "x_low_bound": -10,
    "x_high_bound": 1,
    "finer_tune": False,
    "stage_steps": [-2.8],
    "prep_times_list": [200, 500],
    "plot_times_list": [10, 10],
    "repetitions_list": [1, 1],
    "alpha_list": [0.8, 0.8],
    "dot_size_list": [0.8, 0.8],
    # "graph_line": lambda x: 0.25,
    # "line_attr": {"label": "$x = 0.25$"},
    "prep_times": 200,
    "plot_times": 2,
    "repetitions": 1,
    "alpha": 1,
}

l_exp_config_zoomed_in = {
    "func": l_exp,
    "file_name": "l_exp_zoomed_in",
    "r_low": -3,
    "r_high": -2.5,
    "n_of_r": 1500,
    "dot_size": 1,
    "random_x_0": True,
    "x_low_bound": -10,
    "x_high_bound": 1,
    "finer_tune": True,
    "stage_steps": [-2.8],
    "prep_times_list": [200, 500],
    "plot_times_list": [10, 10],
    "repetitions_list": [1, 1],
    "alpha_list": [0.8, 0.8],
    "dot_size_list": [0.8, 0.8],
    # "graph_line": lambda x: 0.25,
    # "line_attr": {"label": "$x = 0.25$"},
    "prep_times": 100,
    "plot_times": 300,
    "repetitions": 3,
    "alpha": 0.6,
}

# cal_bifurcation_data(
#     config=l_exp_config,
# )
# cal_bifurcation_data(
#     config=l_exp_config_zoomed_in,
# )
#
# fig, ax = plt.subplots(1, 2, figsize=(14, 7))
#
# graph_bifurcation(
#     ax[0],
#     config=l_exp_config,
# )
# graph_bifurcation(
#     ax[1],
#     config=l_exp_config_zoomed_in,
# )
# plt.tight_layout()
# # plt.show()
# plt.savefig("l_exp.png")


# one sided stable point
# Simply shifting, no stretching
def exp_l(c, x):
    return np.exp(x) + c


exp_l_config = {
    "func": exp_l,
    "file_name": "exp_l",
    "r_low": -100,
    "r_high": 20,
    "n_of_r": 1500,
    "dot_size": 1,
    "random_x_0": True,
    "x_low_bound": -10,
    "x_high_bound": 1,
    "finer_tune": False,
    "stage_steps": [-2.8],
    "prep_times_list": [200, 500],
    "plot_times_list": [10, 10],
    "repetitions_list": [1, 1],
    "alpha_list": [0.8, 0.8],
    "dot_size_list": [0.8, 0.8],
    # "graph_line": lambda x: 0.25,
    # "line_attr": {"label": "$x = 0.25$"},
    "prep_times": 200,
    "plot_times": 2,
    "repetitions": 1,
    "alpha": 1,
}

exp_l_config_zoomed_in = {
    "func": exp_l,
    "file_name": "exp_l_zoomed_in",
    "r_low": -1.5,
    "r_high": -0.8,
    "n_of_r": 1500,
    "dot_size": 1,
    "random_x_0": True,
    "x_low_bound": -0.5,
    "x_high_bound": -0.1,
    "finer_tune": False,
    "stage_steps": [-2.8],
    "prep_times_list": [200, 500],
    "plot_times_list": [10, 10],
    "repetitions_list": [1, 1],
    "alpha_list": [0.8, 0.8],
    "dot_size_list": [0.8, 0.8],
    # "graph_line": lambda x: 0.25,
    # "line_attr": {"label": "$x = 0.25$"},
    "prep_times": 100,
    "plot_times": 300,
    "repetitions": 3,
    "alpha": 0.6,
}

# cal_bifurcation_data(
#     config=exp_l_config,
# )
# cal_bifurcation_data(
#     config=exp_l_config_zoomed_in,
# )
#
# fig, ax = plt.subplots(1, 2, figsize=(14, 7))
#
# graph_bifurcation(
#     ax[0],
#     config=exp_l_config,
# )
# graph_bifurcation(
#     ax[1],
#     config=exp_l_config_zoomed_in,
# )
# plt.tight_layout()
# # plt.show()
# plt.savefig("exp_l.png")


logistic_config = {
    "func": logistic,
    "file_name": "logistic_overview",
    "r_low": 0,
    "r_high": 1,
    "n_of_r": 2500,
    "dot_size": 0.1,
    "random_x_0": True,
    "x_low_bound": 0,
    "x_high_bound": 1,
    "finer_tune": True,
    "stage_steps": [0.875],
    "prep_times_list": [100, 200],
    "plot_times_list": [20, 500],
    "repetitions_list": [4, 4],
    "alpha_list": [0.8, 0.5],
    "dot_size_list": [0.1, 0.02],
    "graph_line": lambda x: 0.5,
    "line_attr": {"label": "$x = 0.5$"},
    "prep_times": 200,
    "plot_times": 2,
    "repetitions": 1,
    "alpha": 1,
}

logistic_config_zoom_1 = {
    "func": logistic,
    "file_name": "logistic_overview_zoom_1",
    "r_low": 0.7,
    "r_high": 1,
    "n_of_r": 2500,
    "dot_size": 0.1,
    "random_x_0": True,
    "x_low_bound": 0,
    "x_high_bound": 1,
    "finer_tune": True,
    "stage_steps": [0.875],
    "prep_times_list": [100, 200],
    "plot_times_list": [20, 500],
    "repetitions_list": [4, 4],
    "alpha_list": [0.8, 0.5],
    "dot_size_list": [0.1, 0.02],
    "graph_line": lambda x: 0.5,
    "line_attr": {"label": "$x = 0.5$"},
    "prep_times": 200,
    "plot_times": 2,
    "repetitions": 1,
    "alpha": 1,
}

logistic_config_zoom_2 = {
    "func": logistic,
    "file_name": "logistic_overview_zoom_2",
    "r_low": 0.85,
    "r_high": 0.9,
    "n_of_r": 2500,
    "dot_size": 0.1,
    "random_x_0": True,
    "x_low_bound": 0,
    "x_high_bound": 1,
    "finer_tune": True,
    "stage_steps": [0.875],
    "prep_times_list": [300, 300],
    "plot_times_list": [20, 500],
    "repetitions_list": [4, 4],
    "alpha_list": [0.8, 0.5],
    "dot_size_list": [0.1, 0.02],
    "graph_line": lambda x: 0.5,
    "line_attr": {"label": "$x = 0.5$"},
    "prep_times": 200,
    "plot_times": 2,
    "repetitions": 1,
    "alpha": 1,
}

logistic_config_zoom_3 = {
    "func": logistic,
    "file_name": "logistic_overview_zoom_3",
    "r_low": 0.91,
    "r_high": 0.97,
    "n_of_r": 2500,
    "dot_size": 0.1,
    "random_x_0": True,
    "x_low_bound": 0,
    "x_high_bound": 1,
    "finer_tune": True,
    "stage_steps": [0.875],
    "prep_times_list": [100, 200],
    "plot_times_list": [20, 500],
    "repetitions_list": [4, 4],
    "alpha_list": [0.8, 0.5],
    "dot_size_list": [0.1, 0.02],
    "graph_line": lambda x: 0.5,
    "line_attr": {"label": "$x = 0.5$"},
    "prep_times": 200,
    "plot_times": 2,
    "repetitions": 1,
    "alpha": 1,
}

# cal_bifurcation_data(
#     config=logistic_config,
# )
#
# cal_bifurcation_data(
#     config=logistic_config_zoom_1,
# )

# cal_bifurcation_data(
#     config=logistic_config_zoom_2
# )

# cal_bifurcation_data(
#     config=logistic_config_zoom_3
# )

# fig, ax = plt.subplots(2, 2, figsize=(14, 14))
#
# graph_bifurcation(
#     ax[0][0],
#     config=logistic_config,
# )
#
# graph_bifurcation(
#     ax[0][1],
#     config=logistic_config_zoom_1,
# )
#
# graph_bifurcation(
#     ax[1][0],
#     config=logistic_config_zoom_2,
# )
#
# graph_bifurcation(
#     ax[1][1],
#     config=logistic_config_zoom_3,
# )
#
# plt.tight_layout()
# # plt.show()
# plt.savefig("logistic.png")


# arctan no bifurcation to chaos
def c_arctan(c, x):
    return c * np.arctan(x)


c_arctan_config = {
    "func": c_arctan,
    "file_name": "c_arctan",
    "r_low": -20,
    "r_high": 20,
    "n_of_r": 1500,
    "dot_size": 1,
    "random_x_0": True,
    "x_low_bound": 0,
    "x_high_bound": 1,
    "finer_tune": False,
    "stage_steps": [0.875],
    "prep_times_list": [100, 200],
    "plot_times_list": [20, 500],
    "repetitions_list": [4, 4],
    "alpha_list": [0.8, 0.5],
    "dot_size_list": [0.1, 0.02],
    # "graph_line": lambda x: 0.5,
    # "line_attr": {"label": "$x = 0.5$"},
    "prep_times": 200,
    "plot_times": 200,
    "repetitions": 2,
    "alpha": 1,
}

c_arctan_zoom_config = {
    "func": c_arctan,
    "file_name": "c_arctan_zoom",
    "r_low": -2,
    "r_high": 2,
    "n_of_r": 1500,
    "dot_size": 1,
    "random_x_0": True,
    "x_low_bound": 0,
    "x_high_bound": 1,
    "finer_tune": False,
    "stage_steps": [0.875],
    "prep_times_list": [100, 200],
    "plot_times_list": [20, 500],
    "repetitions_list": [4, 4],
    "alpha_list": [0.8, 0.5],
    "dot_size_list": [0.1, 0.02],
    # "graph_line": lambda x: 0.5,
    # "line_attr": {"label": "$x = 0.5$"},
    "prep_times": 200,
    "plot_times": 200,
    "repetitions": 2,
    "alpha": 1,
}


# cal_bifurcation_data(
#     config=c_arctan_config,
# )
#
# cal_bifurcation_data(
#     config=c_arctan_zoom_config,
# )

# fig, ax = plt.subplots(1, 2, figsize=(14, 7))
# graph_bifurcation(
#     ax[0],
#     config=c_arctan_config,
# )
# graph_bifurcation(
#     ax[1],
#     config=c_arctan_zoom_config,
# )
# plt.tight_layout()
# # plt.show()
# plt.savefig("c_arctan.png")


# no bifurcation to chaos
def arctan_c(c, x):
    return np.arctan(c*x)


arctan_c_config = {
    "func": arctan_c,
    "file_name": "arctan_c",
    "r_low": -20,
    "r_high": 20,
    "n_of_r": 1500,
    "dot_size": 1,
    "random_x_0": True,
    "x_low_bound": 0,
    "x_high_bound": 1,
    "finer_tune": False,
    "stage_steps": [0.875],
    "prep_times_list": [100, 200],
    "plot_times_list": [20, 500],
    "repetitions_list": [4, 4],
    "alpha_list": [0.8, 0.5],
    "dot_size_list": [0.1, 0.02],
    # "graph_line": lambda x: 0.5,
    # "line_attr": {"label": "$x = 0.5$"},
    "prep_times": 200,
    "plot_times": 200,
    "repetitions": 2,
    "alpha": 1,
}

arctan_c_zoom_config = {
    "func": arctan_c,
    "file_name": "arctan_c_zoom",
    "r_low": -2,
    "r_high": 2,
    "n_of_r": 1500,
    "dot_size": 1,
    "random_x_0": True,
    "x_low_bound": 0,
    "x_high_bound": 1,
    "finer_tune": False,
    "stage_steps": [0.875],
    "prep_times_list": [100, 200],
    "plot_times_list": [20, 500],
    "repetitions_list": [4, 4],
    "alpha_list": [0.8, 0.5],
    "dot_size_list": [0.1, 0.02],
    # "graph_line": lambda x: 0.5_c,
    # "line_attr": {"label": "$x = 0.5$"},
    "prep_times": 200,
    "plot_times": 200,
    "repetitions": 2,
    "alpha": 1,
}

# cal_bifurcation_data(
#     config=arctan_c_config,
# )
# cal_bifurcation_data(
#     config=arctan_c_zoom_config,
# )
#
# fig, ax = plt.subplots(1, 2, figsize=(14, 7))
# graph_bifurcation(
#     ax[0],
#     config=arctan_c_config,
# )
# graph_bifurcation(
#     ax[1],
#     config=arctan_c_zoom_config,
# )
# plt.tight_layout()
# # plt.show()
# plt.savefig("arctan_c.png")


def xxtimes_one_minus_x(c, x):
    return c * x * (1 - x) * (1 - x)


xxtimes_one_minus_x_config = {
    "func": xxtimes_one_minus_x,
    "file_name": "xx_one_minus_x",
    "r_low": -3,
    "r_high": 10,
    "n_of_r": 2500,
    "random_x_0": True,
    "x_low_bound": -1,
    "x_high_bound": 1,
    "finer_tune": True,
    "stage_steps": [5],
    "prep_times_list": [100, 100],
    "plot_times_list": [500, 500],
    "repetitions_list": [4, 4],
    "alpha_list": [0.5, 0.5],
    "dot_size_list": [0.02, 0.02],
    "graph_line": lambda x: 0.333,
    "line_attr": {"label": r"$x = \frac{1}{3}$"},
}

xxtimes_one_minus_x_zoom_config = {
    "func": xxtimes_one_minus_x,
    "file_name": "xx_one_minus_x_config",
    "r_low": 3.8,
    "r_high": 5.5,
    "n_of_r": 2500,
    "random_x_0": True,
    "x_low_bound": -1,
    "x_high_bound": 1,
    "finer_tune": True,
    "stage_steps": [5],
    "prep_times_list": [200, 200],
    "plot_times_list": [500, 500],
    "repetitions_list": [4, 4],
    "alpha_list": [0.5, 0.5],
    "dot_size_list": [0.02, 0.02],
    "graph_line": lambda x: 0.333,
    "line_attr": {"label": r"$x = \frac{1}{3}$"},
}

# cal_bifurcation_data(
#     config=xxtimes_one_minus_x_config,
# )
# cal_bifurcation_data(
#     config=xxtimes_one_minus_x_zoom_config,
# )
#
# fig, ax = plt.subplots(1, 2, figsize=(14, 7))
# graph_bifurcation(
#     ax[0],
#     config=xxtimes_one_minus_x_config,
# )
# graph_bifurcation(
#     ax[1],
#     config=xxtimes_one_minus_x_zoom_config,
# )
#
# plt.tight_layout()
# # plt.show()
# plt.savefig("xxtimes_one_minus_x.png")


tent_config = {
    "func": tent,
    "file_name": "tent",
    "r_low": -20,
    "r_high": 20,
    "n_of_r": 1600,
    "random_x_0": True,
    "x_low_bound": -1,
    "x_high_bound": 1,
    "finer_tune": True,
    "stage_steps": [5],
    "prep_times_list": [100, 100],
    "plot_times_list": [500, 500],
    "repetitions_list": [4, 4],
    "alpha_list": [0.3, 0.3],
    "dot_size_list": [0.02, 0.02],
    # "graph_line": lambda x: 0.333,
    # "line_attr": {"label": r"$x = \frac{1}{3}$"},
}

tent_zoom_config = {
    "func": tent,
    "file_name": "tent_zoom",
    "r_low": -1,
    "r_high": 1,
    "n_of_r": 1800,
    "random_x_0": True,
    "x_low_bound": -1,
    "x_high_bound": 1,
    "finer_tune": True,
    "stage_steps": [5],
    "prep_times_list": [200, 200],
    "plot_times_list": [500, 500],
    "repetitions_list": [4, 4],
    "alpha_list": [0.4, 0.4],
    "dot_size_list": [0.02, 0.02],
    # "graph_line": lambda x: 0.333,
    # "line_attr": {"label": r"$x = \frac{1}{3}$"},
}
#
cal_bifurcation_data(
    config=tent_config,
)
cal_bifurcation_data(
    config=tent_zoom_config,
)
# #
# fig, ax = plt.subplots(1, 2, figsize=(14, 7))
# graph_bifurcation(
#     ax[0],
#     config=tent_config,
# )
# graph_bifurcation(
#     ax[1],
#     config=tent_zoom_config,
# )

# plt.tight_layout()
# # # plt.show()
# plt.savefig("tent.png")


def xlog(c, x):
    if x == 0:
        return 0
    return -1 * c * math.log(x) * x


xlog_config = {
    "func": xlog,
    "file_name": "xlog",
    "r_low": 0.001,
    "r_high": np.e,
    "n_of_r": 2600,
    "random_x_0": True,
    "x_low_bound": 0.1,
    "x_high_bound": 1,
    "finer_tune": True,
    "stage_steps": [2.3],
    "prep_times_list": [200, 100],
    "plot_times_list": [20, 600],
    "repetitions_list": [1, 4],
    "alpha_list": [0.8, 0.3],
    "dot_size_list": [0.2, 0.02],
    "graph_line": lambda x: 1/np.e,
    "line_attr": {"label": r"$x = e^{-1}$"},
}

xlog_zoom_config = {
    "func": xlog,
    "file_name": "xlog_zoom",
    "r_low": 2,
    "r_high": np.e,
    "n_of_r": 2800,
    "random_x_0": True,
    "x_low_bound": -1,
    "x_high_bound": 1,
    "finer_tune": True,
    "stage_steps": [2.3],
    "prep_times_list": [200, 100],
    "plot_times_list": [20, 600],
    "repetitions_list": [1, 4],
    "alpha_list": [0.8, 0.3],
    "dot_size_list": [0.2, 0.02],
    "graph_line": lambda x: 1/np.e,
    "line_attr": {"label": r"$x = e^{-1}$"},
}

# NOTE: the great combination!

# cal_bifurcation_data(
#     config=xlog_config,
# )
# cal_bifurcation_data(
#     config=xlog_zoom_config,
# )

# fig, ax = plt.subplots(1, 2, figsize=(14, 7))
# graph_bifurcation(
#     ax[0],
#     config=xlog_config,
# )
# graph_bifurcation(
#     ax[1],
#     config=xlog_zoom_config,
# )
#
# plt.tight_layout()
# # # plt.show()
# plt.savefig("xlog.png")

# plot sin, sin_c, xx_one_minus_x, xlog together
# fig, ax = plt.subplots(4, 2, figsize=(14, 20))
#
# graph_bifurcation(
#     ax[0][0],
#     config=sin_whole,
# )
# ax[0][0].set_title(r"$\lambda \sin(2 \pi x)$", fontsize=22)
# graph_bifurcation(
#     ax[0][1],
#     config=sin_details,
# )
# ax[0][1].set_title(r"$\lambda \sin(2 \pi x)$", fontsize=22)
# graph_bifurcation(
#     ax[1][0],
#     config=sin_c_whole_config,
# )
# ax[1][0].set_title(r"$\lambda + \sin(2 \pi x)$", fontsize=22)
# graph_bifurcation(
#     ax[1][1],
#     config=sin_c_detail_config,
# )
# ax[1][1].set_title(r"$\lambda + \sin(2 \pi x)$", fontsize=22)
# graph_bifurcation(
#     ax[2][0],
#     config=xxtimes_one_minus_x_config,
# )
# ax[2][0].set_title(r"$\lambda x(1-x)(1-x)$", fontsize=22)
# graph_bifurcation(
#     ax[2][1],
#     config=xxtimes_one_minus_x_zoom_config,
# )
# ax[2][1].set_title(r"$\lambda x(1-x)(1-x)$", fontsize=22)
# graph_bifurcation(
#     ax[3][0],
#     config=xlog_config,
# )
# ax[3][0].set_title(r"$- \lambda x \log(x)$", fontsize=22)
# graph_bifurcation(
#     ax[3][1],
#     config=xlog_zoom_config,
# )
# ax[3][1].set_title(r"$ - \lambda x \log(x)$", fontsize=22)
#
# plt.tight_layout()
# # plt.show()
# plt.savefig("combined_bifurcations.png")


# plot l_exp, tent, c_arctan, arctan_c together
fig, ax = plt.subplots(4, 2, figsize=(14, 20))

graph_bifurcation(
    ax[0][0],
    config=l_exp_config,
)
ax[0][0].set_title(r"$\lambda e^x$", fontsize=22)
graph_bifurcation(
    ax[0][1],
    config=l_exp_config_zoomed_in,
)
ax[0][1].set_title(r"$\lambda e^x$", fontsize=22)

graph_bifurcation(
    ax[1][0],
    config=tent_config,
)
ax[1][0].set_title(r"Tent Map", fontsize=22)
graph_bifurcation(
    ax[1][1],
    config=tent_zoom_config,
)
ax[1][1].set_title(r"Tent Map", fontsize=22)

graph_bifurcation(
    ax[2][0],
    config=c_arctan_config,
)
ax[2][0].set_title(r"$\lambda \arctan(x)$", fontsize=22)
graph_bifurcation(
    ax[2][1],
    config=c_arctan_zoom_config,
)
ax[2][1].set_title(r"$\lambda \arctan(x)$", fontsize=22)

graph_bifurcation(
    ax[3][0],
    config=arctan_c_config,
)
ax[3][0].set_title(r"$\arctan(\lambda x)$", fontsize=22)
graph_bifurcation(
    ax[3][1],
    config=arctan_c_zoom_config,
)
ax[3][1].set_title(r"$\arctan(\lambda x)$", fontsize=22)

plt.tight_layout()
# plt.show()
plt.savefig("combined_no_bifurcations.png")
