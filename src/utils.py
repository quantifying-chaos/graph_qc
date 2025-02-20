import numpy as np


def logistic(la, x):
    return 4 * la * x * (1 - x)


def iterate(fn, parameter, input, times):
    """
    fn shall be a function of two inputs:
        fn(parameter, x_0)
    """
    x_0 = input
    for i in range(times):
        x_0 = fn(parameter, x_0)

    return x_0


def iterate_and_record_all_x_0(
    func,
    r,
    prep_times,
    plot_times,
    x_0=0.5,
    random_x_0=True,
    x_high=1,
    x_low=0,
    repetitions=5,
):
    """
    Iterate the func with initial value x_0 for r = r
    x_1 = func(x_0,r), etc.

    func shall have signature func(r, x) -> x, simliar to logistic function

    x_0 to x_{prep_times-1} are ignored.
    x_{prep_times} to x_{prep_times+plot_times-1} are recorded in res
    and returned

    If x_i diverges, return empty list

    If random_x is setted to true, a random_x x is choosen
    from rand_lower to rand_upper
    """

    res = []
    for i in range(repetitions):
        val = x_0
        if random_x_0:
            val = np.random.uniform(0, 1)

        for _ in range(prep_times):
            val = func(r, val)
            if abs(val) > 1e4:
                return []

        # ignore x_500, recording values from x_501
        for _ in range(plot_times):
            val = func(r, val)
            if abs(val) > 1e4:
                return []
            res.append(val)

    return res
