import numpy as np
import time


def logistic(la, x):
    return 4 * la * x * (1 - x)


def tent(la, x):
    if x < 0:
        x = -x
    decimals = 2*x - int(2*x)
    if int(2 * x) % 2 == 0:
        return la * decimals
    else:
        return la * (1 - decimals)


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
            try:
                val = func(r, val)
            except (ZeroDivisionError, ValueError):
                print("Error in func")
                print(f"r = {r}, val = {val}")

            if abs(val) > 1e4:
                return []

        # ignore x_500, recording values from x_501
        for _ in range(plot_times):
            try:
                val = func(r, val)
            except (ZeroDivisionError, ValueError):
                print("Error in func")
                print(f"r = {r}, val = {val}")
            if abs(val) > 1e4:
                return []
            res.append(val)

    return res


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


def timeit(func):
    """Decorator to time a function and print the execution time."""

    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # Start the timer
        result = func(*args, **kwargs)  # Run the function
        end_time = time.perf_counter()  # End the timer
        elapsed_time = end_time - start_time  # Calculate elapsed time

        GREEN = "\033[92m"
        RED = "\033[91m"
        BOLD = "\033[1m"
        RESET = "\033[0m"

        print(
            f"Function {GREEN}{func.__name__}{RESET} took {
                BOLD}{RED}{elapsed_time:.6f}{RESET} seconds"
        )
        return result  # Return the original function's result

    return wrapper
