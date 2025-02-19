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
