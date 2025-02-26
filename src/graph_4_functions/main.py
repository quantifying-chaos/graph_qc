import numpy as np
import matplotlib.pyplot as plt
from utils import tent


def sin_c(c, x):
    return np.sin(2 * np.pi * x) + c


def sin_bi(la, x):
    return la * np.sin(2 * np.pi * x)


def xxtimes_one_minus_x(c, x):
    return c * x * (1 - x) * (1 - x)


def xlog(c, x):
    if (x == 0).any():
        return 0
    return -1 * c * np.log(x) * x


def l_exp(c, x):
    return c * np.exp(x)


def c_arctan(c, x):
    return c * np.arctan(x)


def arctan_c(c, x):
    return np.arctan(c*x)


functions = [sin_c, sin_bi, xxtimes_one_minus_x,
             xlog, l_exp, tent, c_arctan, arctan_c]
function_names = [r"$\sin(2\pi x) + \lambda$", r"$\lambda \sin(2\pi x)$",
                  r"$\lambda x(1-x)(1-x)$", r"$-\lambda x\log(x)$",
                  r"$\lambda e^x$", r"$\text{tent}(x)$",
                  r"$\lambda \arctan(x)$", r"$\arctan(\lambda x)$"
                  ]

lambda_vals = np.linspace(-2, 2, 5)
x = np.linspace(-0.2, 1.2, 100)

fig, ax = plt.subplots(4, 2, figsize=(14, 20))
for i in range(len(functions)):
    row = i // 2
    col = i - row * 2
    for lambda_val in lambda_vals:
        y = [functions[i](lambda_val, x_val) for x_val in x]
        ax[row][col].plot(x, y,
                          label=fr"$\lambda$ =  {lambda_val}")
        ax[row][col].set_title(function_names[i], fontsize=22)
        ax[row][col].legend(fontsize=18)
        ax[row][col].set_xlabel("x", fontsize=18)
        ax[row][col].set_ylabel("y", fontsize=18)
        # set x, y ticks size
        ax[row][col].tick_params(axis="both", labelsize=18)
# plt.show()
plt.tight_layout()
plt.savefig("combined_functions.png")
