# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# +
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Poly, simplify, collect, Eq

sp.init_printing()

# +
a = symbols("a", positive=True)
x = symbols("x")

n = 3
b = symbols(f"b1:{n+1}")

p_x = 1 + sum(b[i] * x ** (2 * i + 2) for i in range(n))

# print("p(x) =")
# display(p_x)

p_neg_x_a = p_x.subs(x, -x / a)  # p(-x/a)
p_p_neg_x_a = p_x.subs(x, p_neg_x_a)  # p(p(-x/a))
T_p_x = simplify(-a * p_p_neg_x_a)  # T(p(x)) = -a * p(p(-x/a))

T_poly = Poly(T_p_x, x)
coeff_dict = T_poly.as_dict()

print("Coefficients of transformed polynomial:")
for exp in sorted(coeff_dict.keys()):
    coeff = coeff_dict[exp]
    print(f"x^{exp}: {coeff}")

# +
eqs = [Eq(coeff_dict[(0,)], 1)]
for i in range(1, n + 1):
    eqs.append(Eq(coeff_dict[(2 * i,)], b[i - 1]))

display(eqs)
# solve for
# coefficients


# +
# evaluate
solution = sp.solve(eqs)
display(solution)

print(solution[0][a].evalf())
# -
