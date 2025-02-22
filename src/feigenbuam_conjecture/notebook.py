# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import sympy as sp 
import numpy as np 
import matplotlib.pyplot as plt
from sympy import symbols, Poly, simplify, collect, Eq
from scipy.optimize import fsolve 

sp.init_printing()

# +
alpha = symbols(r'\alpha', positive=True) 
x = symbols('x')

# n is number of b_i
# there is always 1 alpha, denoted as alpha, and n b_i
n = 6
b = symbols(f'b1:{n+1}') 

p_x = 1 + sum(b[i] * x**(2*i+2) for i in range(n))

# print("p(x) =")
# display(p_x)

p_neg_x_a = p_x.subs(x, -x / alpha)  # p(-x/a)
p_p_neg_x_a = p_x.subs(x, p_neg_x_a) # p(p(-x/a))
T_p_x = simplify( -alpha * p_p_neg_x_a)  # T(p(x)) = -a * p(p(-x/a))

T_poly = Poly(T_p_x, x) 
coeff_dict = T_poly.as_dict()

if n < 4:
    print("Coefficients of transformed polynomial:")
    for exp in sorted(coeff_dict.keys()):
        coeff = coeff_dict[exp]
        print(f"x^{exp}: {coeff}")

# +
eqs_sympy = [Eq(coeff_dict[(0,)], 1)]
for i in range(1, n+1):
    eqs_sympy.append(Eq(coeff_dict[(2*i,)], b[i-1]))

display(eqs_sympy)


# -

def eqs_float(input_array):
    """
    Defineds a R^(n+1) -> R^(n+1) function 
    the variable n is defined in the previous cell
    the input shall be [a, b1, b2, ..., bn]

    The output is a list of n + 1 floats, which are the values of each of the 
    terms in defined below evaluated at input_array

    there are n + 1 variables, a, b1, b2, ..., bn 
    which is the taylor expansion of the even polynomial 
    f = 1 + b1 x^2 + b2 x^4 + ... + bn x^(2n)
    the function f has the property that phi(f)(x) = -a f(-x/a) = f(x)
    where phi is a function operator and a is feigenbuam's constant alpha

    the above cell uses sympy to calculate phi(f)(x) as a polynomial
    p = c_0 + c_2 x^2 + c_4 x^4 + ... + c_2n x^(2n*2n) 
    We disregard all the terms of degree higher than 2n,
    and equating the coefficients of the preceeding terms to the corresponding
    one of the taylor series of f(x), so hereby we get n+1 equations 

    c_0(a, b_1, ...) - 1 
    c_2(a, b_1, ...) - b1 
    c_4(a, b_1, ...) - b2
    ...

    This function evaluates the above equations at the input_array
    """

    # assert len(input_array) == n+1 
    res = []

    # the constant term 
    tmp = coeff_dict[(0,)] - 1
    tmp = tmp.subs(alpha, input_array[0])
    for i in range(1, n+1):
        tmp = tmp.subs(b[i-1], input_array[i])
    res.append(tmp.evalf())

    # the rest
    for i in range(1, n+1): 
        tmp = coeff_dict[(2*i,)] - b [i - 1]
        tmp = tmp.subs(alpha, input_array[0])
        for i in range(1, n+1):
            tmp = tmp.subs(b[i-1], input_array[i])
        res.append(tmp.evalf())
    
    return res


# testing cell
eqs_float([1] * (n + 1))

# +
# our guesses 
alpha_val = 2.5029 
b_list = [-1.52763, 0.10482, -0.02671]
input_array_guess_4 = [alpha_val] + b_list

guess = []
if n <= 3:
    guess = input_array_guess_4[:n+1]
else:
    guess = input_array_guess_4 + [0.01] * (n - 3)
    
print("Initial guess:")
print(guess)

# +
root = fsolve(eqs_float, guess) 

root

# +
# evaluate 
solution = sp.solve(eqs)
display(solution)

print(solution[0][a].evalf())
# -


