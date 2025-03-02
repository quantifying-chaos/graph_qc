# This program takes 45 minutes to run on Ryzen 7945 HX
import sympy as sp
from sympy import symbols, Poly
from scipy.optimize import fsolve

sp.init_printing()


def degree(exp, x):
    return Poly(exp, x).degree()


def product_n(exp1, exp2, n, x):
    """
    Return their product with terms with degree at most n (inclusive)
    """
    exp1_l = exp1.as_ordered_terms()
    exp2_l = exp2.as_ordered_terms()
    res = 0
    for i in exp1_l:
        for j in exp2_l:
            if degree(i, x) + degree(j, x) <= n:
                res += i*j
    return res


def exp_n(exp, n, deg, x):
    """
    Return exp^n with terms with degree at most n (inclusive)
    """
    if n == 0:
        return Poly(1, x).as_expr()

    tmp = exp
    for i in range(n - 1):
        tmp = product_n(exp, tmp, deg, x)

    return tmp


alpha = symbols(r'\alpha', positive=True)
x = symbols('x')

# n is number of b_i, which is the coefficient of x^(2i+2)
# there is always 1 alpha, denoted as alpha, and n b_i
n = 15
b = symbols(f'b1:{n+1}')

p_x = 1 + sum(b[i] * x**(2*i+2) for i in range(n))
p_n_a = p_x.subs(x, -x / alpha)
res = 0
for i in range(len(b)):
    res += b[i] * exp_n(p_n_a, 2*(i+1), 2*n, x)

res += 1
res *= -alpha

T_poly = Poly(res, x)
coeff_dict = T_poly.as_dict()
print("Coefficients of transformed polynomial:")
for exp in sorted(coeff_dict.keys()):
    coeff = coeff_dict[exp]
    print(f"x^{exp}")


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
        tmp = coeff_dict[(2*i,)] - b[i - 1]
        tmp = tmp.subs(alpha, input_array[0])
        for i in range(1, n+1):
            tmp = tmp.subs(b[i-1], input_array[i])
        res.append(tmp.evalf())

    return res


# our guesses
alpha_val = 2.5029
b_list = [-1.52763, 0.10482, -0.02671]
input_array_guess_4 = [alpha_val] + b_list

guess = []
if n <= 3:
    guess = input_array_guess_4[:n+1]
else:
    guess = input_array_guess_4 + [0.01] * (n - 3)

root = fsolve(eqs_float, guess)

for i in root:
    print(f"{i:11.80f}")
