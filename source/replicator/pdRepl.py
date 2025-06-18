import sympy as sp
from sympy import latex
from sympy.utilities.lambdify import lambdify
import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd



A = sp.Matrix([[3, 0],
               [5, 1]])


x = sp.symbols("x")
y = 1 - x

payoffC = 3 * x
payoffD = 5 * x + y

x_dot = (0.9 / 5) * (x * y * (payoffC - payoffD))

fixed_points = sp.solve([x_dot], (x), dict=True)

t = sp.symbols("t")

f = lambdify((t, x), [x_dot], modules="numpy")

def replicatorSystem(t, vars):
    x = vars
    dxdt = f(t,x)
    return [dxdt]

x0 = [0.9]
t_span = (0, 35)
t_eval = np.linspace(*t_span, 35)

sol = solve_ivp(replicatorSystem, t_span, x0, t_eval=t_eval)

x_vals = sol.y[0]
y_vals = 1 - x_vals

df_PD = pd.DataFrame({
    "C": x_vals,
    "D": y_vals
})


def pdNumerical():
    return df_PD