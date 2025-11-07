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

def standardPayoffs(matrix):
  payoffA = matrix.row(0)[0] * x + matrix.row(0)[1] * y
  payoffB = matrix.row(1)[0] * x + matrix.row(1)[1] * y
  return payoffA, payoffB

payoffC = 3 * x
payoffD = 5 * x + y


def standardReplicator(payoffs, w, deltaPi):
  x_dot = (x * y * (payoffs[0] - payoffs[1])) * (w / deltaPi)

  return x_dot
  

# HARDCODED VALUES - FIX THIS
x_dot = (0.9 / 5) * (x * y * (payoffC - payoffD))

# Adjusted replicator dynamics - from paper 1
avgPayoff = x * payoffC + y * payoffD
gamma = (1 - 0.9) / 0.9
x_dot_adj = (x * y * (payoffC - payoffD)) * (1 / (gamma + avgPayoff))

fixed_points = sp.solve([x_dot], (x), dict=True)

t = sp.symbols("t")

f = lambdify((t, x), [x_dot], modules="numpy")
g = lambdify((t, x), [x_dot_adj], modules="numpy")

def replicatorSystem(t, vars):
  x = vars
  dxdt = f(t,x)
  return [dxdt]

def adjReplicatorSystem(t, vars):
  x = vars
  dxdt = g(t,x)
  return [dxdt]

x0 = [0.9]
t_span = (0, 35)
t_eval = np.linspace(*t_span, 500)

sol = solve_ivp(replicatorSystem, t_span, x0, t_eval=t_eval)

adjSol = solve_ivp(adjReplicatorSystem, t_span, x0, t_eval=t_eval)

x_vals = sol.y[0]
y_vals = 1 - x_vals

x_adj_vals = adjSol.y[0]
y_adj_vals = 1 - x_adj_vals

df_PD = pd.DataFrame({
  "C": x_vals,
  "D": y_vals
})

df_PD_ADJ = pd.DataFrame({
  "C": x_adj_vals,
  "D": y_adj_vals
})



"""
Computes the numerical trajectory for the standard replicator dynamics which correspond
to the local update microscopic process with the scaling factor as in literature.
"""
def pdNumerical(matrix, w=0.9, initial_dist=[0.9,0.1]):
  # Convert np matrices to sympy ..
  maxi = matrix.max()
  mini = matrix.min()

  deltaPi = maxi - mini

  #print("DeltaPI for PD:" , deltaPi)

  matrix = sp.Matrix(matrix)
  payoffs = standardPayoffs(matrix)
  x_dot = standardReplicator(payoffs, w, deltaPi)

  t = sp.symbols("t")
  f = lambdify((t, x), [x_dot], modules="numpy")

  x0 = [initial_dist[0]]
  t_span = (0, 35)
  t_eval = np.linspace(*t_span, 500)

  def replicatorSystem(t, vars):
    x = vars
    dxdt = f(t,x)
    return [dxdt]

  sol = solve_ivp(replicatorSystem, t_span, x0, t_eval=t_eval)

  x_vals = sol.y[0]
  y_vals = 1 - x_vals

  df_PD = pd.DataFrame({
    "C": x_vals,
    "D": y_vals
  })

  return df_PD

def pdNumericalAdjusted():
  return df_PD_ADJ, t_eval


if __name__ == "__main__":
  print(standardPayoffs(A))