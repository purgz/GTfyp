import sympy as sp
from sympy import latex
from sympy.utilities.lambdify import lambdify
import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
from scipy.integrate import nquad
from scipy.optimize import brentq

from scipy.integrate import tplquad

from scipy.optimize import fsolve

import matplotlib.pyplot as plt


# This file contains the RPS replicator derivation for the 3x3 standard game

a, c, b, gamma, beta = sp.symbols('a c b gamma beta')



    



A = sp.Matrix([[a, c, b, gamma], 
               [b, a, c, gamma], 
               [c, b, a, gamma], 
               [a + beta, a + beta, a + beta, 0]])


# NxN matrix - N-1 replicator equations
# x - fraction of rock, y = fraction of paper, z = fraction of scissors
x = sp.symbols('x')
y = sp.symbols('y')
z = sp.symbols('z')
q = 1 - x - y - z

w_sym = sp.symbols('w')
delta_pi = sp.symbols('delta_pi') # for local update

payoffR = a * x + y * c + b * z + gamma * q
payoffP = b * x + a * y + c * z + gamma * q
payoffS = c * x + b * y + a * z + gamma * q 
payoffL = (a + beta) * x + (a + beta) * y + (a + beta) * z


averagePayoff = payoffR * x + payoffP * y + payoffS * z + payoffL * q

# Pairwise replicator equations 
"""x_dot = (x * y) * (payoffR - payoffP) + (x * z) * (payoffR - payoffS) + (x * q) * (payoffR - payoffL)
y_dot = (y * x) * (payoffP - payoffR) + (y * z) * (payoffP - payoffS) + (y * q) * (payoffP - payoffL)
z_dot = (z * x) * (payoffS - payoffR) + (z * y) * (payoffS - payoffP) + (z * q) * (payoffS - payoffL)
"""
def replicators(matrix, interactionProcess="Moran", w=0.2, local_delta_pi = 2):
  
  # Standard payoffs
  payoffR = a * x + y * c + b * z + gamma * q
  payoffP = b * x + a * y + c * z + gamma * q
  payoffS = c * x + b * y + a * z + gamma * q 
  payoffL = (a + beta) * x + (a + beta) * y + (a + beta) * z

  averagePayoff = payoffR * x + payoffP * y + payoffS * z + payoffL * q

  """
  Other form using avg instead of payoff pairwise comparison - should be equivalent.
  """

  if w is None:
    w = w_sym

  match interactionProcess:

    case "Moran":
      # Adjusted dynamics
      gam = (1 - w) / w
      adjustedScaling = 1 / (gam + averagePayoff)

      x_dot = x * (payoffR - averagePayoff) * adjustedScaling
      y_dot = y * (payoffP - averagePayoff) * adjustedScaling
      z_dot = z * (payoffS - averagePayoff) * adjustedScaling
    case "Local":
    
      k = w / (local_delta_pi)

      x_dot = x * (payoffR - averagePayoff) * k
      y_dot = y * (payoffP - averagePayoff) * k
      z_dot = z * (payoffS - averagePayoff) * k
    case _:
      x_dot = x * (payoffR - averagePayoff) 
      y_dot = y * (payoffP - averagePayoff) 
      z_dot = z * (payoffS - averagePayoff) 


  return x_dot, y_dot, z_dot



def substituteHyperParams(equations, config, variables):
    
  subs = []
  for idx, equation in enumerate(equations):
     subs.append(equation.subs(config))


  return subs
    


def getFixedPoints(subs, variables):
  fixed_points = sp.solve(subs, variables, dict=True)

  return fixed_points



def numericalIntegration(equations, numPoints = 5000, timeSpan = 150, initial_dist = [0.5,0.2,0.2]):


  # Returns a dataframe with trajectory data for numerical solution to replicators

  # Symbol for time
  t = sp.symbols("t")
  # Equations = x_dot, y_dot, z_dot with substituted values for hyper-params
  f = lambdify((t, x , y , z), equations, modules="numpy")

  def replicatorSystem(t, vars):
      x,y,z = vars
      dxdt,dydt,dzdt = f(t,x,y,z)
      return [dxdt, dydt, dzdt]


  x0 = initial_dist
  t_span=(0,timeSpan)
  t_eval=np.linspace(*t_span, numPoints)

  sol = solve_ivp(replicatorSystem, t_span, x0, t_eval=t_eval)


  x_vals = sol.y[0]
  y_vals = sol.y[1]
  z_vals = sol.y[2]
  w_vals = 1 - x_vals - y_vals - z_vals

  df = pd.DataFrame({
      "c1": x_vals,
      "c2": y_vals,
      "c3": z_vals,
      "c4": w_vals
  })

  return df, t_eval


def numericalTrajectory(interactionProcess="Moran", w=0.2, initial_dist=[0.5,0.2,0.2], matrix=None):
  # Runge kutta order 5
  # External module method.
  # Derive replicator equations

  # convert to sympy Matrix for replicators.
  local_delta_pi = 2 # Default value

  if matrix is not None:

    local_delta_pi = matrix.max() - matrix.min()
    matrix = sp.Matrix(matrix)

  x_dot, y_dot, z_dot = replicators(matrix=matrix, interactionProcess=interactionProcess, w=w, local_delta_pi=local_delta_pi)

  #print("Computing numerical solutions to ", x_dot, y_dot, z_dot)
  #print(matrix[0])

  standardConfig = {a: matrix.row(0)[0], b: matrix.row(0)[2], c: matrix.row(0)[1], gamma: matrix.row(0)[3], beta: matrix.row(3)[0]}

  substitutions = substituteHyperParams([x_dot, y_dot, z_dot], standardConfig, (x,y,z))

  df = numericalIntegration(substitutions, initial_dist=initial_dist)

  return df


def findEigenvalues(replicators, config, vars, substitution):
  
  """  J = sp.Matrix()

    # Construct the Jacobian! :)
    for idx, var in enumerate(vars):
      col = []
      for jdx, eq in enumerate(replicators):
        xDiff = sp.diff(eq, var)
        col.append(xDiff)

      J = J.col_insert(idx, sp.Matrix(col))
  """

  #J = sp.Matrix([[sp.diff(eq, var) for var in vars] for eq in replicators])

  x_dot, y_dot, z_dot = replicators

  F_x = sp.diff(x_dot, x)
  F_y = sp.diff(x_dot, y)
  F_z = sp.diff(x_dot, z)

  G_x = sp.diff(y_dot, x)
  G_y = sp.diff(y_dot, y)
  G_z = sp.diff(y_dot, z)

  P_x = sp.diff(z_dot, x)
  P_y = sp.diff(z_dot, y)
  P_z = sp.diff(z_dot, z)

  # 3x3 Jacobian Matrix
  J = sp.Matrix([[F_x, F_y, F_z],
                [G_x, G_y, G_z],
                [P_x, P_y, P_z]])


  eigenvalues = J.eigenvals()
  
  eigenvalues_sub = {eig.subs(config) for eig in eigenvalues}

  results = {eig.subs(substitution) for eig in eigenvalues_sub}

  for result in results:
    print(latex(result.as_real_imag()))


def local_reproductive_func(payoffs, i, j, w):
  diff = payoffs[j] - payoffs[i]
  return (1/2) + (w / 2) * (diff / delta_pi)

def moran_reproductive_func(payoffs, i, j, w):
  payoff = payoffs[j]
  numerator = 1 - w + w * payoff
  denominator = 1 - w + w * averagePayoff
  return numerator / denominator
  

# Returns the transition probabilities given a repoductive function and payoffs.
# Returns dict indexed by T_RS for example between Rocker and Scissors
# The 4th strategy is labelled as L
def transition_probs_moran(reproductive_func, payoffs : list):
  # x y z, q = 1 - x - y - z

  names = ["R", "P", "S", "L"]
  freqs = [x, y, z, q]

  return {
      f"T_{a}{b}": reproductive_func(payoffs, i, j, w_sym) * freqs[i] * freqs[j]
      for i, a in enumerate(names)
      for j, b in enumerate(names)
      if i != j
  }


# Test method to see if stationary dist is the culprit - hardcoded for a c=-0.8, peak in Rock
def non_uniform_weight(x,y,z, alpha=20):
    center = np.array([0.5, 0.2, 0.1])
    point = np.array([x, y, z])
    dist_sq = np.sum((point - center)**2)
    return np.exp(-alpha * dist_sq)


def numerical_H_value(transitions, N = 100):
  
  """expression = (6 / (N * N * N * N * N)) * (q * (1-q) * (transitions["T_RL"] + transitions["T_PL"]
                    +transitions["T_SL"] + transitions["T_LR"]
                    +transitions["T_LP"] + transitions["T_LS"])
              - (q + (1 / N)) * (1 - q - (1 / N)) * (transitions["T_RL"] + transitions["T_PL"] + transitions["T_SL"])  
              - (q - (1 / N)) * (1 - q + (1 / N)) * (transitions["T_LR"] + transitions["T_LP"] + transitions["T_LS"]))
  
 
  """

  
  # New corrected normalized expression.
  expression = ((12/N) * (1 - x - y - z) * (transitions["T_RL"] - transitions["T_LR"] + transitions["T_PL"] - transitions["T_LP"] 
                              + transitions["T_SL"] - transitions["T_LS"]) 
                + (12 / (N*N)) * (transitions["T_LR"] + transitions["T_LP"] + transitions["T_LS"])                
                )
  

  

  """expression_rps = (12 / (N*N*N*N*N)) * ((x * y * z) * (transitions["T_RP"]  
    + transitions["T_RS"] + transitions["T_RL"]
    + transitions["T_PR"] + transitions["T_PS"] 
    + transitions["T_PL"] + transitions["T_SR"] 
    + transitions["T_SP"] + transitions["T_SL"]
    + transitions["T_LR"] + transitions["T_LP"] + transitions["T_LS"])
    - z*(x - (1/N))*(y + (1/N)) * transitions["T_RP"]
    - (x - (1/N)) * y * (z + (1/N)) * transitions["T_RS"]
    - y * z * (x - (1/N)) * transitions["T_RL"]
    - z * (x + (1/N)) * (y - (1/N)) * transitions["T_PR"] 
    - x * (y - (1/N)) * (z + (1/N)) * transitions["T_PS"]
    - x * z * (y - (1/N)) * transitions["T_PL"] 
    - (x + (1/N)) * y * (z - (1/N)) * transitions["T_SR"]
    - x * (y + (1/N)) * (z - (1/N)) * transitions["T_SP"]
    - x * y * (z - (1/N)) * transitions["T_SL"]
    - y * z * (x + (1/N)) * transitions["T_LR"]
    - x * z * (y + (1/N)) * transitions["T_LP"]
    - x * y * (z + (1/N)) * transitions["T_LS"])"""
  
  expression_rps = ((2/N) * (z * (y-x) * (transitions["T_RP"] - transitions["T_PR"])
          + y * (z - x)*(transitions["T_RS"] - transitions["T_SR"])
          + x * (y - z)*(transitions["T_SP"] - transitions["T_PS"]) +
          x * z * (transitions["T_PL"] - transitions["T_LP"]) + 
          y * z * (transitions["T_RL"] - transitions["T_LR"]) +
          x * y * (transitions["T_SL"] - transitions["T_LS"]))
          + (2 / (N*N)) * (z * (transitions["T_RP"] + transitions["T_PR"]) + 
                            y * (transitions["T_RS"] + transitions["T_SR"]) + 
                            x * (transitions["T_SP"] + transitions["T_PS"]))
                  )
  

  config = {a: 0, b: 1, c: -0.8, gamma: 0.2, beta: 0.1}

  expression = expression.subs(w_sym, 0.45)
  expression = expression.subs(config)

  expression_rps = expression_rps.subs(w_sym, 0.45)
  expression_rps = expression_rps.subs(config)


  f = lambdify((x,y,z), expression, "numpy")

  f_rps = lambdify((x,y,z), expression_rps, "numpy")


  def integrand(z,y,x):
    
    return f(x, y, z)
  
  TOL = 1e-14
    

  def integrand_rps(z,y,x):
    return f_rps(x, y, z)
  
  """res, err = nquad(integrand, [
    lambda y, x: [0,1-x-y],
    lambda x: [0, 1 - x], 
    [0,1]])"""
  

  res, err = tplquad(
    integrand,
    0, 1,
    lambda x: 0,
    lambda x: 1 - x,
    lambda x, y: 0,
    lambda x, y: 1 - x - y,
  )
  

  res_rps, err = nquad(integrand_rps, [
    lambda y, x: [0,1-x-y],
    lambda x: [0, 1 - x], 
    [0,1]])
  

  
  print("N", N, "SD:", res, "RPS:", res_rps)

  return res, res_rps


# Root finding with Brent's method to find critical value in a given interval.
def find_critical_N_fixed_w(config, w, transitions):

  transitions = {key: val.subs(config).subs(w_sym, w) for key, val in transitions.items()} # Numeric transitions

  N = sp.symbols('N')

  """expression = (1 / (N ** 2)) * (q * (1-q) * (transitions["T_RL"] + transitions["T_PL"]
                    +transitions["T_SL"] + transitions["T_LR"]
                    +transitions["T_LP"] + transitions["T_LS"])
              - (q + (1 / N)) * (1 - q - (1 / N)) * (transitions["T_RL"] + transitions["T_PL"] + transitions["T_SL"])  
              - (q - (1 / N)) * (1 - q + (1 / N)) * (transitions["T_LR"] + transitions["T_LP"] + transitions["T_LS"]))
  """

  expression = ((12/N) * q * (transitions["T_RL"] - transitions["T_LR"] + transitions["T_PL"] - transitions["T_LP"] + transitions["T_SL"] - transitions["T_LS"]) 
                + (12 / (N*N)) * (transitions["T_LR"] + transitions["T_LP"] + transitions["T_LS"])                
                )

  def brentq_func(N_val):
    expr_sub = expression.subs(N, N_val)
    f = lambdify((x,y,z), expr_sub, "numpy")

    def integrand(z,y,x):
      return f(x,y,z)
    
    res, err = nquad(integrand, [
      lambda y, x: [0,1-x-y],
      lambda x: [0, 1 - x], 
      [0,1]])
   
    return res

  critical_N = brentq(brentq_func, 50, 1500)
  
  return critical_N




def numerical_delta_H_range(n_range = np.linspace(50, 600, 100), config : dict = {a: 0, b: 1, c: -1, gamma: 0.2, beta: 0.1}, plot=True):


  delta_H_SD = []
  delta_H_RPS = []

  for n in n_range:
    res_sd, res_rps = numerical_H_value(transitions, n)
    delta_H_SD.append(res_sd)
    delta_H_RPS.append(res_rps)

  if plot:
    plt.xlabel("$N$")
    plt.ylabel(r"$\langle \Delta H \rangle N^2$", rotation=0)
    plt.plot(n_range, delta_H_SD, label="SD")
    plt.plot(n_range, delta_H_RPS, label="RPS")
    plt.axline((n_range[0], 0), (n_range[0] + 1,0), linewidth=0.3, color="black")
    plt.legend()
    plt.show()


"""
Simulation matches when using a weak drichlet - boundary distribution, i guess makes sense for integral 0 - 1, 

Add option for this, or for original in simulation code.

Need to add derivation and then code for delta H_4, then make some plots of all 3,

Try and see the point at which double drift occurs, any pattern.

Critial population size for double drift in the combined game.
"""


# Compute fixed point
def find_fixed_point_a_x(matrix, w=0.45,N_val=1):
  # Matrix : np.matrix



  a_val = matrix[0, 0]
  b_val = matrix[1, 0]
  c_val = matrix[0, 1]
  gamma_val = matrix[0, 3]
  beta_val = matrix[3, 0]


  # Construct config from matrix.
  config = {a : a_val, b : b_val, c : c_val, gamma : gamma_val, beta : beta_val}

  delta_pi = matrix.max() - matrix.min()
  matrix = sp.Matrix(matrix)
  
  transitions = transition_probs_moran(moran_reproductive_func,  [payoffR, payoffP, payoffS, payoffL])
  transitions = {key: val.subs(delta_pi, 2) for key, val in transitions.items()} # Numeric transitions

  a_x = (transitions["T_PR"] + transitions["T_SR"] + transitions["T_LR"]
         - transitions["T_RP"] - transitions["T_RS"] - transitions["T_RL"])

  a_y = (transitions["T_RP"] + transitions["T_SP"] + transitions["T_LP"]
         - transitions["T_PR"] - transitions["T_PS"] - transitions["T_PL"])
  

  a_z = (transitions["T_RS"] + transitions["T_PS"] + transitions["T_LS"]
         - transitions["T_SR"] - transitions["T_SP"] - transitions["T_SL"])
    
  a_x_sub = a_x.subs(config).subs(w_sym, w)
  a_y_sub = a_y.subs(config).subs(w_sym, w) 
  a_z_sub = a_z.subs(config).subs(w_sym, w)

  numerical = lambdify((x,y,z), [a_x_sub, a_y_sub, a_z_sub], "numpy") 

  def equations(vars):
    return numerical(vars[0], vars[1], vars[2])

  initial_guess = np.array([0.2,0.2,0.2])
  sol = fsolve(equations, initial_guess, full_output=True)

  print("Solution found", sol[0])

  return np.concatenate([sol[0], [1 - np.sum(sol[0])]])





if __name__ == "__main__":

  basic_rps = np.array(
  [[0, -0.8, 1,     0.2], 
    [1, 0, -0.8,     0.2], 
    [-0.8, 1, 0,     0.2], 
    [0.1, 0.1, 0.1, 0]]
  )
  
  find_fixed_point_a_x(basic_rps)

  # Derive replicator equations
  x_dot, y_dot, z_dot = replicators(matrix=A, interactionProcess=None, w=None)

  """print(latex(x_dot))
  print("************************")
  print(latex(y_dot))
  print("*************************")
  print(latex(z_dot))
  """
  """standardConfig = {a: 0, b: 1, c: -1, gamma: sp.Rational(1,5), beta: sp.Rational(1,10)}

  substitutions = substituteHyperParams([x_dot, y_dot, z_dot], standardConfig, (x,y,z))

  fixedPoints = getFixedPoints(substitutions, (x, y, z))
  """
  #print(latex(fixedPoints))

  #eigenvalues = findEigenvalues([x_dot, y_dot, z_dot], standardConfig, (x,y,z), {x: 2/9, y: 2/9, z: 2/9})
  #eigenvalues = findEigenvalues([x_dot, y_dot, z_dot], standardConfig, (x,y,z), {x: 1, y: 0, z: 0})

  print("**************************************************")
  #transitions = transition_probs_moran(moran_reproductive_func,  [payoffR, payoffP, payoffS, payoffL])

  transitions = transition_probs_moran(moran_reproductive_func,  [payoffR, payoffP, payoffS, payoffL])
  transitions = {key: val.subs(delta_pi, 2) for key, val in transitions.items()} # Numeric transitions

  #print(latex(sp.simplify(transitions["T_RP"])))
  #print("***************************")
  #print(latex(sp.simplify(transitions["T_RL"])))

  N = sp.symbols('N')

  a_x = (transitions["T_PR"] + transitions["T_SR"] + transitions["T_LR"]
         - transitions["T_RP"] - transitions["T_RS"] - transitions["T_RL"])
  
  b_x = (transitions["T_PR"] + transitions["T_SR"] + transitions["T_LR"] + transitions["T_RP"] + transitions["T_RS"] + transitions["T_RL"]) / N

  a_y = (transitions["T_RP"] + transitions["T_SP"] + transitions["T_LP"]
         - transitions["T_PR"] - transitions["T_PS"] - transitions["T_PL"])
  
  b_y = (transitions["T_RP"] + transitions["T_SP"] + transitions["T_LP"] + transitions["T_PR"] + transitions["T_PS"] + transitions["T_PL"]) / N
  
  a_z = (transitions["T_RS"] + transitions["T_PS"] + transitions["T_LS"]
         - transitions["T_SR"] - transitions["T_SP"] - transitions["T_SL"])
  
  b_z = (transitions["T_RS"] + transitions["T_PS"] + transitions["T_LS"] + transitions["T_SR"] + transitions["T_SP"] + transitions["T_SL"]) / N

  #a_x = sp.simplify(a_x)
  #x_dot = sp.simplify(x_dot)
  #print("Correct adjusted: ")
  #print(latex(x_dot))
  

  """expression = (1 / (N ** 2)) * (q * (1-q) * (transitions["T_RL"] + transitions["T_PL"]
                    +transitions["T_SL"] + transitions["T_LR"]
                    +transitions["T_LP"] + transitions["T_LS"])
              - (q + (1 / N)) * (1 - q - (1 / N)) * (transitions["T_RL"] + transitions["T_PL"] + transitions["T_SL"])  
              - (q - (1 / N)) * (1 - q + (1 / N)) * (transitions["T_LR"] + transitions["T_LP"] + transitions["T_LS"]))
  
  
  expression =sp.collect(sp.simplify(expression), N)
  print(latex(expression))
  """


  #expression = expression.subs(w_sym, 0)
  #print(latex(sp.simplify(expression)))

  # Can essentially use a_x from fp derivation as the return for numerical trajectory now!

  
  #print("a(x) langevin:")
  #print(latex(a_x))

  #diff = sp.simplify(a_x - x_dot)
  #print("DIFF")
  #print(latex(diff))

  #ratio = sp.simplify(a_x / x_dot)
  #print("ratio a_x / x_dot =")
  #print(latex(ratio))

  #print(a_x.equals(x_dot))

  # a_x = x_dot * scaling factor moran... therefores factored should just give us R
  #factor = sp.cancel(a_x / x_dot) # Calculated the scaling factor for moran

  # Adjusted dynamics
  #gam = (1 - w_sym) / w_sym
  #adjustedScaling = 1 / (gam + averagePayoff)  
  
  #print("Does adjusted scale equal")
  #print(factor.equals(adjustedScaling))

  #print(factor.equals(adjustedScaling))

  #print(latex(adjustedScaling))

  #formatted = sp.factor(sp.factor(a_x,w_sym), adjustedScaling)

  #print(latex(formatted))

  #print("With w = 0")
  #print(latex(formatted.subs(w_sym, 0)))


  numerical_delta_H_range()

  estimated = pd.read_csv("G:/Game theory project/GTfyp/results/critical_N_w_2.csv", comment='#')

  ws = np.linspace(0.1, 0.5, 10)
  critical_Ns = []

  for w in ws:
    critical_N = find_critical_N_fixed_w({a: 0, b: 1, c: -0.8, gamma: 0.2, beta: 0.1}, w, transitions)
    print(f"Critical N at w={w} is ", critical_N)
    critical_Ns.append(critical_N)

  plt.plot(ws, critical_Ns)
  plt.xlabel(r"$w$")
  plt.ylabel(r"$N_c$")
  #plt.yscale("log", base=10)

  plt.plot(estimated["W"], estimated["critical_N"], marker='o', label='Simulated Critical N')

  plt.show()


