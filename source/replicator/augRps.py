import sympy as sp
from sympy import latex

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

payoffR = a * x + y * c + b * z + gamma * q
payoffP = b * x + a * y + c * z + gamma * q
payoffS = c * x + b * y + a * z + gamma * q 
payoffL = (a + beta) * x + (a + beta) * y + (a + beta) * z

# Pairwise replicator equations 
x_dot = (x * y) * (payoffR - payoffP) + (x * z) * (payoffR - payoffS) + (x * q) * (payoffR - payoffL)
y_dot = (y * x) * (payoffP - payoffR) + (y * z) * (payoffP - payoffS) + (y * q) * (payoffP - payoffL)
z_dot = (z * x) * (payoffS - payoffR) + (z * y) * (payoffS - payoffP) + (z * q) * (payoffS - payoffL)


x_dot_sub = x_dot.subs({a: 0, b: 1, c: -1, gamma: 0.2, beta: 0.1})
y_dot_sub = y_dot.subs({a: 0, b: 1, c: -1, gamma: 0.2, beta: 0.1})
z_dot_sub = z_dot.subs({a: 0, b: 1, c: -1, gamma: 0.2, beta: 0.1})
fixed_points = sp.solve([x_dot_sub, y_dot_sub, z_dot_sub], (x, y, z), dict=True)
print("Fixed points for augmented RPS game: ", fixed_points)



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



standardConfig = {a: 0, b: 1, c: -1, gamma: 0.2, beta: 0.1}


eigenvalues_sub = {eig.subs(standardConfig) for eig in eigenvalues}


results = {eig.subs({x: 2/9, y: 2/9, z: 2/9}) for eig in eigenvalues_sub}

print(latex(results))

#print(latex(eigenvalues_sub))



"""


Find eigenvalues for fixed points  of 
basicRps = np.array([[0,   -1,   1,       0.2],
                    [1,    0,   -1,       0.2],
                    [-1,   1,   0,        0.2],
                    [0.3, 0.3, 0.3, 0]])

Because I think the results are different with drift upwards towards full loner, where there should be a stable point at the center, so the direction of drift is different.



I have also observed drift in this dynamics too!

.4690246902469025e-06  Moran drift
-2.009020090200902e-06  local drift
           c1      c2      c3      c4
99995  0.1658  0.1651  0.1662  0.5029
99996  0.1667  0.1653  0.1666  0.5014
99997  0.1669  0.1650  0.1666  0.5015
99998  0.1666  0.1652  0.1674  0.5008
99999  0.1664  0.1650  0.1677  0.5009
         c1    c2   c3    c4
99995  0.01  0.04  0.0  0.95
99996  0.01  0.04  0.0  0.95
99997  0.01  0.04  0.0  0.95
99998  0.01  0.04  0.0  0.95
99999  0.01  0.04  0.0  0.95
"""