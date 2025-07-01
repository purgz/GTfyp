import sympy as sp
from sympy import latex

# This file contains the RPS replicator derivation for the 3x3 standard game

a, c, b = sp.symbols('a c b')

"""
A = [a         c,         b,     
     b         a,         c,        
     c         b,         a,       
    ]
"""

A = sp.Matrix([[a, c, b], [b, a, c], [c, b, a]])


# NxN matrix - N-1 replicator equations
# x - fraction of rock, y = fraction of paper, z = fraction of scissors
x = sp.symbols('x')
y = sp.symbols('y')
z = 1 - x - y

payoffR = a * x + c * y + b * z
payoffP = b * x + a * y + c * z
payoffS = c * x + b * y + a * z

x_dot = (x * y) * (payoffR - payoffP) + (x * z) * (payoffR - payoffS)
y_dot = (y * z) * (payoffP - payoffS) + (y * x) * (payoffP - payoffR)


# Fixed points
x_dot_sub = x_dot.subs({a: 0, b: 1, c: -1})
y_dot_sub = y_dot.subs({a: 0, b: 1, c: -1})
fixed_points = sp.solve([x_dot_sub, y_dot_sub], (x,y), dict=True)

print("RPS REPLICATORS")
print("Fixed points: ", fixed_points)



F_x = sp.diff(x_dot, x)
F_y = sp.diff(x_dot, y)
G_x = sp.diff(y_dot, x)
G_y = sp.diff(y_dot, y)

J = sp.Matrix([[F_x, F_y], [G_x, G_y]])

eigenvalues = J.eigenvals()

eigenvalues_sub = {eig.subs({a: 0, b: 1, c: -1}) for eig in eigenvalues}


# Evaluate at the fixed point (1/3, 1/3, 1/3)

results = {eig.subs({x: 1/3, y: 1/3}) for eig in eigenvalues_sub}

"""
print(latex(payoffR))
print(latex(payoffP))
print(latex(payoffS))


print(latex(x_dot))
print(latex(y_dot))
"""

#print(F_x)
#print(F_y)
print(latex(G_y))


#print(latex(J))

#print(latex(eigenvalues_sub))
#print(latex(results))