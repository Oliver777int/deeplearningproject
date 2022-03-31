import numpy as np
import csv
import math

delta = -1
av = 15.56
a_s = 17.23
ac = 0.697
aa = 93.14
ap = delta*12

def BE(Z,A):
    # return -av * A + a_s * A ** (2 / 3) + ac * Z ** 2 / A ** (1 / 3) + aa * (Z - A / 2) ** 2 / A + ap * A ** (-1 / 2)
    return  -(-av*A + a_s*A**(2/3) + ac*Z**2/A**(1/3) + aa*(Z-A/2)**2/A + ap*A**(-1/2))

A = 238
Z = 92
c = 3e8
Q = (BE(Z,A)-2*BE(Z/2,A/2))

ans = 10**9*19.3*10**6*6.022e23*1e-3/197

lam = np.log(2)/(5730*365*24*60)
math.e**(-lam)

a4 = 0.02544
a3 = 0.0007626
A = 88
z = 2*a4*A/(A**(2/3)*a3+4*a4)
print(z)