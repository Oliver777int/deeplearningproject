import numpy as np
import csv
import math

#delta = 1
u = 1.66054e-27
a1 = 0.016919*u
a2 = 0.019114*u
a3 = 0.0007626*u
a4 = 0.02544*u
a5 = 0.036*u
mp = 1.67262e-27 #kg
mn = 1.67492749804e-27 #kg
to_MeV = 6.242e12
def M(Z,A):
    M0 = Z*mp+(A-Z)*mn
    M1 = -a1*A
    M2 = a2*A**(2/3)
    M3 = a3*Z**2/A**(3/2)
    M4 = a4*(A-2*Z)**2/A
    delta = a5*A**(-3/4)
    if A%2 != 0:
        delta = 0
    elif Z%2 ==0:
        delta *= -1

    M = M0+M1+M2+M3+M4+delta
    return M

A = 237
Z = 92
c = 3e8
Q = (M(Z,A)+mn -M(Z,A+1))*c**2*to_MeV
#Q = (mn)*c**2*to_MeV
print(Q)
