from machine_learning_modelling import *
import sympy
from sympy import symbols
from math import exp
import numpy as np



def param(FAR,FRR,k) :
    D0=0.55
    Dhigh=0.8
    Dlow=0.3
    x=symbols('x')
    N=1+exp(-3)+exp(-6)
    F1=1-pow(2,-0.005*(k))
    F2=1-pow(2,-0.005)
    G=F1/F2
    f=x*G+N/2*pow(1-x,2)
    A=np.array([[-f.subs(x,FRR),f.subs(x,1-FRR)],[-f.subs(x,1-FAR),f.subs(x,FAR)]],dtype=np.float64)
    B=np.array([[Dhigh-D0],[Dlow-D0]],dtype=np.float64)
    solutions=np.linalg.solve(A,B)
    return solutions[0][0],solutions[1][0]