import sympy as sym
from sympy import *
import numpy

def ex1():
    x = sym.symbols("x")
    eq = -x**3 + 3*x**2 + 10*x - 24
    res = solveset(eq, x)
    dataX = numpy.arange(-5,5,.1)
    data = [dataX, [eq.subs(x, i).evalf() for i in dataX]]


def ex2():
    x, y = sym.symbols("x y")
    z = nonlinsolve([x**2 + 3*y -10, 4*x - y**2 + 2], [x,y])
    for s in z:
        for q,w in s.items():
            print(q, w.evalf())
    return z

def ex3(equation):
    for s in equation:
        for x, y in s.items():
            print(x, y.evalf())


def main():
    print(ex2())

if __name__ == "__main__":
    #printing.init_printing(use_latex=True)
    sym.init_printing(use_latex=True)
    main()