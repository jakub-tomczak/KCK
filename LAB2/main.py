import sympy as sym
from sympy import *
import numpy

def ex1():
    x = sym.symbols("x")
    eq = -x**3 + 3*x**2 + 10*x - 24
    res = solveset(eq, x)
    print("Pierwiastki równania ",eq," :",res)
    dataX = numpy.arange(-5,5,.1)
    dataY = [eq.subs(x, i).evalf() for i in dataX]

    from matplotlib import pyplot as plt
    plt.rc('font', family="Times New Roman")
    plt.plot(dataX, dataY)
    plt.ylabel("Y")
    plt.xlabel("X")
    plt.show()


def ex2():
    x, y = sym.symbols("x y")
    z = solve([x**2 + 3*y -10, 4*x - y**2 + 2])
    print("Rowiązanie układu równań\n", z)
    print("Liczba rozwiązań (par) ", len(z))
    return z

def ex3(equation):
    print("Rozwiązanie w postaci numerycznej:")
    for s in equation:
        for x, y in s.items():
            print(x, y.evalf())

def ex4():
    x = sym.symbols("x")
    eq = sin(log(x,2))*cos(x**2)/x
    eq = eq.diff()
    print("Pochodna funkcji z zadania 5-tego:\n",eq)


def ex5():
    print("Biblioteka numpy")
    import numpy as np
    M = np.array([[1,3,1,2], [1,2,5,8], [3,1,2,9], [5,4,2,1]])

    print("Przed usunięciem\n", M)
    M = M[1:-1, :-1]
    print("Po usunięciu\n", M)

    M2 = np.array([[2,3,1], [5,1,3]])
    print("Przed transpozycją\n", M2)
    M2 = M2.T
    print("Po transpozycji\n", M2)


    dotProduct = M.dot(M2)
    print("Iloczyn macierzy\n", dotProduct)


    #plot
    from matplotlib import pyplot as plt

    colors = ['b', 'r', 'g']
    x = [np.arange(-np.pi, np.pi, i) for i in [np.pi, 2*np.pi/10, 2*np.pi/100]]
    for j,color in zip(x, colors):
        plt.plot(j,[sym.sin(i) for i in j], color)

    plt.show()

def main():
    ex1()
    result = ex2()
    ex3(result)
    ex4()
    ex5()

if __name__ == "__main__":
    sym.init_printing(use_latex=True)
    main()