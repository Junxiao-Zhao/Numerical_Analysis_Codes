import numpy as np
import scipy as sp
import time
from functools import wraps
import matplotlib.pyplot as plt

# store the time consumption
manual_LU = []
manual_time = []
scipy_LU = []
scipy_time = []


# wrapper to calculate time consumption
def specify_func(store: list):

    def time_consume(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            store.append(time.time() - start)
            return result

        return wrapper

    return time_consume


# q3.2 LU decomposition
@specify_func(manual_LU)
def LU_decomp(A: np.ndarray):
    n = A.shape[0]
    U = np.copy(A)
    L = np.eye(n)
    P = np.eye(n)

    for j in range(n - 1):
        i = np.argmax(abs(U[j:, j])) + j

        U[[j, i], j:] = U[[i, j], j:]
        L[[j, i], :j] = L[[i, j], :j]
        P[[j, i], :] = P[[i, j], :]

        L[j + 1:, j] = U[j + 1:, j] / U[j, j]
        U[j + 1:, j:] -= L[j + 1:, j, np.newaxis] * U[j, j:]

    return P, L, U


# q3.3: forward substitution
def forward_sub(L: np.ndarray, c: np.ndarray):
    n = L.shape[0]
    y = np.zeros(n)

    for i in range(n):
        y[i] = (c[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    return y


# q3.3: backward substitution
def backward_sub(U: np.ndarray, d: np.ndarray):
    n = U.shape[0]
    z = np.zeros(n)

    for i in range(n - 1, -1, -1):
        z[i] = (d[i] - np.dot(U[i, i + 1:], z[i + 1:])) / U[i, i]

    return z


# q3.4: compute manually
@specify_func(manual_time)
def manual(A: np.ndarray, b: np.ndarray):
    P, L, U = LU_decomp(A)
    y = forward_sub(L, np.dot(P, b))
    z = backward_sub(U, y)
    return z


@specify_func(scipy_LU)
def sp_lu(A: np.ndarray):
    return sp.linalg.lu(A)


# q3.4: using scipy
@specify_func(scipy_time)
def with_scipy(A: np.ndarray, b: np.ndarray):
    P, L, U = sp_lu(A)
    y = sp.linalg.solve_triangular(L, P.dot(b), lower=True)
    z = sp.linalg.solve_triangular(U, y)
    return z


def draw(x: list, y: list, title: str):
    x = np.log10(x)
    y = np.log10(y)

    fig = plt.figure()
    plt.scatter(x, y, s=6)
    plt.plot(x, y)
    plt.grid()
    plt.xlabel(r"$log_{10}(n)$")
    plt.ylabel(r"$log_{10}(CPU\_time)$")
    plt.title(title)
    fig.savefig(title + ".png")


if __name__ == "__main__":
    n_list = [10, 30, 100, 300, 1000, 3000, 10000]
    for n in n_list:
        A = np.random.randn(n, n)
        b = np.random.randn(n)
        manual(A, b)
        with_scipy(A, b)

    manual_solve = [manual_time[i] - manual_LU[i] for i in range(len(n_list))]
    scipy_solve = [scipy_time[i] - scipy_LU[i] for i in range(len(n_list))]

    print(manual_LU)
    print(manual_solve)
    print(scipy_LU)
    print(scipy_solve)

    draw(n_list, manual_LU, "Manually LU")
    draw(n_list, scipy_LU, "Scipy LU")
    draw(n_list, manual_solve, "Manually solve Linear System")
    draw(n_list, scipy_solve, "Scipy solve Linear System")
