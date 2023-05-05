import numpy as np
from scipy.special import roots_chebyt
import matplotlib.pyplot as plt

np.seterr(all='raise')


# the Witch of Agnesi
def f(x):
    return 1 / (1 + x**2)


a, b = -1, 1  # range
x_values = np.linspace(a, b, 1000)
y_values = f(x_values)


# barycentric weights
def w(j, x_n):

    denom = 1
    for k in range(len(x_n)):
        if j != k:
            denom *= (x_n[j] - x_n[k])
    return 1 / denom


# the second barycentric form
def p(x, x_n, w_j):

    numerator = 0
    denom = 0

    for j in range(len(x_n)):
        if x == x_n[j]:
            return f(x_n[j])

        temp = w_j[j] / (x - x_n[j])
        numerator += temp * f(x_n[j])
        denom += temp

    return numerator / denom


# approximation
def approx(x_n):

    w_j = []
    for j in range(len(x_n)):
        w_j.append(w(j, x_n))

    y_appr = []
    diff = []

    for x in x_values:
        pn = p(x, x_n, w_j)
        y_appr.append(pn)
        diff.append(np.abs(f(x) - pn))

    return y_appr, np.max(diff)


# plot the maximum difference
def plot_diff(max_diff, save_path, title):

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xticks(range(21))
    plt.xlim([0, 20])
    plt.plot(range(1, 21), max_diff)
    plt.xlabel("n")
    plt.ylabel(r"$||f-p_n||_{[a,b]}$")
    plt.title(title)
    plt.savefig(save_path)


# plot the approximation
def plot_comp(ys, save_path, title):

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i, (x, y) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        ax = axs[x, y]
        ax.plot(x_values, y_values, label="f")
        ax.plot(x_values, ys[i * 5], color="red", label="$p_n$")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f"n={i*5+1}")
        ax.legend()
    fig.suptitle(title)
    plt.savefig(save_path)


# use different nodes
def diff_nodes(name, func):
    max_diff = []
    ys = []
    for n in range(2, 22):
        y_v, md = approx(func(n))
        ys.append(y_v)
        max_diff.append(md)
    plot_diff(max_diff, name + "_diff.png", name)
    plot_comp(ys, name + "_pn.png", name)


# different nodes
uniform_grid = lambda n: np.linspace(a, b, n)
uniform_distrib = lambda n: np.random.uniform(a, b, n)
chebyshev = lambda n: roots_chebyt(n)[0]
for name, func in [("uniform grid", uniform_grid),
                   ("uniform distribution", uniform_distrib),
                   ("Chebyshev", chebyshev)]:
    diff_nodes(name, func)
