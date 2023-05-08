import numpy as np
import matplotlib.pyplot as plt


def l0(x):
    return x * (x - 1) / 2


def l1(x):
    return -(x + 1) * (x - 1)


def l2(x):
    return (x + 1) * x / 2


xs = np.linspace(0, 1, 100)

plt.figure()
plt.title("Lagrange basis functions")
plt.plot(xs, l0(xs), label="$L_0$")
plt.plot(xs, l1(xs), label="$L_1$")
plt.plot(xs, l2(xs), label="$L_2$")
plt.legend()
plt.savefig("Lagrange basis functions.png")

plt.figure()
plt.title("2D Lagrange basis functions")
ls = [l0, l1, l2]
for m, lm in enumerate(ls):
    for n, ln in enumerate(ls):
        plt.plot(xs, lm(xs) * ln(xs), label=r"$L_{%s%s}$" % (m, n))
plt.legend()
plt.savefig("2D Lagrange basis functions.png")

plt.show()
