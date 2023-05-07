import numpy as np
import matplotlib.pyplot as plt


def l0(x):
    return 2 * x**2 - 3 * x + 1


def l1(x):
    return -4 * x**2 + 4 * x


def l2(x):
    return 2 * x**2 - x


def p(x):
    return 7 * x**2 / 4 - 3 * x / 4


def f(x):
    return x**4


xs = np.linspace(0, 1, 100)

plt.figure()
plt.title("Lagrange basis functions")
plt.plot(xs, l0(xs), label="$L_0$")
plt.plot(xs, l1(xs), label="$L_1$")
plt.plot(xs, l2(xs), label="$L_2$")
plt.legend()
plt.savefig("Lagrange basis functions.png")

plt.figure()
plt.title("Lagrange Interpolation")
plt.plot(xs, f(xs), label="f")
plt.plot(xs, p(xs), label="p")
plt.legend()
plt.savefig("Lagrange Interpolation.png")

plt.figure()
plt.title("Error")
err = np.abs(f(xs) - p(xs))
print(np.max(err))
plt.plot(xs, err)
plt.savefig("Error.png")

plt.show()
