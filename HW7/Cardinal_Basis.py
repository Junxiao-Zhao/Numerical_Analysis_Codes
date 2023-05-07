import numpy as np
import matplotlib.pyplot as plt


def p(x, c1, c2):
    return c1 * x + c2 * x**2


def phi0(x):
    return 1 - 3 * x + 1.5 * x**2


def phi1(x):
    return -0.5 * x + 0.75 * x**2


def phi2(x):
    return 3 * x - 1.5 * x**2


xs = np.linspace(0, 1, 1000)

plt.figure()
plt.plot(xs, phi0(xs), label="$\Phi_0$")
plt.plot(xs, phi1(xs), label="$\Phi_1$")
plt.plot(xs, phi2(xs), label="$\Phi_2$")
plt.legend()
plt.title("Cardinal Basis")
plt.savefig("Cardinal Basis.png")

plt.figure()
plt.plot(xs, p(xs, -0.5, 0.75), label="$\mu = 0$")
plt.plot(xs, p(xs, 2.5, -0.75), label="$\mu = 1$")
plt.plot(xs, p(xs, 5.5, -9 / 4), label="$\mu = 2$")
plt.legend()
plt.title("$p_{\mu}$")
plt.savefig("p_mu.png")

plt.show()
