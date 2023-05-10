import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


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

X, Y = np.meshgrid(xs, xs)

fig, axes = plt.subplots(3,
                         3,
                         figsize=(10, 10),
                         subplot_kw={'projection': '3d'})
# fig = plt.figure(figsize=(18, 18))
# ax = plt.axes(projection='3d')
fig.suptitle("2D Lagrange basis functions")
ls = [l0, l1, l2]
for m, lm in enumerate(ls):
    for n, ln in enumerate(ls):
        ax = axes[m, n]
        ax.plot_surface(X, Y, lm(X) * ln(Y), cmap='viridis')
        ax.set_title(r"$L_{%s%s}$" % (m, n))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel(r"$L_{%s%s}$" % (m, n))

        # plt.plot(xs, lm(xs) * ln(xs), label=r"$L_{%s%s}$" % (m, n))
# plt.legend()
plt.savefig("2D Lagrange basis functions.png")

plt.show()
