import numpy as np
import matplotlib.pyplot as plt

xs = np.linspace(-1, 1, 10)
phi0, phi1 = [], []
w0, w1 = [], []
x_axis = []

for i in range(len(xs) - 1):

    x_i = xs[i]
    x_i1 = xs[i + 1]

    x = np.linspace(x_i, x_i1, 100)
    x_axis.append(x)
    a = (x - x_i) / (x_i1 - x_i)

    phi0.append(2 * a**3 - 3 * a**2 + 1)
    phi1.append(-2 * a**3 + 3 * a**2)
    w0.append((x_i1 - x_i) * (a**3 - 2 * a**2 + a))
    w1.append((x_i1 - x_i) * (a**3 - a**2))

x_axis = np.ravel(x_axis)
fig = plt.figure()
ax = fig.gca()
ax.set_xticks(xs)
plt.plot(x_axis, np.ravel(phi0))
plt.plot(x_axis, np.ravel(phi1))
plt.plot(x_axis, np.ravel(w0))
plt.plot(x_axis, np.ravel(w1))
plt.savefig("cubic spline cardinal basis.png")
plt.show()
