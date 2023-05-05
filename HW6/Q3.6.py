import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


# the Witch of Agnesi
def f(x):
    return 1 / (1 + x**2)


x = np.linspace(-1, 1, 1000)
y = f(x)

n_values = np.arange(2, 22)
max_error = np.zeros(len(n_values))

cmap = plt.get_cmap('tab20')
colors = [cmap(i) for i in np.linspace(0, 1, 22)]

plt.figure(figsize=(10, 10))
plt.plot(x, y, label='Witch of Agnesi', color=colors[-1])

for i, n in enumerate(n_values):

    x_interp = np.linspace(-1, 1, n)
    y_interp = f(x_interp)

    cs = CubicSpline(x_interp, y_interp, bc_type='natural')
    y_spline = cs(x)

    error = np.max(np.abs(y_spline - y))
    max_error[i] = error

    label = f'n={n-1}'
    plt.plot(x, y_spline, label=label, color=colors[i])

plt.legend()
plt.savefig("splines.png")

fig = plt.figure()
ax = fig.gca()
ax.set_xticks(range(21))
plt.xlim([0, 20])
plt.plot(range(1, 21), max_error)
plt.xlabel('n')
plt.ylabel('Maximum Error')
plt.savefig("errors.png")
plt.show()
