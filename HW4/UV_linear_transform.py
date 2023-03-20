import matplotlib.pyplot as plt
import numpy as np

plt.figure()
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)
plt.xlim([-15, 15])
plt.ylim([-15, 15])
ax = plt.gca()
ax.set_aspect('equal')

# unit ball
theta = np.linspace(0, 2 * np.pi, 100)
x = np.cos(theta)
y = np.sin(theta)
unit_ball = np.array([x, y])
plt.plot(x, y)

# image
sqrt2 = np.sqrt(2)
A = np.array([[-2, 11], [-10, 5]])
S = np.array([-10 * sqrt2, -5 * sqrt2])
U = np.array([[1, -1], [1, 1]]) / sqrt2
V = np.array([[3, 4], [-4, 3]]) / 5
image = A @ unit_ball
plt.plot(image[0], image[1])
# σu1 and σu2
for i in range(2):
    plt.arrow(0,
              0,
              U[0][i] * S[i],
              U[1][i] * S[i],
              head_width=0.5,
              length_includes_head=True,
              color='red')
    plt.text(U[0][i] * S[i] / 2,
             U[1][i] * S[i] / 2,
             'σu' + str(i),
             fontsize=12,
             ha='center',
             va='center')
    plt.text(U[0][i] * S[i],
             U[1][i] * S[i],
             "[%.2f, %.2f]" % (U[0][i] * S[i], U[1][i] * S[i]),
             fontsize=12,
             ha='center',
             va='center')
plt.savefig("unit ball & image.png")

# singular vectors
plt.figure()
plt.plot(x, y)
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)
ax = plt.gca()
ax.set_aspect('equal')

for i in range(2):
    plt.arrow(0,
              0,
              V[0][i],
              V[1][i],
              head_width=0.05,
              length_includes_head=True,
              color='green')
    plt.text(V[0][i] / 2,
             V[1][i] / 2,
             "v" + str(i),
             fontsize=12,
             ha='center',
             va='center')
    plt.text(V[0][i],
             V[1][i],
             "[%.2f, %.2f]" % (V[0][i], V[1][i]),
             fontsize=12,
             ha='center',
             va='center')

plt.savefig("singular vectors.png")
