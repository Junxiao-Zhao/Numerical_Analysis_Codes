import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


def u_ellipse(u: np.ndarray, s: np.ndarray, title: str):

    # ellipse and x, y axes
    u1 = u[:, 0]
    theta = np.arctan2(u1[1], u1[0]) * 180 / np.pi
    ell = Ellipse((0, 0),
                  width=s[0] * 2,
                  height=s[1] * 2,
                  angle=theta,
                  fill=False,
                  lw=1,
                  color='blue')

    plt.figure()
    plt.axhline(0, color='black', lw=1)
    plt.axvline(0, color='black', lw=1)
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])

    # σu1 and σu2
    for i in range(2):
        plt.arrow(0,
                  0,
                  u[0][i] * s[i],
                  u[1][i] * s[i],
                  head_width=0.1,
                  length_includes_head=True)
        plt.text(u[0][i] * s[i] / 2,
                 u[1][i] * s[i] / 2,
                 'σu' + str(i),
                 fontsize=12,
                 ha='center',
                 va='center')

    title += " ellipse"
    plt.title(title)
    ax = plt.gca()
    ax.add_artist(ell)
    ax.set_aspect('equal')
    plt.savefig(title + ".png")


def v_unit_circle(v: np.ndarray, title: str):

    # unit circle and x, y axes
    theta = np.linspace(0, 2 * np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)

    plt.figure()
    plt.axhline(0, color='black', lw=1)
    plt.axvline(0, color='black', lw=1)

    # v1 and v2
    for i in range(2):
        plt.arrow(0,
                  0,
                  v[i][0],
                  v[i][1],
                  head_width=0.02,
                  length_includes_head=True)
        plt.text(v[i][0] / 2,
                 v[i][1] / 2,
                 'v' + str(i),
                 fontsize=12,
                 ha='center',
                 va='center')

    title += " unit circle"
    plt.plot(x, y)
    plt.title(title)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.savefig(title + ".png")


if __name__ == "__main__":
    A = np.array([[1, 2], [0, 2]])
    u, s, vh = np.linalg.svd(A, False)

    sqrt2 = np.sqrt(2)
    U_list = [
        u,
        np.array([[1, 0], [0, -1]]),
        np.array([[0, 1], [1, 0]]),
        np.array([[1, 0], [0, 1], [0, 0]]),
        np.eye(2),
        np.array([[1, 1], [1, -1]]) / sqrt2
    ]
    S_list = [s, [3, 2], [3, 2], [2, 0], [sqrt2, 0], [2, 0]]
    V_list = [
        vh.T,
        np.eye(2),
        np.array([[0, 1], [1, 0]]),
        np.array([[0, 1], [1, 0]]),
        np.array([[1, 1], [1, -1]]) / sqrt2,
        np.array([[1, 1], [1, -1]]) / sqrt2
    ]
    title_list = ["3.7"] + ["4.1" + i for i in ['a', 'b', 'c', 'd', 'e']]

    for i in range(len(S_list)):
        v_unit_circle(V_list[i], title_list[i])
        u_ellipse(U_list[i], S_list[i], title_list[i])
