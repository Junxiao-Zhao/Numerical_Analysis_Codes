import numpy as np
import matplotlib.pyplot as plt


def calc_draw(m, m_ivs, t1, t2):

    x = []
    y = []
    z = []
    for i in range(16):
        epsilon = 10**(-i)

        # A_e
        A = m(epsilon)

        # A_e^-1
        A_ivs = m_ivs(epsilon)

        x.append(-i)

        Aepsinv = np.linalg.inv(A)
        y.append(
            np.log10(
                np.linalg.norm(A_ivs - Aepsinv, ord=2) /
                np.linalg.norm(A_ivs, ord=2)))

        # cond(A)
        z.append(np.log10(np.linalg.cond(A)))

    # 3.4
    fig = plt.figure()
    plt.scatter(x, y, s=4)
    plt.grid()
    plt.xlim((-16, 0))
    plt.ylim((-16, 0))
    plt.xticks(np.arange(-16, 0, 1))
    plt.yticks(np.arange(-16, 0, 1))
    plt.xlabel(r"$log_{10}(\epsilon)$")
    plt.ylabel(
        r"$log_{10}(||A_{\epsilon}^{-1} - Aepsinv||_2/||A_{\epsilon}^{-1}||_2)$"
    )
    plt.title(t1)
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('right')
    fig.savefig(t1 + ".png")

    # 3.5
    fig = plt.figure()
    plt.scatter(x, z, s=4)
    plt.grid()
    plt.xlim((-16, 0))
    plt.ylim((0, 16))
    plt.xticks(np.arange(-16, 0, 1))
    plt.yticks(np.arange(0, 16, 1))
    plt.xlabel(r"$log_{10}(\epsilon)$")
    plt.ylabel(r"$log_{10}(\kappa(A_{\epsilon}))$")
    plt.title(t2)
    ax = plt.gca()
    ax.yaxis.set_ticks_position('right')
    fig.savefig(t2 + ".png")

    # plt.show()


if __name__ == "__main__":
    m3 = lambda e: np.matrix([[1, 1], [1, 1 + e]])
    m3_ivs = lambda e: np.matrix([[1 + 1 / e, -1 / e], [-1 / e, 1 / e]])
    calc_draw(m3, m3_ivs, "Q3.4", "Q3.5")

    m4 = lambda e: np.matrix([[4, 2], [2, 1 + e]])
    m4_ivs = lambda e: np.matrix([[1 / (4 * e) + 1 / 4, -1 /
                                   (2 * e)], [-1 / (2 * e), 1 / e]])
    calc_draw(m4, m4_ivs, "Q4.1", "Q4.2")
