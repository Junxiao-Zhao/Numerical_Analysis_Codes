import matplotlib.pyplot as plt
import numpy as np


# generate m*m matrix
def gen_m(num: int):
    M = np.zeros((num, num))
    for i in range(num):
        for j in range(i, num):
            if i == j:
                M[i, j] = 0.1
            else:
                M[i, j] = 1
    return M


# calculate min singular directly
def std_svd(m: np.ndarray):
    u, s, vh = np.linalg.svd(m)
    return np.min(s)


# calculate min singular using sqrt of min lamda of A*A
def sqrt_eig(m: np.ndarray):
    w, v = np.linalg.eig(m.T @ m)
    return np.sqrt(np.min(w))


# draw the loglog plot
def draw(std: list, sqrt: list):
    fig = plt.figure()
    x = np.log10(np.array(range(1, 31)))
    std = np.log10(np.array(std))
    sqrt = np.log10(np.array(sqrt))

    plt.scatter(x, std, s=4, c="red")
    plt.plot(x, std, color="red")
    plt.text(x[-5], std[-1] + 1, "SVD", color="red")
    plt.scatter(x, sqrt, s=4, c="blue")
    plt.plot(x, sqrt, color="blue")
    plt.text(x[-10], sqrt[-1] + 1, "sqrt of eigenvalue", color="blue")
    plt.xlabel(r"$\log_{10}{m}$")
    plt.ylabel(r"$\log_{10}{min\_singular}$")
    plt.savefig("31.4.png")


if __name__ == "__main__":
    std = []
    sqrt = []

    for i in range(1, 31):
        m = gen_m(i)
        std.append(std_svd(m))
        sqrt.append(sqrt_eig(m))

    draw(std, sqrt)
