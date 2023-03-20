import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import copy


# truncate SVD
def truncate(u: np.ndarray,
             s: np.ndarray,
             vh: np.ndarray,
             k: int,
             only_img: bool = True):
    uk = u[:, :k]
    sk = np.diag(s[:k])
    vhk = vh[:k, :]
    imgk = uk @ sk @ vhk

    if only_img:
        return imgk
    else:
        return uk.nbytes + sk.nbytes + vhk.nbytes, imgk


# draw matrix
def draw(ax, img: np.ndarray, title: str, cmap: str):
    ax.imshow(img, cmap=cmap)
    ax.set_title(title)


# draw truncated image for different k
def draw_trunc(u: np.ndarray, s: np.ndarray, vh: np.ndarray):
    img_list = []
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    for k in [2, 4, 6, 8]:
        k_img = truncate(u, s, vh, k)
        img_list.append(k_img)

    for i, (x, y) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        title = "k=%d" % ((i + 1) * 2)
        draw(axs[x, y], img_list[i], title, 'gray')

    plt.savefig("k images.png")


# draw single pointwise error
def pointwise(u: np.ndarray, s: np.ndarray, vh: np.ndarray,
              norm_img: np.ndarray, k: int):
    trunc_img = truncate(u, s, vh, k)
    err_img = np.abs(norm_img - trunc_img)
    avg_err = np.mean(err_img)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    title = "Pointwise Error\nk=%d, average error=%.3f" % (k, avg_err)
    draw(axs[1], err_img, title, 'viridis')
    fig.colorbar(ScalarMappable())
    draw(axs[0], trunc_img, "truncate image", 'gray')
    plt.savefig('pointwise.png')


# pointwise errors vs k
def draw_err(u: np.ndarray, s: np.ndarray, vh: np.ndarray):
    err_list = []

    for k in range(1, 1025):
        err_img = np.abs(norm_img - truncate(u, s, vh, k))
        err_list.append(np.max(err_img))

    fig = plt.figure()
    ax = fig.gca()
    plt.semilogy(range(1, 1025), err_list)
    plt.xlabel("k")
    plt.ylabel(r"$\log_{10}{max\_pointwise\_err}$")
    ax.set_title("Pointwise error vs k")
    plt.savefig("k_pointwise.png")


# compressed image size vs k
def draw_size(u: np.ndarray, s: np.ndarray, vh: np.ndarray):
    size_list = []

    for k in range(1, 1025):
        size_list.append(truncate(u, s, vh, k, False)[0])

    fig = plt.figure()
    ax = fig.gca()
    plt.semilogy(range(1, 1025), size_list)
    plt.xlabel("k")
    plt.ylabel(r"$\log_{10}{compressed\_image\_size}$")
    ax.set_title("compressed image size vs k")
    plt.savefig("k_img_size.png")


# split a matrix into sub-matrices
def split(array, nrows):
    _, h = array.shape
    return (array.reshape(h // nrows, nrows, -1,
                          nrows).swapaxes(1, 2).reshape(-1, nrows, nrows))


# search for block size and best k
def search_k_block(norm_img: np.ndarray):
    avg_err = 0.007
    best_k = []
    block_list = []
    size_list = []
    block_size = 0
    target_block_size = block_size
    min_size = np.infty
    target_list = []

    # try block size
    for i in range(1, 11):
        block_size = 2**i
        best_k.clear()
        block_list.clear()
        size_list.clear()
        suitable_size = True

        # each subblock
        for block in split(norm_img, block_size):
            ub, sb, vhb = np.linalg.svd(block)

            k = 1
            # try different k
            while k <= block_size:
                c_size, c_img = truncate(ub, sb, vhb, k, False)
                c_err = np.mean(np.abs(block - c_img))

                # find a suitable k, stop for this block
                if c_err <= avg_err:
                    best_k.append(k)
                    block_list.append(c_img)
                    size_list.append(c_size)
                    break

                k *= 2

            # failed if cannot find a suitable k within current block size
            if k > block_size:
                suitable_size = False
                break

        # stop if find suitable block size and it's minimum
        if suitable_size:
            cur_size = sum(size_list)
            if cur_size < min_size:
                min_size = cur_size
                target_list = copy.deepcopy(block_list)
                target_block_size = block_size

    horizon = []
    num_rows = int(1024 / target_block_size)
    for i in range(0, len(target_list), num_rows):
        horizon.append(np.hstack(target_list[i:i + num_rows]))

    compress_img = np.vstack(horizon)
    err_img = np.abs(norm_img - compress_img)
    avg_err = np.mean(err_img)

    title = "Pointwise Error for block matrix\nblock size=%d*%d\naverage error=%.3f\ncompress size=%d" % (
        target_block_size, target_block_size, avg_err, min_size)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    draw(axs[1], err_img, title, 'viridis')
    fig.colorbar(ScalarMappable())
    draw(axs[0], compress_img, "compressed block", 'gray')
    plt.savefig('block_pointwise.png')


if __name__ == "__main__":
    norm_img = np.array(Image.open("Grayscale_moon.jpg").convert("L")) / 255
    # u, s, vh = np.linalg.svd(norm_img)
    """ fig = plt.figure()
    ax = fig.gca()
    draw(ax, norm_img, "norm", 'gray')
    plt.savefig("norm_moon.jpg")
    draw_trunc(u, s, vh) """
    """ k = 1
    while k <= 1024:
        pointwise(u, s, vh, norm_img, k)
        k *= 2 """
    """ pointwise(u, s, vh, norm_img, 128)
    draw_err(u, s, vh)
    draw_size(u, s, vh) """
    search_k_block(norm_img)
