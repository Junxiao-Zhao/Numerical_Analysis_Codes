import math
import random

e = 1e-6


def converge(cur_y, y):
    if (abs(y - cur_y) <= e):
        return True
    else:
        return False


def f(x):
    return x**2


# FLO = 7
def Secant(x0, x1, y):
    fx1 = f(x1)
    return x1 - (fx1 - y) * (x1 - x0) / (fx1 - f(x0))


# FLO = 3
def Babylonian(x, y):
    return 0.5 * (x + y / x)


# Secant
def calc_secant(y):

    # FLO = 8
    count = 0
    xList = [y / 2, 0]
    while not converge(f(xList[-1]), y):
        xList.append(Secant(xList[-2], xList[-1], y))
        count += 1

    err = abs(
        (xList[-1] - math.sqrt(y)) /
        (xList[0] - math.sqrt(y))) if (xList[0] - math.sqrt(y)) != 0 else 0

    print(
        "Secant:\n  x = %.10f\n  Iterations: %d\n  FLO count: %d\n  error: %.20f"
        % (xList[-1], count, count * 8, err))


# Babylonian
def calc_babylonian(y):

    # FLO = 4
    count = 0
    xList = [y / 2]
    while not converge(f(xList[-1]), y):
        xList.append(Babylonian(xList[-1], y))
        count += 1

    err = abs(
        (xList[-1] - math.sqrt(y)) /
        (xList[0] - math.sqrt(y))) if (xList[0] - math.sqrt(y)) != 0 else 0
    print(
        "Babylonian:\n  x = %.10f\n  Iterations: %d\n  FLO count: %d\n  error: %.20f"
        % (xList[-1], count, count * 4, err))


# Bisection
def calc_bisection(y):

    # FLO = 4
    l = 0
    r = y if y >= 1 else 1
    mid = -1
    count = 0
    while l < r:
        mid = (l + r) / 2

        cur_y = f(mid)
        count += 1
        if converge(cur_y, y):
            break

        if cur_y > y:
            r = mid
        else:
            l = mid

    print(
        "Bisection:\n  x = %.10f\n  Iterations: %d\n  FLO count: %d\n  error: %.20f"
        % (mid, count, count * 4,
           abs((mid - math.sqrt(y)) /
               ((y if y >= 1 else 1) / 2 - math.sqrt(y)))))


if __name__ == "__main__":
    for i in range(5):
        y = random.random() * 10**i
        #y = i**2
        # for y in (0, 64, 2**8, 2**10, 2**12):
        print("y =", y)
        calc_secant(y)
        calc_babylonian(y)
        calc_bisection(y)
        print()
