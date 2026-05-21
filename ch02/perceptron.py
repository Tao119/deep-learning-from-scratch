import numpy as np


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    return 0 if tmp <= 0 else 1


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    return 0 if tmp <= 0 else 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    return 0 if tmp <= 0 else 1


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    return AND(s1, s2)


if __name__ == "__main__":
    for x1, x2 in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        print(f"AND({x1},{x2}) = {AND(x1, x2)}")
    print()
    for x1, x2 in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        print(f"NAND({x1},{x2}) = {NAND(x1, x2)}")
    print()
    for x1, x2 in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        print(f"OR({x1},{x2}) = {OR(x1, x2)}")
    print()
    for x1, x2 in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        print(f"XOR({x1},{x2}) = {XOR(x1, x2)}")
