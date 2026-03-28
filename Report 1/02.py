import numpy as np


Array1X3 = np.ndarray[tuple[int], np.dtype[np.float64]]
Array3X3 = np.ndarray[tuple[int, int], np.dtype[np.float64]]


def LU_decompose(A: Array3X3) -> tuple[Array3X3, Array3X3]:
    L = np.eye(A.shape[0])
    U = A.copy()

    for j in range(A.shape[0]):
        for i in range(j + 1, A.shape[0]):
            factor = U[i, j] / U[j, j]

            L[i, j] = factor
            U[i, j:] -= factor * U[j, j:]

    return L, U


def forward_subsitution(L: Array3X3, B: Array1X3) -> Array1X3:
    D = np.zeros(len(B))

    for i in range(len(B)):
        d = 0.0
        for j in range(i):
            d += L[i, j] * D[j]

        D[i] = B[i] - d

    return D


def backward_subsitution(U: Array3X3, D: Array1X3) -> Array1X3:
    X = np.zeros(len(D))

    for i in range(len(D) - 1, -1, -1):
        x = 0.0
        for j in range(i + 1, len(D)):
            x += U[i, j] * X[j]

        X[i] = (D[i] - x) / U[i, i]

    return X


if __name__ == "__main__":
    A = np.array(
        [
            [3.0, 2.0, -1.0],
            [-1.0, 3.0, 3.0],
            [1.0, -1.0, -1.0],
        ]
    )
    B = np.array([10.0, 5.0, -1.0])

    L, U = LU_decompose(A)
    D = forward_subsitution(L, B)
    X = backward_subsitution(U, D)

    print(f"X: {X}")
    print()
    print(f"AX: {A@X}")
    print(f"B: {B}")
