from typing import Callable

import numpy as np


Array1X3 = np.ndarray[tuple[3], np.dtype[np.float64]]
Array3X3 = np.ndarray[tuple[3, 3], np.dtype[np.float64]]


def jacobian(f: Callable[[Array1X3], Array1X3], x: Array1X3, eps=0.0001) -> Array3X3:
    J = np.zeros((len(x), len(x)))

    for i in range(len(x)):
        x_forward = x.copy()
        x_backward = x.copy()

        x_forward[i] += eps
        x_backward[i] -= eps
        dfdx = (f(x_forward) - f(x_backward)) / (2 * eps)

        J[:, i] = dfdx

    return np.array(J)


def newton_gaussian(
    f: Callable[[Array1X3], Array1X3],
    x0: Array1X3,
    max_iter=10,
    tol=0.0001,
    eps=0.0001,
):
    x = x0.copy()

    for i in range(max_iter):
        J = jacobian(f, x, eps=eps)
        inv_J: Array3X3 = np.linalg.inv(J).astype(np.float64)
        dx: Array1X3 = inv_J @ -f(x)

        x = x + dx

        err = np.linalg.norm(dx).item()
        if err < tol:
            print(f"총 iteration: {i}")

            break
    else:
        raise Exception(
            f"ERROR:: 최대 iteration({max_iter}) 안에 근을 찾지 못했습니다!"
        )

    return x


def object_function(x: Array1X3) -> Array1X3:
    A = np.array(
        [
            [3, 2, -1],
            [-1, 3, 3],
            [1, -1, -1],
        ]
    )
    B = np.array(
        [
            [10],
            [5],
            [-1],
        ]
    )
    Y = A @ x.reshape(3, 1) - B

    return Y.squeeze(1)


if __name__ == "__main__":
    x0 = np.array([1.0, 1.0, 1.0])

    root = newton_gaussian(object_function, x0, eps=1e-10, tol=1e-10)
    result = object_function(root)

    print()
    print(f"x1 = {root[0]}")
    print(f"x2 = {root[1]}")
    print(f"x3 = {root[2]}")
    print(f"f1(x) = {result[0]}")
    print(f"f2(x) = {result[1]}")
    print(f"f3(x) = {result[2]}")
