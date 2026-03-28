from typing import Callable

import numpy as np


Array1X2 = np.ndarray[tuple[2], np.dtype[np.float64]]
Array2X2 = np.ndarray[tuple[2, 2], np.dtype[np.float64]]


def jacobian(f: Callable[[Array1X2], Array1X2], x: Array1X2, eps=0.0001) -> Array2X2:
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
    f: Callable[[Array1X2], Array1X2],
    x0: Array1X2,
    max_iter=10,
    tol=0.0001,
    eps=0.0001,
):
    x = x0.copy()

    for _ in range(max_iter):
        J = jacobian(f, x, eps=eps)
        inv_J: Array2X2 = np.linalg.inv(J).astype(np.float64)
        dx: Array1X2 = inv_J @ -f(x)

        x = x + dx

        err = np.linalg.norm(dx).item()
        if err < tol:
            break
    else:
        raise Exception(
            f"ERROR:: 최대 iteration({max_iter}) 안에 근을 찾지 못했습니다!"
        )

    return x


def object_function(x: Array1X2) -> Array1X2:
    function1 = x[0] - 4 * x[0] * x[0] - x[0] * x[1]
    function2 = 2 * x[1] - x[1] * x[1] + 3 * x[0] * x[1]

    return np.array([function1, function2])


if __name__ == "__main__":
    x0 = np.array([-1.0, -1.0])

    root = newton_gaussian(object_function, x0)
    result = object_function(root)

    print(f"x1 = {root[0]}")
    print(f"x2 = {root[1]}")
    print(f"f1(x) = {result[0]}")
    print(f"f2(x) = {result[1]}")
