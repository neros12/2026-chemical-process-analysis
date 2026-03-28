from typing import Callable

import numpy as np


def newton_method(
    f: Callable[[float], float],
    x0: float,
    max_iter=10,
    tol=0.0001,
    eps=0.0001,
) -> float:
    for _ in range(max_iter):
        if abs(f(x0)) < tol:
            return x0

        df = (f(x0 + eps) - f(x0 - eps)) / (2 * eps)
        x0 = x0 - f(x0) / df
    else:
        raise Exception(
            f"ERROR:: 최대 iteration({max_iter}) 안에 근을 찾지 못했습니다!"
        )


def object_function(x: float) -> float:
    return np.exp(-x) - x


if __name__ == "__main__":
    root = newton_method(object_function, 10)
    print(f"x = {root}")
    print(f"f(x) = {object_function(root)}")
