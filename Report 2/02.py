import numpy as np
import matplotlib.pyplot as plt


def f(t: float, y: float, z: float):
    """
    z = y'
    """
    return z


def g(t: float, y: float, z: float):
    """
    z' = 8 - y/4
    """
    return 8.0 - y / 4.0


def runge_kutta_4th_order_method(y0: float, z0: float, a: float, b: float, n: int):
    h = (b - a) / n

    xs = [0.0]
    ys = [y0]
    zs = [z0]

    x_old = 0.0
    y_old = y0
    z_old = z0
    for _ in range(n):
        k1y = f(x_old, y_old, z_old)
        k1z = g(x_old, y_old, z_old)

        k2y = f(
            x_old + (1 / 2) * h,
            y_old + (1 / 2) * k1y * h,
            z_old + (1 / 2) * k1z * h,
        )
        k2z = g(
            x_old + (1 / 2) * h,
            y_old + (1 / 2) * k1y * h,
            z_old + (1 / 2) * k1z * h,
        )

        k3y = f(
            x_old + (1 / 2) * h,
            y_old + (1 / 2) * k2y * h,
            z_old + (1 / 2) * k2z * h,
        )
        k3z = g(
            x_old + (1 / 2) * h,
            y_old + (1 / 2) * k2y * h,
            z_old + (1 / 2) * k2z * h,
        )

        k4y = f(
            x_old + h,
            y_old + k3y * h,
            z_old + k3z * h,
        )
        k4z = g(
            x_old + h,
            y_old + k3y * h,
            z_old + k3z * h,
        )

        # Update
        x_new = x_old + h
        y_new = y_old + (1 / 6) * (k1y + 2 * k2y + 2 * k3y + k4y) * h
        z_new = z_old + (1 / 6) * (k1z + 2 * k2z + 2 * k3z + k4z) * h
        xs.append(x_new)
        ys.append(y_new)
        zs.append(z_new)
        x_old = x_new
        y_old = y_new
        z_old = z_new

    return xs, ys, zs


def runge_kutta_with_shooting(
    y0: float,
    b: float,
    yb: float,
    u0: float,
    u1: float,
    tol=1e-8,
    max_iter=100,
):
    for _ in range(max_iter):
        xs0, ys0, zs0 = runge_kutta_4th_order_method(y0, u0, 0.0, b, 100)
        xs1, ys1, zs1 = runge_kutta_4th_order_method(y0, u1, 0.0, b, 100)

        # Check
        if abs(ys0[-1] - yb) <= tol:
            break

        u2 = u1 - (ys1[-1]) * ((u1 - u0) / (ys1[-1] - ys0[-1]))

        # Update
        u0 = u1
        u1 = u2
    else:
        raise RuntimeError(
            f"ERROR:: 최대 iteration({max_iter}) 안에 수렴하지 못했습니다!"
        )

    return xs0, ys0, zs0


def tridiagonal_matrix_method(a: float, b: float, ya: float, yb: float, m: int):
    """
    d2y/dx2 + Ay = B
    A = 1/4
    B = 8
    """
    A = 1 / 4
    B = 8
    h = (b - a) / m
    n = m - 1

    alpha = -2 + A * h * h
    beta = B * h * h
    J = (
        np.diag(np.full(n, alpha))
        + np.diag(np.ones(n - 1), k=1)
        + np.diag(np.ones(n - 1), k=-1)
    )
    R = np.full((n, 1), beta)
    R[0] -= ya
    R[-1] -= yb

    ys = np.linalg.inv(J) @ R

    ys = ys.flatten().tolist()
    ys = [ya, *ys, yb]
    xs = np.linspace(a, b, m + 1)

    return xs, ys


if __name__ == "__main__":
    a = 0.0
    b = 10.0
    ya = 0.0
    yb = 0.0
    u0 = 1
    u1 = 2

    rk_xs, rk_ys, rk_zs = runge_kutta_with_shooting(ya, b, yb, u0, u1)
    tm_xs, tm_ys = tridiagonal_matrix_method(a, b, ya, yb, 100)

    plt.plot(rk_xs, rk_ys, c="red", lw=1.5, zorder=2, label="Runge-Kutta with Shooting")
    plt.plot(tm_xs, tm_ys, c="blue", lw=4, zorder=1, label="Tridiagonal Matrix Method")
    plt.xlabel("x")
    plt.xlim(rk_xs[0], rk_xs[-1])
    plt.ylim(min(rk_ys), max(rk_ys) + 5)
    plt.tick_params(direction="in")
    plt.legend()
    plt.show()
