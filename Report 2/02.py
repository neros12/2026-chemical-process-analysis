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
        else:
            u2 = u1 - (ys1[-1]) * ((u1 - u0) / (ys1[-1] - ys0[-1]))

            # Update
            u0 = u1
            u1 = u2
    else:
        raise RuntimeError("수렴하지 않았습니다.")

    return xs0, ys0, zs0


def thomas_algorithm(lower, diag, upper, rhs):
    n = len(diag)

    # 복사해서 원본 보존
    a = lower.astype(float).copy()
    b = diag.astype(float).copy()
    c = upper.astype(float).copy()
    d = rhs.astype(float).copy()

    # forward elimination
    for i in range(1, n):
        w = a[i - 1] / b[i - 1]
        b[i] = b[i] - w * c[i - 1]
        d[i] = d[i] - w * d[i - 1]

    # backward substitution
    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]

    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

    return x


def solve_bvp_fdm_tridiag(a, b, ya, yb, m):
    h = (b - a) / m
    x = np.linspace(a, b, m + 1)

    n = m - 1

    lower = np.ones(n - 1)
    diag = np.full(n, -2.0 + h**2 / 4.0)
    upper = np.ones(n - 1)
    rhs = np.full(n, 8.0 * h**2)

    # 경계조건 반영
    rhs[0] -= ya
    rhs[-1] -= yb

    y_inner = thomas_algorithm(lower, diag, upper, rhs)

    y = np.zeros(m + 1)
    y[0] = ya
    y[-1] = yb
    y[1:-1] = y_inner

    return x, y


if __name__ == "__main__":
    y0 = 0.0
    b = 10.0
    yb = 0.0
    u0 = 1
    u1 = 2

    xs, ys, zs = runge_kutta_with_shooting(y0, b, yb, u0, u1)
    x, y = solve_bvp_fdm_tridiag(0.0, b, y0, yb, 100)

    plt.plot(xs, ys, c="black", lw=1, label="y(x)")
    # plt.plot(x, y, c="dimgray", ls="--", lw=1, label="z = y'(x)")
    plt.xlabel("x")
    plt.xlim(xs[0], xs[-1])
    plt.ylim(min(ys), max(ys) + 5)
    plt.tick_params(direction="in")
    plt.legend()
    plt.show()
