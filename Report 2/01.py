import matplotlib.pyplot as plt


def f(x, y, z):
    return z


def g(x, y, z):
    return -2 * z - 4 * y


def runge_kutta_4th_order_method(y0: float, z0: float, h: float, n: int):
    xs = [0.0]
    ys = [y0]
    zs = [z0]

    x_old = 0.0
    y_old = y0
    z_old = z0
    for _ in range(n):
        # k1
        k1y = f(x_old, y_old, z_old)
        k1z = g(x_old, y_old, z_old)

        # k2
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

        # k3
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

        # k4
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

        # update
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


# 초기조건 및 구간
if __name__ == "__main__":
    y0 = 2.0
    z0 = 0.0
    step_size = 0.01
    n_iter = 500
    xs, ys, zs = runge_kutta_4th_order_method(y0, z0, step_size, n_iter)

    plt.plot(xs, ys, c="black", lw=1, label="y(x)")
    plt.plot(xs, zs, c="dimgray", ls="--", lw=1, label="z = y'(x)")
    plt.xlabel("x")
    plt.xlim(xs[0], xs[-1])
    plt.tick_params(direction="in")
    plt.legend()
    plt.show()
