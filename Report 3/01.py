import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# ============================================================
# Constants
# ============================================================
theta_a = 0.543
k = 5.8
v0 = 1.0
v1 = 40.0
theta0 = 1.0
St_list = [0.1, 0.5, 1.0]


def f(theta: float, v: float, St: float):
    """
    theta' = St * v^(1/6) * (theta_a - theta)
    """
    return St * v ** (1.0 / 6.0) * (theta_a - theta)


def g(theta: float, v: float, q: float):
    """
    v' = q * v / exp[k(1/theta - 1)]

    where
    q = exp[k(1/theta - 1)] * (1/v) * v'
    """
    return q * v / (np.exp(k * (1.0 / theta - 1.0)))


def runge_kutta_4th_order_method(St: float, q: float, *, num_grid):
    xs = [0.0]
    thetas = [theta0]
    vs = [v0]
    x_old = xs[0]
    theta_old = thetas[0]
    v_old = vs[0]

    n: list[float] = np.linspace(0, 1, num_grid)[1:].tolist()
    h = n[1] - n[0]
    for _ in n:
        k1y = f(theta_old, v_old, St)
        k1z = g(theta_old, v_old, q)

        k2y = f(theta_old + (1 / 2) * k1y * h, v_old + (1 / 2) * k1z * h, St)
        k2z = g(theta_old + (1 / 2) * k1y * h, v_old + (1 / 2) * k1z * h, q)

        k3y = f(theta_old + (1 / 2) * k2y * h, v_old + (1 / 2) * k2z * h, St)
        k3z = g(theta_old + (1 / 2) * k2y * h, v_old + (1 / 2) * k2z * h, q)

        k4y = f(theta_old + k3y * h, v_old + k3z * h, St)
        k4z = g(theta_old + k3y * h, v_old + k3z * h, q)

        # Update
        x_new = x_old + h
        theta_new = theta_old + (1 / 6) * (k1y + 2 * k2y + 2 * k3y + k4y) * h
        v_new = v_old + (1 / 6) * (k1z + 2 * k2z + 2 * k3z + k4z) * h

        xs.append(float(x_new))
        thetas.append(float(theta_new))
        vs.append(float(v_new))
        x_old = x_new
        theta_old = theta_new
        v_old = v_new

    return xs, thetas, vs


def runge_kutta_with_shooting(
    q0: float,
    q1: float,
    St: float,
    *,
    num_grid,
    tol=1e-8,
    max_iter=100,
):
    for i in range(max_iter):
        xs0, thetas0, vs0 = runge_kutta_4th_order_method(St, q0, num_grid=num_grid)
        xs1, thetas1, vs1 = runge_kutta_4th_order_method(St, q1, num_grid=num_grid)
        res0 = vs0[-1] - v1
        res1 = vs1[-1] - v1

        print(f"St = {St}, iteration = {i}, residual = {res0:.3e}")

        # Check
        if abs(res0) <= tol:
            break

        q2 = q1 - res1 * ((q1 - q0) / (res1 - res0))

        # Update
        q0 = q1
        q1 = q2
    else:
        raise RuntimeError(
            f"ERROR:: 최대 iteration({max_iter}) 안에 수렴하지 못했습니다!"
        )

    return xs0, thetas0, vs0


if __name__ == "__main__":
    q0 = 10
    q1 = 100
    num_grid = 101

    ax1: Axes
    ax2: Axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for St in St_list:
        print()
        print("======================================")
        print(f"Solving for St = {St}")
        print("======================================")

        xs, thetas, vs = runge_kutta_with_shooting(q0, q1, St, num_grid=num_grid)

        ax1.plot(xs, thetas, label=f"St = {St}")
        ax1.set_xlabel("x")
        ax1.set_ylabel("theta")
        ax1.set_xlim(0, 1)
        ax1.grid(True)
        ax1.legend()

        ax2.plot(xs, vs, label=f"St = {St}")
        ax2.set_xlabel("x")
        ax2.set_ylabel("v")
        ax2.set_xlim(0, 1)
        ax2.grid(True)
        ax2.legend()

    plt.tight_layout()
    plt.show()
