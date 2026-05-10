import numpy as np
from numpy.typing import NDArray
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


# ============================================================
# Newton method using finite difference
# ============================================================
def residuals(theta, v, St: float, *, num_grid=100):
    n = num_grid - 1
    xs = np.linspace(0.0, 1.0, num_grid)

    h = xs[1] - xs[0]

    theta = np.asarray(theta, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    R_theta = np.zeros(num_grid)
    R_v = np.zeros(num_grid)

    # --------------------------------------------------------
    # Boundary conditions
    # --------------------------------------------------------
    R_theta[0] = theta[0] - theta0
    R_v[0] = v[0] - v0
    R_v[-1] = v[-1] - v1

    # --------------------------------------------------------
    # theta equation
    #
    # theta' = St * v^(1/6) * (theta_a - theta)
    # --------------------------------------------------------
    for i in range(1, n):
        R_theta[i] = (theta[i + 1] - theta[i - 1]) / (2.0 * h) - f(theta[i], v[i], St)

    # 마지막 theta 방정식은 backward difference 사용
    R_theta[n] = (theta[n] - theta[n - 1]) / h - f(theta[n], v[n], St)

    # --------------------------------------------------------
    # v equation
    #
    # d/dx [ exp{k(1/theta - 1)} * (1/v) * dv/dx ] = 0
    # --------------------------------------------------------
    for i in range(1, n):
        theta_plus = 0.5 * (theta[i + 1] + theta[i])
        theta_minus = 0.5 * (theta[i] + theta[i - 1])

        v_plus = 0.5 * (v[i + 1] + v[i])
        v_minus = 0.5 * (v[i] + v[i - 1])

        q_plus = np.exp(k * (1.0 / theta_plus - 1.0)) * ((v[i + 1] - v[i]) / h) / v_plus

        q_minus = (
            np.exp(k * (1.0 / theta_minus - 1.0)) * ((v[i] - v[i - 1]) / h) / v_minus
        )

        R_v[i] = (q_plus - q_minus) / h

    return R_theta, R_v


def residual_error(R_theta, R_v):
    theta_error = np.max(np.abs(R_theta))
    v_error = np.max(np.abs(R_v))

    return max(theta_error, v_error)


def solve_newton_step(J, R_theta, R_v):
    residual_matrix = np.vstack([R_theta, R_v])
    delta_matrix = np.linalg.solve(J, -residual_matrix.reshape(-1)).reshape(
        residual_matrix.shape
    )

    return delta_matrix


def jacobian(theta, v, St: float, *, num_grid=100):
    size = 2 * num_grid
    J = np.zeros((size, size))

    theta = np.asarray(theta, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    R_theta0, R_v0 = residuals(theta, v, St, num_grid=num_grid)

    for j in range(size):
        theta_new = theta.copy()
        v_new = v.copy()

        if j < num_grid:
            eps = 1e-6 * max(1.0, abs(theta[j]))
            theta_new[j] += eps
        else:
            v_index = j - num_grid
            eps = 1e-6 * max(1.0, abs(v[v_index]))
            v_new[v_index] += eps

        R_theta1, R_v1 = residuals(theta_new, v_new, St, num_grid=num_grid)

        J[:num_grid, j] = (R_theta1 - R_theta0) / eps
        J[num_grid:, j] = (R_v1 - R_v0) / eps

    return J


def damped_newton_update(
    theta,
    v,
    delta_theta,
    delta_v,
    St: float,
    error: float,
    *,
    num_grid=100,
    max_damping_iter=20,
):
    alpha = 1.0
    theta = np.asarray(theta, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    for _ in range(max_damping_iter):
        theta_new = theta + alpha * delta_theta
        v_new = v + alpha * delta_v
        R_theta_new, R_v_new = residuals(theta_new, v_new, St, num_grid=num_grid)

        if residual_error(R_theta_new, R_v_new) <= error:
            break

        alpha *= 0.5
    else:
        raise RuntimeError("Damping did not converge.")

    return theta_new, v_new, R_theta_new, R_v_new, alpha


def newton_method(
    thetas,
    vs,
    St: float,
    *,
    num_grid=100,
    tol=1e-8,
    max_iter=100,
):
    xs = np.linspace(0.0, 1.0, num_grid, dtype=np.float64)
    thetas = np.linspace(0.0, 1.0, num_grid, dtype=np.float64)
    vs = np.linspace(0.0, 1.0, num_grid, dtype=np.float64)
    h = xs[1] - xs[0]

    thetas[0] = theta0
    vs[0] = v0
    vs[-1] = v1

    R_thetas, R_vs = residuals(thetas, vs, St, num_grid=num_grid)
    for i in range(max_iter):
        error = residual_error(R_thetas, R_vs)

        print(f"St = {St}, Newton iteration = {i}, residual = {error:.3e}")

        if error < tol:
            break

        J = jacobian(thetas, vs, St, num_grid=num_grid)

        delta_matrix = solve_newton_step(J, R_thetas, R_vs)
        delta_thetas = delta_matrix[0]
        delta_vs = delta_matrix[1]

        thetas, vs, R_thetas, R_vs, alpha = damped_newton_update(
            thetas,
            vs,
            delta_thetas,
            delta_vs,
            St,
            error,
            num_grid=num_grid,
        )

        theta_error = np.max(np.abs(alpha * delta_thetas))
        v_error = np.max(np.abs(alpha * delta_vs))

        if max(theta_error, v_error) < tol:
            break

    else:
        raise RuntimeError(
            f"ERROR:: 최대 iteration({max_iter}) 안에 수렴하지 못했습니다!"
        )

    return xs, thetas, vs


if __name__ == "__main__":
    num_grid = 100
    thetas0 = np.ones(num_grid, dtype=np.float64)
    vs0 = np.ones(num_grid, dtype=np.float64)

    ax1: Axes
    ax2: Axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for St in St_list:
        xs, thetas, vs = newton_method(thetas0, vs0, St, num_grid=num_grid)

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
