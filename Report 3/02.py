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
    return q * v / np.exp(k * (1.0 / theta - 1.0))


# ============================================================
# Fixed boundary version
# ============================================================
def unpack_U(U, num_grid: int):
    """
    Boundary values are fixed:

        theta[0] = theta0
        v[0] = v0
        v[-1] = v1

    Unknown vector:

        U = [theta_1, ..., theta_N, v_1, ..., v_{N-1}]
    """
    n = num_grid - 1

    theta = np.zeros(num_grid)
    v = np.zeros(num_grid)

    theta[0] = theta0
    v[0] = v0
    v[n] = v1

    theta[1:] = U[:n]
    v[1:n] = U[n:]

    return theta, v


def residual_vector(U, St: float, *, num_grid=100):
    """
    Residual vector for fixed-boundary finite difference Newton method.

    Unknowns:
        theta_1 ... theta_N      -> n unknowns
        v_1 ... v_{N-1}          -> n-1 unknowns

    Total unknowns = 2n - 1
    Total residuals = 2n - 1
    """
    n = num_grid - 1
    xs = np.linspace(0.0, 1.0, num_grid)
    h = xs[1] - xs[0]

    theta, v = unpack_U(U, num_grid)

    R_theta = np.zeros(n)
    R_v = np.zeros(n - 1)

    # --------------------------------------------------------
    # theta equation
    #
    # theta' = St * v^(1/6) * (theta_a - theta)
    #
    # Unknown theta points are theta_1 ... theta_N.
    # For theta_1 ... theta_{N-1}, use central difference.
    # For theta_N, use backward difference.
    # --------------------------------------------------------
    for i in range(1, n):
        R_theta[i - 1] = (theta[i + 1] - theta[i - 1]) / (2.0 * h) - f(
            theta[i], v[i], St
        )

    R_theta[n - 1] = (theta[n] - theta[n - 1]) / h - f(theta[n], v[n], St)

    # --------------------------------------------------------
    # v equation
    #
    # d/dx [ exp{k(1/theta - 1)} * (1/v) * dv/dx ] = 0
    #
    # Unknown v points are v_1 ... v_{N-1}.
    # Therefore use residuals at i = 1 ... N-1.
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

        R_v[i - 1] = (q_plus - q_minus) / h

    return np.concatenate([R_theta, R_v])


def jacobian(U, St: float, *, num_grid=100):
    size = len(U)
    J = np.zeros((size, size))

    R0 = residual_vector(U, St, num_grid=num_grid)

    for i in range(size):
        dU = np.zeros(size)
        eps = 1e-6 * max(1.0, abs(U[i]))
        dU[i] = eps

        R1 = residual_vector(U + dU, St, num_grid=num_grid)

        J[:, i] = (R1 - R0) / eps

    return J


def newton_method(
    St: float,
    *,
    num_grid=100,
    tol=1e-8,
    max_iter=30,
):
    xs = np.linspace(0.0, 1.0, num_grid)
    n = num_grid - 1

    # --------------------------------------------------------
    # Initial guess including boundary values
    # --------------------------------------------------------
    theta_guess_full = theta_a + (theta0 - theta_a) * np.exp(-St * xs)
    v_guess_full = v0 * (v1 / v0) ** xs

    # --------------------------------------------------------
    # Remove fixed boundary values from unknown vector
    #
    # Unknown vector:
    #   theta_1 ... theta_N
    #   v_1 ... v_{N-1}
    # --------------------------------------------------------
    U = np.concatenate(
        [
            theta_guess_full[1:],  # theta_1 ... theta_N
            v_guess_full[1:n],  # v_1 ... v_{N-1}
        ]
    )

    for i in range(max_iter):
        R = residual_vector(U, St, num_grid=num_grid)
        error = np.max(np.abs(R))

        print(f"St = {St}, iteration = {i}, residual = {error:.3e}")

        if error < tol:
            break

        J = jacobian(U, St, num_grid=num_grid)

        delta = np.linalg.inv(J) @ (-R)
        U += delta

    else:
        raise RuntimeError("Newton method did not converge.")

    thetas, vs = unpack_U(U, num_grid)

    return xs, thetas, vs


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    num_grid = 100

    results = {}

    ax1: Axes
    ax2: Axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for St in St_list:
        print()
        print("======================================")
        print(f"Solving for St = {St}")
        print("======================================")

        xs, thetas, vs = newton_method(St, num_grid=num_grid)

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
