# ============================================================
# Governing equations:
#   a_t + (a v)_z = 0
#   d/dz(a v_z) = 0
#
# Steady solution:
#   v_s(z) = r^z
#   a_s(z) = r^(-z)
#
# Relative sinusoidal perturbation:
#   a(t,z) = a_s(z) [1 + alpha(z) exp(i omega t)]
#   v(t,z) = v_s(z) [1 + beta(z) exp(i omega t)]
#
# Linearized frequency-domain equations:
#   i omega alpha + v_s alpha_z + v_s beta_z = 0
#   v_s' alpha_z + v_s' beta_z + v_s beta_zz = 0
#
# Split into real and imaginary parts:
#   v_s alpha_R' - omega alpha_I + v_s beta_R' = 0
#   omega alpha_R + v_s alpha_I' + v_s beta_I' = 0
#   v_s' alpha_R' + v_s' beta_R' + v_s beta_R'' = 0
#   v_s' alpha_I' + v_s' beta_I' + v_s beta_I'' = 0
#
# Boundary forcing cases:
#   Spinneret area:
#       alpha(0) = 1, beta(0) = 0, beta(1) = 0
#
#   Initial velocity:
#       alpha(0) = 0, beta(0) = 1, beta(1) = 0
#
#   Take-up velocity:
#       alpha(0) = 0, beta(0) = 0, beta(1) = 1
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def Vs(z: FloatArray, *, r: float = 25.0) -> FloatArray:
    return r**z


def Vs_dz(z: FloatArray, *, r: float = 25.0) -> FloatArray:
    return np.log(r) * r**z


def As(z: FloatArray, *, r: float = 25.0) -> FloatArray:
    return 1.0 / r**z


def finite_difference_matrices(
    n_grid: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """
    Uniform finite difference grid on z in [0, 1].

    D1: first derivative matrix
    D2: second derivative matrix

    Interior:
        central difference

    Boundary:
        one-sided second-order difference
    """
    z = np.linspace(0.0, 1.0, n_grid)
    dz = z[1] - z[0]

    D1 = np.zeros((n_grid, n_grid), dtype=float)
    D2 = np.zeros((n_grid, n_grid), dtype=float)

    # ------------------------------------------------------------
    # First derivative
    # ------------------------------------------------------------
    for i in range(1, n_grid - 1):
        D1[i, i - 1] = -0.5 / dz
        D1[i, i + 1] = 0.5 / dz

    # z = 0, one-sided
    D1[0, 0] = -3.0 / (2.0 * dz)
    D1[0, 1] = 4.0 / (2.0 * dz)
    D1[0, 2] = -1.0 / (2.0 * dz)

    # z = 1, one-sided
    D1[-1, -3] = 1.0 / (2.0 * dz)
    D1[-1, -2] = -4.0 / (2.0 * dz)
    D1[-1, -1] = 3.0 / (2.0 * dz)

    # ------------------------------------------------------------
    # Second derivative
    # ------------------------------------------------------------
    for i in range(1, n_grid - 1):
        D2[i, i - 1] = 1.0 / dz**2
        D2[i, i] = -2.0 / dz**2
        D2[i, i + 1] = 1.0 / dz**2

    # z = 0, one-sided
    D2[0, 0] = 2.0 / dz**2
    D2[0, 1] = -5.0 / dz**2
    D2[0, 2] = 4.0 / dz**2
    D2[0, 3] = -1.0 / dz**2

    # z = 1, one-sided
    D2[-1, -4] = -1.0 / dz**2
    D2[-1, -3] = 4.0 / dz**2
    D2[-1, -2] = -5.0 / dz**2
    D2[-1, -1] = 2.0 / dz**2

    return z, D1, D2


def idx_alpha_R(i: int, n: int) -> int:
    return i


def idx_alpha_I(i: int, n: int) -> int:
    return n + i


def idx_beta_R(i: int, n: int) -> int:
    return 2 * n + i


def idx_beta_I(i: int, n: int) -> int:
    return 3 * n + i


def build_residual_and_jacobian(
    omega: float,
    *,
    draw_ratio: float,
    n_grid: int,
) -> tuple[FloatArray, FloatArray]:
    z, D1, D2 = finite_difference_matrices(n_grid)
    vs = Vs(z, r=draw_ratio)
    vs_z = Vs_dz(z, r=draw_ratio)

    n = n_grid

    J = np.zeros((4 * n_grid, 4 * n_grid))
    b = np.zeros(4 * n_grid)

    # ------------------------------------------------------------
    # Fill equations at all grid points.
    #
    # Eq.1:
    #   v_s alpha_R' - omega alpha_I + v_s beta_R' = 0
    #
    # Eq.2:
    #   omega alpha_R + v_s alpha_I' + v_s beta_I' = 0
    #
    # Eq.3:
    #   v_s' alpha_R' + v_s' beta_R' + v_s beta_R'' = 0
    #
    # Eq.4:
    #   v_s' alpha_I' + v_s' beta_I' + v_s beta_I'' = 0
    # ------------------------------------------------------------
    for i in range(n):
        row_eq1 = i
        row_eq2 = n + i
        row_eq3 = 2 * n + i
        row_eq4 = 3 * n + i

        for j in range(n):
            # Eq.1
            J[row_eq1, idx_alpha_R(j, n)] += vs[i] * D1[i, j]
            J[row_eq1, idx_beta_R(j, n)] += vs[i] * D1[i, j]

            # Eq.2
            J[row_eq2, idx_alpha_I(j, n)] += vs[i] * D1[i, j]
            J[row_eq2, idx_beta_I(j, n)] += vs[i] * D1[i, j]

            # Eq.3
            J[row_eq3, idx_alpha_R(j, n)] += vs_z[i] * D1[i, j]
            J[row_eq3, idx_beta_R(j, n)] += vs_z[i] * D1[i, j]
            J[row_eq3, idx_beta_R(j, n)] += vs[i] * D2[i, j]

            # Eq.4
            J[row_eq4, idx_alpha_I(j, n)] += vs_z[i] * D1[i, j]
            J[row_eq4, idx_beta_I(j, n)] += vs_z[i] * D1[i, j]
            J[row_eq4, idx_beta_I(j, n)] += vs[i] * D2[i, j]

        # Eq.1 omega term
        J[row_eq1, idx_alpha_I(i, n)] += -omega

        # Eq.2 omega term
        J[row_eq2, idx_alpha_R(i, n)] += omega

    # ------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------
    alpha0_R = 1.0
    alpha0_I = 0.0
    beta0_R = 0.0
    beta0_I = 0.0
    betaN_R = 0.0
    betaN_I = 0.0

    def impose_bc(row: int, col: int, value: float) -> None:
        J[row, :] = 0.0
        J[row, col] = 1.0
        b[row] = value

    # alpha_R(0) = prescribed
    impose_bc(0, idx_alpha_R(0, n), alpha0_R)

    # alpha_I(0) = 0
    impose_bc(n, idx_alpha_I(0, n), alpha0_I)

    # beta_R(0) = prescribed
    impose_bc(2 * n, idx_beta_R(0, n), beta0_R)

    # beta_I(0) = 0
    impose_bc(3 * n, idx_beta_I(0, n), beta0_I)

    # beta_R(1) = prescribed
    impose_bc(2 * n + (n - 1), idx_beta_R(n - 1, n), betaN_R)

    # beta_I(1) = 0
    impose_bc(3 * n + (n - 1), idx_beta_I(n - 1, n), betaN_I)

    return J, b


def newton_solve_linear_system(
    J: FloatArray,
    b: FloatArray,
    *,
    tol: float = 1e-8,
    max_iter: int = 300,
) -> FloatArray:
    """
    Newton method for F(x) = J x - b = 0.

    For this problem, F is linear, so Newton converges in one iteration
    up to numerical precision.

    Newton update:

        J delta = -F(x)
        x_new = x + delta
    """
    x = np.zeros_like(b)

    for _ in range(max_iter):
        residual = J @ x - b
        residual_norm = np.linalg.norm(residual, ord=np.inf)

        if residual_norm < tol:
            return x

        delta = np.linalg.solve(J, -residual)
        x += delta

        residual = J @ x - b
        residual_norm = np.linalg.norm(residual, ord=np.inf)

        if residual_norm < tol:
            return x

    raise RuntimeError(f"Newton method did not converge within {max_iter} iterations.")


def response_at_frequency(
    omega: float,
    draw_ratio: float,
    n_grid: int,
) -> float:
    J, b = build_residual_and_jacobian(
        omega,
        draw_ratio=draw_ratio,
        n_grid=n_grid,
    )

    result = newton_solve_linear_system(J, b)

    alpha_R_end: float = result[idx_alpha_R(n_grid - 1, n_grid)]
    alpha_I_end: float = result[idx_alpha_I(n_grid - 1, n_grid)]

    return (alpha_R_end**2 + alpha_I_end**2) ** 0.5


if __name__ == "__main__":
    n_grid = 100
    draw_ratio = 25.0

    frequencies = np.logspace(0.0, 2.0, 100)
    spinneret_area = np.empty_like(frequencies)
    for i, omega in enumerate(frequencies):
        spinneret_area[i] = response_at_frequency(
            float(omega),
            draw_ratio=draw_ratio,
            n_grid=n_grid,
        )

    # ============================================================
    # Plot
    # ============================================================
    fig, ax = plt.subplots()
    ax.loglog(frequencies, spinneret_area, c="black", lw=0.7)
    ax.set_xlim(1.0, 1.0e2)
    ax.set_ylim(1.0e-1, 1.0e2)
    ax.set_xlabel("Frequency")
    ax.tick_params(direction="in", which="both", top=True, right=True)
    fig.tight_layout()
    plt.title(f"r={draw_ratio:g}  Spinneret area response")
    plt.show()
