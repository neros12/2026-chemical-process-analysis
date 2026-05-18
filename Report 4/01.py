import numpy as np
from scipy.linalg import eig
from scipy.optimize import brentq

# ============================================================
# Linear stability analysis of isothermal Newtonian spinning
#
# Dimensionless equations:
#   a_t + (a v)_z = 0
#   d/dz ( a v_z ) = 0
#
# Steady solution:
#   v_s(z) = r^z
#   a_s(z) = 1 / v_s(z)
#
# Perturbation:
#   a(z,t) = a_s(z) + alpha(z) exp(lambda t)
#   v(z,t) = v_s(z) + beta(z) exp(lambda t)
#
# Boundary conditions:
#   alpha(0) = 0
#   beta(0)  = 0
#   beta(1)  = 0
# ============================================================


def spinning_eigenvalues(r, N=300):
    """
    Compute eigenvalues for a given drawdown ratio r and mesh number N.

    Parameters
    ----------
    r : float
        Drawdown ratio.
    N : int
        Number of mesh intervals.

    Returns
    -------
    eigvals : ndarray
        Eigenvalues lambda.
    """

    dz = 1.0 / N
    z = np.linspace(0.0, 1.0, N + 1)

    log_r = np.log(r)

    vs = r**z
    vs_z = log_r * vs
    vs_zz = log_r**2 * vs

    # Unknowns:
    # alpha_1, alpha_2, ..., alpha_N
    # beta_1, beta_2, ..., beta_{N-1}
    #
    # alpha_0 = 0
    # beta_0 = 0
    # beta_N = 0

    n_alpha = N
    n_beta = N - 1
    n_total = n_alpha + n_beta

    A = np.zeros((n_total, n_total), dtype=float)
    B = np.zeros((n_total, n_total), dtype=float)

    def ia(i):
        """Index of alpha_i, i = 1,...,N"""
        return i - 1

    def ib(i):
        """Index of beta_i, i = 1,...,N-1"""
        return n_alpha + i - 1

    # ------------------------------------------------------------
    # Linearized EOC:
    #
    # lambda alpha =
    #   - v_s alpha_z
    #   - v_s_z alpha
    #   + v_s_z / v_s^2 beta
    #   - 1 / v_s beta_z
    # ------------------------------------------------------------

    for i in range(1, N + 1):
        row = i - 1

        # - v_s alpha_z
        if i == N:
            # backward difference at z = 1
            A[row, ia(N)] += -vs[i] / dz
            A[row, ia(N - 1)] += vs[i] / dz
        else:
            # central difference
            if i + 1 <= N:
                A[row, ia(i + 1)] += -vs[i] / (2.0 * dz)
            if i - 1 >= 1:
                A[row, ia(i - 1)] += vs[i] / (2.0 * dz)

        # - v_s_z alpha
        A[row, ia(i)] += -vs_z[i]

        # - 1 / v_s beta_z
        if i == N:
            # beta_N = 0
            A[row, ib(N - 1)] += 1.0 / (vs[i] * dz)
        else:
            if i + 1 <= N - 1:
                A[row, ib(i + 1)] += -1.0 / (vs[i] * 2.0 * dz)
            if i - 1 >= 1:
                A[row, ib(i - 1)] += 1.0 / (vs[i] * 2.0 * dz)

        # + v_s_z / v_s^2 beta
        if i <= N - 1:
            A[row, ib(i)] += vs_z[i] / vs[i] ** 2

        # RHS: lambda alpha_i
        B[row, ia(i)] = 1.0

    # ------------------------------------------------------------
    # Linearized EOM:
    #
    # 0 =
    #   v_s v_s_zz alpha
    #   + v_s v_s_z alpha_z
    #   - v_s_z / v_s beta_z
    #   + beta_zz
    # ------------------------------------------------------------

    for i in range(1, N):
        row = n_alpha + i - 1

        # v_s v_s_zz alpha
        A[row, ia(i)] += vs[i] * vs_zz[i]

        # v_s v_s_z alpha_z
        if i + 1 <= N:
            A[row, ia(i + 1)] += vs[i] * vs_z[i] / (2.0 * dz)
        if i - 1 >= 1:
            A[row, ia(i - 1)] += -vs[i] * vs_z[i] / (2.0 * dz)

        # beta_zz
        A[row, ib(i)] += -2.0 / dz**2
        if i + 1 <= N - 1:
            A[row, ib(i + 1)] += 1.0 / dz**2
        if i - 1 >= 1:
            A[row, ib(i - 1)] += 1.0 / dz**2

        # - v_s_z / v_s beta_z
        coeff = -vs_z[i] / vs[i]

        if i + 1 <= N - 1:
            A[row, ib(i + 1)] += coeff / (2.0 * dz)
        if i - 1 >= 1:
            A[row, ib(i - 1)] += -coeff / (2.0 * dz)

    # Generalized eigenvalue problem:
    # A q = lambda B q
    eigvals = eig(A, B, right=False)

    eigvals = eigvals[np.isfinite(eigvals)]

    return eigvals


def leading_eigenvalue(r, N=300):
    """
    Return eigenvalue with largest real part.
    """

    eigvals = spinning_eigenvalues(r, N)

    lam = eigvals[np.argmax(eigvals.real)]

    # Report positive imaginary part
    if lam.imag < 0:
        lam = np.conjugate(lam)

    return lam


def critical_draw_ratio(N=300, r_left=15.0, r_right=25.0):
    """
    Find critical draw ratio where max(real(lambda)) = 0.
    """

    def growth_rate(r):
        return leading_eigenvalue(r, N).real

    return brentq(growth_rate, r_left, r_right, xtol=1e-8, rtol=1e-8)


# ============================================================
# Main calculation
# ============================================================

if __name__ == "__main__":

    # ------------------------------------------------------------
    # Table 1
    # ------------------------------------------------------------

    r_values = [15.0, 20.0, 20.218, 22.0, 25.0]
    N_table1 = 500

    print("\nTable 1. Largest real and imaginary parts of eigenvalues")
    print(f"{'r':>10s} {'real part':>15s} {'imaginary part':>18s}")
    print("-" * 46)

    for r in r_values:
        lam = leading_eigenvalue(r, N=N_table1)
        print(f"{r:10.3f} {lam.real:15.6f} {abs(lam.imag):18.6f}")

    # ------------------------------------------------------------
    # Table 2
    # ------------------------------------------------------------

    mesh_values = [100, 200, 300, 400, 500]

    print("\nTable 2. Critical drawdown ratio with number of mesh")
    print(f"{'Number of mesh':>18s} {'Critical draw ratio':>22s}")
    print("-" * 44)

    for N in mesh_values:
        r_crit = critical_draw_ratio(N=N, r_left=18.0, r_right=23.0)
        print(f"{N:18d} {r_crit:22.8f}")

    # ------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------

    N_final = 500
    r_critical = critical_draw_ratio(N=N_final, r_left=18.0, r_right=23.0)

    print("\nFinal result")
    print("-" * 30)
    print(f"Critical drawdown ratio at N = {N_final}: {r_critical:.8f}")
