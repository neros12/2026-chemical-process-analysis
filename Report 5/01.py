import numpy as np

# ============================================================
# Linear stability analysis of isothermal Newtonian spinning
# Shift-Invert transformation method
#
# Original generalized eigenvalue problem:
#
#     A q = lambda M q
#
# Shift-invert transformation:
#
#     inv(A - sigma M) M q = mu q
#
# where
#
#     mu = 1 / (lambda - sigma)
#     lambda = sigma + 1 / mu
#
# ============================================================


def build_matrices(r: float, N=300):
    """
    Build original A and M matrices for

        A q = lambda M q

    Unknown vector:
        q = [alpha_1, ..., alpha_N, beta_1, ..., beta_{N-1}]

    Boundary conditions:
        alpha_0 = 0
        beta_0  = 0
        beta_N  = 0
    """

    dz = 1.0 / N
    z = np.linspace(0.0, 1.0, N + 1)

    vs = r**z
    vs_z = np.log(r) * vs
    vs_zz = np.log(r) ** 2 * vs

    n_alpha = N
    n_beta = N - 1
    n_total = n_alpha + n_beta

    A = np.zeros((n_total, n_total), dtype=float)

    # Mass matrix M
    # EOC has lambda * alpha
    # EOM has no time derivative, so corresponding rows are zero
    M = np.zeros((n_total, n_total), dtype=float)

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
            A[row, N - 1] += -vs[i] / dz
            A[row, N - 2] += vs[i] / dz
        else:
            # alpha_{i+1}
            if i + 1 <= N:
                A[row, i] += -vs[i] / (2.0 * dz)

            # alpha_{i-1}
            if i - 1 >= 1:
                A[row, i - 2] += vs[i] / (2.0 * dz)

        # - v_s_z alpha_i
        A[row, i - 1] += -vs_z[i]

        # - 1 / v_s beta_z
        if i == N:
            # beta_N = 0
            # beta_{N-1} index = n_alpha + N - 2
            A[row, n_alpha + N - 2] += 1.0 / (vs[i] * dz)
        else:
            # beta_{i+1}
            if i + 1 <= N - 1:
                A[row, n_alpha + i] += -1.0 / (vs[i] * 2.0 * dz)

            # beta_{i-1}
            if i - 1 >= 1:
                A[row, n_alpha + i - 2] += 1.0 / (vs[i] * 2.0 * dz)

        # + v_s_z / v_s^2 beta_i
        if i <= N - 1:
            A[row, n_alpha + i - 1] += vs_z[i] / vs[i] ** 2

        # RHS: lambda alpha_i
        M[row, i - 1] = 1.0

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

        # v_s v_s_zz alpha_i
        A[row, i - 1] += vs[i] * vs_zz[i]

        # v_s v_s_z alpha_z
        # alpha_{i+1}
        if i + 1 <= N:
            A[row, i] += vs[i] * vs_z[i] / (2.0 * dz)

        # alpha_{i-1}
        if i - 1 >= 1:
            A[row, i - 2] += -vs[i] * vs_z[i] / (2.0 * dz)

        # beta_zz
        # beta_i
        A[row, n_alpha + i - 1] += -2.0 / dz**2

        # beta_{i+1}
        if i + 1 <= N - 1:
            A[row, n_alpha + i] += 1.0 / dz**2

        # beta_{i-1}
        if i - 1 >= 1:
            A[row, n_alpha + i - 2] += 1.0 / dz**2

        # - v_s_z / v_s beta_z
        coeff = -vs_z[i] / vs[i]

        # beta_{i+1}
        if i + 1 <= N - 1:
            A[row, n_alpha + i] += coeff / (2.0 * dz)

        # beta_{i-1}
        if i - 1 >= 1:
            A[row, n_alpha + i - 2] += -coeff / (2.0 * dz)

        # M row remains zero because EOM has no time derivative

    return A, M


def solve_problem(r: float, N=300, sigma=0.0, tol=1e-10) -> tuple[float, float]:
    """
    Compute leading eigenvalue using shift-invert transformation.

    Original problem:
        A q = lambda M q

    Shift-invert problem:
        inv(A - sigma M) M q = mu q

    Eigenvalue relation:
        lambda = sigma + 1 / mu

    Parameters
    ----------
    r : float
        Drawdown ratio.
    N : int
        Number of mesh intervals.
    sigma : float
        Shift value.
    mu_tol : float
        Tolerance for removing mu near zero.

    Returns
    -------
    real_part : float
        Real part of leading eigenvalue.
    imag_part : float
        Absolute imaginary part of leading eigenvalue.
    """

    A, M = build_matrices(r, N)

    # Shift-invert matrix
    #
    # B_shift = inv(A - sigma M) M
    #
    # This matches the slide directly.
    K = A - sigma * M
    B_shift = np.linalg.inv(K) @ M

    mu_vals = np.linalg.eigvals(B_shift)

    # Remove zero eigenvalues of transformed problem.
    # These correspond to infinite eigenvalues of original generalized problem.
    mu_vals = mu_vals[np.abs(mu_vals) > tol]

    eigvals = sigma + 1.0 / mu_vals
    eigvals = eigvals[np.isfinite(eigvals)]

    lam = eigvals[np.argmax(np.real(eigvals))]

    return lam.real, abs(lam.imag)


def critical_draw_ratio(N=300, sigma=1.0, *, max_iter=100, tol=1e-6):
    r0 = 15
    r1 = 16
    for _ in range(max_iter):
        f0, _ = solve_problem(r0, N, sigma)
        f1, _ = solve_problem(r1, N, sigma)

        if abs(f1) < tol:
            return r1

        r2 = r1 - f1 * (r1 - r0) / (f1 - f0)
        r0 = r1
        r1 = r2
    else:
        raise Exception("ERROR::Iteration Failed!")


if __name__ == "__main__":
    # shift value
    # leading eigenvalue의 real part가 0 근처이므로 sigma = 0 사용
    sigma = 1.0
    rs = [15.0, 20.0, 20.218, 22.0, 25.0]
    N = 300
    for r in rs:
        real, imag = solve_problem(r, N, sigma)
        print(f"{r:5.3f} {real:15.4f} {imag:18.3f}")

    print()
    print("---------------------------------------")
    print()

    Ns = [100, 200, 300, 400, 500, 700, 1000, 1100, 1200, 2000]
    for N in Ns:
        r_crit = critical_draw_ratio(N)
        print(f"{N:4.0f} {r_crit:10.4f}")
