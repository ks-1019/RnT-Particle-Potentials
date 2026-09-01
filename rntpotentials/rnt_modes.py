import numpy as np


def initialise_modes(
    A,
    L=1.0,
    plot_initial=False,
):
    a_indices_full = np.arange(
        -A,
        A + 1,
    )

    U_base = np.zeros(
        2 * A + 1,
        dtype=np.complex128,
    )

    # initial sawtooth potential as was in the paper
    nonzero = a_indices_full != 0

    U_base[nonzero] = (
        1j * L**2
        / (
            2.0
            * np.pi
            * a_indices_full[nonzero]
        )
    )

    return a_indices_full, U_base


def U_modes(
    theta,
    U_base,
    A_k,
    a_indices_full,
    kind="imaginary",
):
    del U_base

    theta = np.asarray(
        theta,
        dtype=float,
    )

    a_indices_full = np.asarray(
        a_indices_full,
        dtype=int,
    )

    expected = np.arange(
        -A_k,
        A_k + 1,
    )

    if not np.array_equal(
        a_indices_full,
        expected,
    ):
        raise ValueError(
            "a_indices_full must equal "
            "np.arange(-A_k,A_k+1)."
        )

    index = {
        int(a): i
        for i, a in enumerate(a_indices_full)
    }

    U_active = np.zeros(
        2 * A_k + 1,
        dtype=np.complex128,
    )

    if kind == "imaginary":
        if len(theta) != A_k:
            raise ValueError(
                f"Expected {A_k} parameters, "
                f"received {len(theta)}."
            )

        for a in range(1, A_k + 1):
            value = 1j * theta[a - 1]

            U_active[index[a]] = value
            U_active[index[-a]] = np.conjugate(value)

    elif kind == "full":
        if len(theta) != 2 * A_k - 1:
            raise ValueError(
                f"Expected {2 * A_k - 1} parameters, "
                f"received {len(theta)}."
            )

        # U1 is purely imaginary.
        U_active[index[1]] = 1j * theta[0]
        U_active[index[-1]] = -1j * theta[0]

        # Real parts of U2,...,UA.
        for a in range(2, A_k + 1):
            p = a - 1

            U_active[index[a]] += theta[p]
            U_active[index[-a]] += theta[p]

        # Imaginary parts of U2,...,UA.
        for a in range(2, A_k + 1):
            p = A_k + a - 2

            U_active[index[a]] += 1j * theta[p]
            U_active[index[-a]] -= 1j * theta[p]

    else:
        raise ValueError(
            "kind must be 'imaginary' or 'full'."
        )

    return U_active


def derivative_modes_from_theta(
    theta,
    A_k,
    a_indices_full,
    kind="imaginary",
):
    theta = np.asarray(
        theta,
        dtype=float,
    )

    a_indices_full = np.asarray(
        a_indices_full,
        dtype=int,
    )

    expected = np.arange(
        -A_k,
        A_k + 1,
    )

    if not np.array_equal(
        a_indices_full,
        expected,
    ):
        raise ValueError(
            "a_indices_full must equal "
            "np.arange(-A_k,A_k+1)."
        )

    index = {
        int(a): i
        for i, a in enumerate(a_indices_full)
    }

    U = np.zeros(
        2 * A_k + 1,
        dtype=np.complex128,
    )

    if kind == "imaginary":
        if len(theta) != A_k:
            raise ValueError(
                f"Expected {A_k} parameters, "
                f"received {len(theta)}."
            )

        dU = np.zeros(
            (A_k, 2 * A_k + 1),
            dtype=np.complex128,
        )

        for a in range(1, A_k + 1):
            p = a - 1

            U[index[a]] = 1j * theta[p]
            U[index[-a]] = -1j * theta[p]

            dU[p, index[a]] = 1j
            dU[p, index[-a]] = -1j

    elif kind == "full":
        if len(theta) != 2 * A_k - 1:
            raise ValueError(
                f"Expected {2 * A_k - 1} parameters, "
                f"received {len(theta)}."
            )

        dU = np.zeros(
            (2 * A_k - 1, 2 * A_k + 1),
            dtype=np.complex128,
        )

        U[index[1]] = 1j * theta[0]
        U[index[-1]] = -1j * theta[0]

        dU[0, index[1]] = 1j
        dU[0, index[-1]] = -1j

        for a in range(2, A_k + 1):
            p = a - 1

            U[index[a]] += theta[p]
            U[index[-a]] += theta[p]

            dU[p, index[a]] = 1.0
            dU[p, index[-a]] = 1.0

        for a in range(2, A_k + 1):
            p = A_k + a - 2

            U[index[a]] += 1j * theta[p]
            U[index[-a]] -= 1j * theta[p]

            dU[p, index[a]] = 1j
            dU[p, index[-a]] = -1j

    else:
        raise ValueError(
            "kind must be 'imaginary' or 'full'."
        )

    return U, dU


def initial_theta_from_base(
    U_base,
    a_indices_full,
    A_k,
    kind="imaginary",
):
    U_base = np.asarray(
        U_base,
        dtype=np.complex128,
    )

    a_indices_full = np.asarray(
        a_indices_full,
        dtype=int,
    )

    full_index = {
        int(a): i
        for i, a in enumerate(a_indices_full)
    }

    if kind == "imaginary":
        theta = np.zeros(
            A_k,
            dtype=float,
        )

        for a in range(1, A_k + 1):
            theta[a - 1] = U_base[full_index[a]].imag

        return theta

    if kind == "full":
        theta = np.zeros(
            2 * A_k - 1,
            dtype=float,
        )

        theta[0] = U_base[full_index[1]].imag

        for a in range(2, A_k + 1):
            theta[a - 1] = U_base[full_index[a]].real
            theta[A_k + a - 2] = U_base[full_index[a]].imag

        return theta

    raise ValueError(
        "kind must be 'imaginary' or 'full'."
    )


def pad_theta(
    theta_old,
    old_A,
    new_A,
    kind,
):
    old_indices = np.arange(
        -old_A,
        old_A + 1,
    )

    old_U = U_modes(
        theta=theta_old,
        U_base=None,
        A_k=old_A,
        a_indices_full=old_indices,
        kind=kind,
    )

    old_index = {
        int(a): i
        for i, a in enumerate(old_indices)
    }

    if kind == "imaginary":
        theta_new = np.zeros(
            new_A,
            dtype=float,
        )

        for a in range(1, old_A + 1):
            theta_new[a - 1] = old_U[old_index[a]].imag

        return theta_new

    if kind == "full":
        theta_new = np.zeros(
            2 * new_A - 1,
            dtype=float,
        )

        theta_new[0] = old_U[old_index[1]].imag

        for a in range(2, old_A + 1):
            theta_new[a - 1] = old_U[old_index[a]].real
            theta_new[new_A + a - 2] = old_U[old_index[a]].imag

        return theta_new

    raise ValueError(
        "kind must be 'imaginary' or 'full'."
    )


def reconstruct_potential(
    U_modes,
    a_indices,
    L,
    x_values,
):
    """
    Reconstruct U(x) from Fourier modes U_a.

    Uses:
        U(x) = (1 / L) * sum_a exp(i k_a x) U_a
        k_a = 2 pi a / L
    """
    U_modes = np.asarray(
        U_modes,
        dtype=np.complex128,
    )

    a_indices = np.asarray(
        a_indices,
        dtype=int,
    )

    x_values = np.asarray(
        x_values,
        dtype=float,
    )

    if U_modes.shape != a_indices.shape:
        raise ValueError(
            "U_modes and a_indices must have the same shape."
        )

    k_values = 2.0 * np.pi * a_indices / L

    return (
        np.sum(
            U_modes[:, None]
            * np.exp(
                1j
                * k_values[:, None]
                * x_values[None, :]
            ),
            axis=0,
        )
        / L
    )