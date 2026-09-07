import numpy as np
from scipy.signal import fftconvolve
from scipy.special import factorial


def check_radius_of_convergence(
    coeffs,
    nu=1.0,
):
    coeffs = np.asarray(
        coeffs,
        dtype=float,
    )

    odd = [
        m
        for m in range(3, len(coeffs), 2)
        if np.isfinite(coeffs[m])
        and abs(coeffs[m]) > 0.0
    ]

    if not odd:
        return {
            "radius_current": np.inf,
            "radius_derivative": np.inf,
            "radius_used": np.inf,
            "inside_radius": True,
            "radii_current": np.array([]),
            "radii_derivative": np.array([]),
        }

    largest = odd[-1]

    radius_current = (
        abs(float(coeffs[largest]))
        ** (-1.0 / largest)
    )

    radii_derivative = np.array(
        [
            abs(float(m * coeffs[m]))
            ** (-1.0 / (m - 1))
            for m in odd
        ],
        dtype=float,
    )

    radius_derivative = float(
        np.min(radii_derivative)
    )

    radius_used = min(
        radius_current,
        radius_derivative,
    )

    return {
        "radius_current": float(radius_current),
        "radius_derivative": radius_derivative,
        "radius_used": float(radius_used),
        "inside_radius": bool(
            abs(nu) < radius_used
        ),
        "radii_current": np.array(
            [radius_current],
            dtype=float,
        ),
        "radii_derivative": radii_derivative,
    }


def _convolve_rows(
    left,
    right,
):
    return fftconvolve(
        left,
        right,
        mode="full",
        axes=1,
    )


def _derivative_recurrence_step(
    rho,
    mom,
    drho,
    dmom,
    W,
    dW,
    c_hat,
    c_none,
    c_und,
    chunk_size=16,
):
    P = drho.shape[0]

    output_length = (
        drho.shape[1]
        + len(W)
        - 1
    )

    if len(c_hat) != output_length:
        raise ValueError(
            "c_hat length does not match "
            "the convolution output."
        )

    drho_next = np.empty(
        (P, output_length),
        dtype=np.complex128,
    )

    dmom_next = np.empty(
        (P, output_length),
        dtype=np.complex128,
    )

    for start in range(
        0,
        P,
        chunk_size,
    ):
        stop = min(
            start + chunk_size,
            P,
        )

        sl = slice(
            start,
            stop,
        )

        d_rho_w = (
            _convolve_rows(
                drho[sl],
                W[None, :],
            )
            + _convolve_rows(
                rho[None, :],
                dW[sl],
            )
        )

        d_mom_w = (
            _convolve_rows(
                dmom[sl],
                W[None, :],
            )
            + _convolve_rows(
                mom[None, :],
                dW[sl],
            )
        )

        drho_next[sl] = (
            c_hat[None, :] * d_mom_w
            - c_none[None, :] * d_rho_w
        )

        dmom_next[sl] = (
            c_hat[None, :] * d_rho_w
            - c_und[None, :] * d_mom_w
        )

    return (
        drho_next,
        dmom_next,
    )


def current_series_and_gradient(
    a_indices,
    U_modes,
    dU_dtheta,
    A_k,
    D,
    w,
    L,
    gamma,
    N_nu,
    nu,
    return_coeffs=False,
    return_details=False,
):
    a_indices = np.asarray(
        a_indices,
        dtype=int,
    )

    expected = np.arange(
        -A_k,
        A_k + 1,
    )

    if not np.array_equal(
        a_indices,
        expected,
    ):
        raise ValueError(
            "a_indices must equal "
            "np.arange(-A_k,A_k+1)."
        )

    U_modes = np.asarray(
        U_modes,
        dtype=np.complex128,
    )

    dU_dtheta = np.asarray(
        dU_dtheta,
        dtype=np.complex128,
    )

    N_modes = 2 * A_k + 1
    P = dU_dtheta.shape[0]

    if U_modes.shape != (N_modes,):
        raise ValueError(
            "U_modes has the wrong shape."
        )

    if dU_dtheta.shape != (
        P,
        N_modes,
    ):
        raise ValueError(
            "dU_dtheta has the wrong shape."
        )

    k_potential = (
        2.0
        * np.pi
        * a_indices
        / L
    )

    W = (
        k_potential
        * U_modes
        / L
    )

    dW = (
        dU_dtheta
        * k_potential[None, :]
        / L
    )

    U_minus = U_modes[::-1]
    dU_minus = dU_dtheta[:, ::-1]
    prefactor = 1j * k_potential

    rho = np.array(
        [1.0 + 0.0j],
    )

    mom = np.array(
        [0.0 + 0.0j],
    )

    drho = np.zeros(
        (P, 1),
        dtype=np.complex128,
    )

    dmom = np.zeros(
        (P, 1),
        dtype=np.complex128,
    )

    terms = np.zeros(
        N_nu + 1,
        dtype=float,
    )

    gradient_terms = np.zeros(
        (N_nu + 1, P),
        dtype=float,
    )

    for recurrence_order in range(
        1,
        N_nu,
    ):
        support = recurrence_order * A_k

        output_modes = np.arange(
            -support,
            support + 1,
        )

        k = (
            2.0
            * np.pi
            * output_modes
            / L
        )

        denominator = (
            D * D * k * k
            + w * w
            + 2.0 * gamma * D
        )

        c_hat = np.zeros(
            len(k),
            dtype=np.complex128,
        )

        c_none = np.zeros(
            len(k),
            dtype=np.complex128,
        )

        c_und = np.zeros(
            len(k),
            dtype=np.complex128,
        )

        nonzero = output_modes != 0

        c_hat[nonzero] = (
            1j
            * w
            / denominator[nonzero]
        )

        c_none[nonzero] = (
            D * k[nonzero]**2
            + 2.0 * gamma
        ) / (
            k[nonzero]
            * denominator[nonzero]
        )

        c_und[nonzero] = (
            D
            * k[nonzero]
            / denominator[nonzero]
        )

        rho_w = np.convolve(
            rho,
            W,
        )

        mom_w = np.convolve(
            mom,
            W,
        )

        rho_next = (
            c_hat * mom_w
            - c_none * rho_w
        )

        mom_next = (
            c_hat * rho_w
            - c_und * mom_w
        )

        drho_next, dmom_next = (
            _derivative_recurrence_step(
                rho=rho,
                mom=mom,
                drho=drho,
                dmom=dmom,
                W=W,
                dW=dW,
                c_hat=c_hat,
                c_none=c_none,
                c_und=c_und,
            )
        )

        offset = support

        rho_slice = rho_next[
            offset - A_k:
            offset + A_k + 1
        ]

        drho_slice = drho_next[
            :,
            offset - A_k:
            offset + A_k + 1
        ]

        current_order = recurrence_order + 1

        terms[current_order] = (
            np.sum(
                prefactor
                * U_minus
                * rho_slice
            ).real
            / (L * L)
        )

        gradient_terms[current_order] = (
            np.sum(
                prefactor[None, :]
                * (
                    dU_minus
                    * rho_slice[None, :]
                    + U_minus[None, :]
                    * drho_slice
                ),
                axis=1,
            ).real
            / (L * L)
        )

        rho = rho_next
        mom = mom_next
        drho = drho_next
        dmom = dmom_next

    powers = nu ** np.arange(
        N_nu + 1,
    )

    odd_orders = np.arange(
        1,
        N_nu + 1,
        2,
    )
    resummation = "borel"
    if resummation == "series":
      current = float(
          np.sum(
              terms[odd_orders]
              * powers[odd_orders]
          )
      )
      gradient = np.sum(
          gradient_terms[odd_orders]
          * powers[odd_orders, None],
          axis=0,
      )
    elif resummation == "borel":
      current, gradient = borel_resum(
          terms=terms,
          gradient_terms=gradient_terms,
          nu=nu,
      )

    partials = np.zeros(
        N_nu + 1,
        dtype=float,
    )

    running = 0.0

    for n in range(
        N_nu + 1,
    ):
        if n % 2 == 1:
            running += (
                terms[n]
                * powers[n]
            )

        partials[n] = running

    radius_info = (
        check_radius_of_convergence(
            coeffs=terms,
            nu=nu,
        )
    )

    if return_details:
        return {
            "current": float(current),
            "gradient": np.asarray(
                gradient,
                dtype=float,
            ),
            "coeffs": terms,
            "gradient_coeffs": gradient_terms,
            "partials": partials,
            "radius": radius_info,
        }

    if return_coeffs:
        return (
            float(current),
            np.asarray(
                gradient,
                dtype=float,
            ),
            terms.copy(),
        )

    return (
        float(current),
        np.asarray(
            gradient,
            dtype=float,
        ),
    )


def borel_resum(
    terms,
    gradient_terms,
    nu,
    cutoff=20.0,
    n_quad=200,
):
    s = np.linspace(0.0, cutoff, n_quad)

    B = np.zeros_like(s)
    dB = np.zeros(
        (len(s), gradient_terms.shape[1])
    )

    for n in range(len(terms)):
        coeff = terms[n] / factorial(n)

        B += coeff * (nu*s)**n

        dB += (
            gradient_terms[n]
            / factorial(n)
        )[None, :] * (nu*s)[:, None]**n

    weight = np.exp(-s)

    current = np.trapezoid(weight*B, s)

    gradient = np.trapezoid(
        weight[:, None]*dB,
        s,
        axis=0,
    )

    return current, gradient