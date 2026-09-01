import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rnt_current import current_series_and_gradient
from rnt_modes import (
    derivative_modes_from_theta,
    initialise_modes,
    initial_theta_from_base,
    reconstruct_potential,
)
from rnt_optimise import (
    finite_difference_gradient_check,
    optimiser,
)


def main():
    A_max = 400
    step = 50

    L_domain = 1.0
    D_diff = 1.0
    w_freq = 1.0
    gamma_rtp = 1.0

    N_nu_val = 100
    nu_height = 1.0

    parameterization_kind = "imaginary"

    a_indices_full, U_base = initialise_modes(
        A=A_max,
        L=L_domain,
    )
    x_initial = np.linspace(
        0.0,
        L_domain,
        500,
        endpoint=False,
    )

    U_initial_x = reconstruct_potential(
        U_modes=U_base,
        a_indices=a_indices_full,
        L=L_domain,
        x_values=x_initial,
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(
        x_initial / L_domain,
        U_initial_x.real,
        label="Initial sawtooth",
    )

    ax.set_xlabel("x/L")
    ax.set_ylabel("U(x)")
    ax.set_title("Initial potential")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    plt.show()

    theta_initial = initial_theta_from_base(
        U_base=U_base,
        a_indices_full=a_indices_full,
        A_k=A_max,
        kind=parameterization_kind,
    )

    U_initial, dU_initial = (
    derivative_modes_from_theta(
        theta=theta_initial,
        A_k=A_max,
        a_indices_full=a_indices_full,
        kind=parameterization_kind,
    )
)

    initial_details = current_series_and_gradient(
        a_indices=a_indices_full,
        U_modes=U_initial,
        dU_dtheta=dU_initial,
        A_k=A_max,
        D=D_diff,
        w=w_freq,
        L=L_domain,
        gamma=gamma_rtp,
        N_nu=N_nu_val,
        nu=nu_height,
        return_details=True,
    )

    print(
        "Initial current =",
        initial_details["current"],
    )

    print(
        "Gradient shape =",
        initial_details["gradient"].shape,
    )

    print(
        "Gradient =",
        initial_details["gradient"],
    )

    print(
        "Radius =",
        initial_details["radius"],
    )
    coeffs = initial_details["coeffs"]

    print("\nFirst perturbative coefficients:\n")

    for n in range(
        min(12, len(coeffs)),
    ):
        print(
            f"J^({n}) = "
            f"{coeffs[n]: .12e}"
        )

    even_orders = np.arange(
        0,
        len(coeffs),
        2,
    )

    odd_orders = np.arange(
        1,
        len(coeffs),
        2,
    )

    max_even = np.max(
        np.abs(
            coeffs[even_orders]
        )
    )

    max_odd = np.max(
        np.abs(
            coeffs[odd_orders]
        )
    )

    print(
        "\nMaximum even coefficient =",
        max_even,
    )

    print(
        "Maximum odd coefficient =",
        max_odd,
    )

    print(
        "Even/odd coefficient ratio =",
        max_even / max(
            max_odd,
            1e-30,
        ),
    )
    def finite_difference_gradient_check(
        theta,
        objective,
        epsilon=2e-6,
    ):
        theta = np.asarray(
            theta,
            dtype=float,
        )

        analytic = objective(theta)["gradient"]

        finite_difference = np.zeros_like(
            theta,
            dtype=float,
        )

        for p in range(len(theta)):
            theta_plus = theta.copy()
            theta_minus = theta.copy()

            theta_plus[p] += epsilon
            theta_minus[p] -= epsilon

            J_plus = objective(theta_plus)["current"]
            J_minus = objective(theta_minus)["current"]

            finite_difference[p] = (
                J_plus - J_minus
            ) / (2.0 * epsilon)

        difference = (
            analytic
            - finite_difference
        )

        relative_error = (
            np.linalg.norm(difference)
            / max(
                np.linalg.norm(
                    finite_difference
                ),
                1e-14,
            )
        )

        return {
            "analytic": analytic,
            "finite_difference": finite_difference,
            "relative_error": relative_error,
            "max_absolute_error": np.max(
                np.abs(difference)
            ),
        }
    A_audit = min(
        3,
        A_max,
    )

    a_indices_audit = np.arange(
        -A_audit,
        A_audit + 1,
    )

    theta_audit = initial_theta_from_base(
        U_base=U_base,
        a_indices_full=a_indices_full,
        A_k=A_audit,
        kind="full",
    )


    def audit_objective(theta):
        U_audit, dU_audit = (
            derivative_modes_from_theta(
                theta=theta,
                A_k=A_audit,
                a_indices_full=a_indices_audit,
                kind="full",
            )
        )

        details = current_series_and_gradient(
            a_indices=a_indices_audit,
            U_modes=U_audit,
            dU_dtheta=dU_audit,
            A_k=A_audit,
            D=D_diff,
            w=w_freq,
            L=L_domain,
            gamma=gamma_rtp,
            N_nu=min(N_nu_val, 15),
            nu=nu_height,
            return_details=True,
        )

        return {
            "current": details["current"],
            "gradient": details["gradient"],
        }
    gradient_audit = finite_difference_gradient_check(
        theta=theta_audit,
        objective=audit_objective,
        epsilon=2e-6,
    )

    print(
        "\n===== Gradient audit ====="
    )

    print(
        "Relative gradient error =",
        gradient_audit["relative_error"],
    )

    print(
        "Maximum absolute gradient error =",
        gradient_audit["max_absolute_error"],
    )

    print(
        "Analytic gradient =",
        gradient_audit["analytic"],
    )

    print(
        "Finite-difference gradient =",
        gradient_audit["finite_difference"],
    )
    start = time.time()
    theta_opt, a_indices_full, U_base, stage_results = (
        optimiser(
            A_max=A_max,
            step=step,
            L_domain=L_domain,
            D_diff=D_diff,
            w_freq=w_freq,
            gamma_rtp=gamma_rtp,
            N_nu_val=N_nu_val,
            nu_height=nu_height,
            parameterization_kind=parameterization_kind,
        )
    )
    print(
        "Elapsed:",
        time.time() - start,
        "seconds"
    )
    a_indices_final = np.arange(
        -A_max,
        A_max + 1,
    )

    U_opt, dU_opt = (
        derivative_modes_from_theta(
            theta=theta_opt,
            A_k=A_max,
            a_indices_full=a_indices_final,
            kind=parameterization_kind,
        )
    )

    final_details = current_series_and_gradient(
        a_indices=a_indices_final,
        U_modes=U_opt,
        dU_dtheta=dU_opt,
        A_k=A_max,
        D=D_diff,
        w=w_freq,
        L=L_domain,
        gamma=gamma_rtp,
        N_nu=N_nu_val,
        nu=nu_height,
        return_details=True,
    )

    print(
        "\n===== Final result ====="
    )

    print(
        "Final current =",
        final_details["current"],
    )

    print(
        "Final gradient norm =",
        np.linalg.norm(
            final_details["gradient"],
            ord=np.inf,
        ),
    )

    print(
        "Final radius =",
        final_details["radius"],
    )
    table_rows = []

    for result in stage_results:
        info = result["frprmn_info"]

        table_rows.append(
            {
                "stage": result["stage"],
                "A": result["A_k"],
                "J_initial": result["J_initial"],
                "J_final": result["J_final"],
                "improvement": (
                    result["J_final"]
                    - result["J_initial"]
                ),
                "gradient_inf": result[
                    "gradient_norm"
                ],
                "iterations": info[
                    "iterations"
                ],
                "radius": result[
                    "radius_used"
                ],
                "inside_radius": result[
                    "inside_radius"
                ],
                "status": info[
                    "stopped_reason"
                ],
            }
        )

    results_table = pd.DataFrame(
        table_rows,
    )

    print(
        "\n===== Modal continuation results =====\n"
    )

    print(
        results_table.to_string(
            index=False,
            formatters={
                "J_initial": "{:.6e}".format,
                "J_final": "{:.6e}".format,
                "improvement": "{:.6e}".format,
                "gradient_inf": "{:.3e}".format,
                "radius": "{:.6e}".format,
            },
        )
    )
    x_plot = np.linspace(
        0.0,
        L_domain,
        2000,
        endpoint=False,
    )

    U_opt_x = reconstruct_potential(
        U_modes=U_opt,
        a_indices=a_indices_final,
        L=L_domain,
        x_values=x_plot,
    )
    final_info = stage_results[-1][
        "frprmn_info"
    ]

    history = pd.DataFrame(
        final_info["history"]
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 4),
    )

    axes[0].plot(
        results_table["A"],
        results_table["J_initial"],
        "o--",
        label="initial",
    )

    axes[0].plot(
        results_table["A"],
        results_table["J_final"],
        "s-",
        label="final",
    )

    axes[0].set_xlabel(
        "Maximum active mode A"
    )

    axes[0].set_ylabel(
        "Current J"
    )

    axes[0].set_title(
        "Modal continuation"
    )

    axes[0].grid(
        alpha=0.3,
    )

    axes[0].legend()

    axes[1].plot(
        x_plot / L_domain,
        U_opt_x.real,
        linewidth=2.0,
    )

    axes[1].set_xlabel(
        "x/L"
    )

    axes[1].set_ylabel(
        "U(x)"
    )

    axes[1].set_title(
        f"Final potential, A={A_max}"
    )

    axes[1].grid(
        alpha=0.3,
    )

    axes[2].plot(
        history["iteration"],
        history["J"],
        "o-",
    )

    axes[2].set_xlabel(
        "frprmn iteration"
    )

    axes[2].set_ylabel(
        "Current J"
    )

    axes[2].set_title(
        "Optimisation history"
    )

    axes[2].grid(
        alpha=0.3,
    )

    fig.tight_layout()
    plt.show()
    plt.figure(
        figsize=(6, 4),
    )

    plt.semilogy(
        history["iteration"],
        history["grad_norm"],
        "o-",
    )

    plt.xlabel(
        "frprmn iteration"
    )

    plt.ylabel(
        "||grad||inf"
    )

    plt.title(
        f"Gradient history, A={A_max}"
    )

    plt.grid(
        alpha=0.3,
    )

    plt.tight_layout()
    plt.show()
    orders = [
        3,
        5,
        7,
        9,
        11,
        13,
    ]

    order_currents = []

    for N_test in orders:
        test_details = current_series_and_gradient(
            a_indices=a_indices_final,
            U_modes=U_opt,
            dU_dtheta=dU_opt,
            A_k=A_max,
            D=D_diff,
            w=w_freq,
            L=L_domain,
            gamma=gamma_rtp,
            N_nu=N_test,
            nu=nu_height,
            return_details=True,
        )

        order_currents.append(
            test_details["partials"][N_test]
        )

    order_table = pd.DataFrame(
        {
            "N_nu": orders,
            "J": order_currents,
        }
    )

    print(
        "\n===== Perturbative-order diagnostic =====\n"
    )

    print(
        order_table.to_string(
            index=False,
            formatters={
                "J": "{:.8e}".format,
            },
        )
    )
    A_values = np.array(
        [
            result["A_k"]
            for result in stage_results
        ]
    )

    J_values = np.array(
        [
            result["J_final"]
            for result in stage_results
        ]
    )

    plt.figure(
        figsize=(7, 4),
    )

    plt.plot(
        A_values,
        J_values,
        "o-",
        linewidth=2,
        markersize=6,
    )

    plt.xlabel(
        "Active Fourier cutoff A"
    )

    plt.ylabel(
        "Optimised current J"
    )

    plt.title(
        "Optimised current versus number of modes"
    )

    plt.grid(
        alpha=0.3,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()