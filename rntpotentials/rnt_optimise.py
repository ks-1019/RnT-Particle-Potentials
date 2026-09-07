import numpy as np

from rnt_current import current_series_and_gradient
from rnt_modes import (
    derivative_modes_from_theta,
    initialise_modes,
    initial_theta_from_base,
    pad_theta,
)
def frprmn_maximise(
    theta0,
    J_and_grad,
    max_iter=500,
    gtol=1e-6,
    ftol=1e-10,
    initial_step=1e-2,
    beta_max=1.0,
    ascent_eta=1e-4,
    armijo_c1=1e-4,
    backtrack_factor=0.5,
    verbose=True,
):
    theta = np.asarray(
        theta0,
        dtype=float,
    ).copy()

    J, grad_J = J_and_grad(theta)

    J = float(J)
    grad_J = np.asarray(
        grad_J,
        dtype=float,
    )

    if not np.isfinite(J):
        raise ValueError(
            "Initial current is not finite."
        )

    if not np.all(
        np.isfinite(grad_J)
    ):
        raise ValueError(
            "Initial gradient contains NaN or inf."
        )

    direction = grad_J.copy()

    history = [
        {
            "iteration": 0,
            "J": J,
            "grad_norm": np.linalg.norm(
                grad_J,
                ord=np.inf,
            ),
            "alpha": 0.0,
            "beta": 0.0,
        }
    ]

    converged = False
    stopped_reason = (
        "maximum iterations reached"
    )

    for iteration in range(
        1,
        max_iter + 1,
    ):
        grad_norm = np.linalg.norm(
            grad_J,
            ord=np.inf,
        )

        if grad_norm < gtol:
            converged = True
            stopped_reason = (
                "gradient tolerance"
            )
            break

        slope = float(
            np.dot(
                grad_J,
                direction,
            )
        )

        if slope <= (
            ascent_eta
            * np.dot(
                grad_J,
                grad_J,
            )
        ):
            direction = grad_J.copy()

            slope = float(
                np.dot(
                    grad_J,
                    direction,
                )
            )

        alpha = min(initial_step, 0.2 / max(np.linalg.norm(direction), 1e-12))
        accepted = False

        for _ in range(30):
            theta_trial = (
                theta
                + alpha * direction
            )

            J_trial, grad_trial = (
                J_and_grad(
                    theta_trial,
                )
            )

            if (
                np.isfinite(J_trial)
                and np.all(
                    np.isfinite(grad_trial)
                )
                and J_trial
                >= J
                + armijo_c1
                * alpha
                * slope
            ):
                accepted = True
                break

            alpha *= backtrack_factor

        if not accepted:
            stopped_reason = (
                "Armijo line-search failure"
            )
            break

        theta_new = (
            theta
            + alpha * direction
        )

        J_new, grad_new = (
            J_and_grad(theta_new)
        )

        J_new = float(J_new)
        grad_new = np.asarray(
            grad_new,
            dtype=float,
        )

        if not np.isfinite(J_new):
            stopped_reason = (
                "non-finite current after update"
            )
            break

        if not np.all(
            np.isfinite(grad_new)
        ):
            stopped_reason = (
                "non-finite gradient after update"
            )
            break

        denominator = max(
            float(
                np.dot(
                    grad_J,
                    grad_J,
                )
            ),
            1e-30,
        )

        beta_pr = float(
            np.dot(
                grad_new,
                grad_new - grad_J,
            )
            / denominator
        )

        beta = min(
            beta_max,
            max(
                0.0,
                beta_pr,
            )
        )

        direction_new = (
            grad_new
            + beta * direction
        )

        if np.dot(
            grad_new,
            direction_new,
        ) <= (
            ascent_eta
            * np.dot(
                grad_new,
                grad_new,
            )
        ):
            direction_new = grad_new.copy()
            beta = 0.0

        relative_change = (
            abs(J_new - J)
            / max(
                1.0,
                abs(J),
            )
        )

        history.append(
            {
                "iteration": iteration,
                "J": J_new,
                "grad_norm": np.linalg.norm(
                    grad_new,
                    ord=np.inf,
                ),
                "alpha": alpha,
                "beta": beta,
            }
        )

        if verbose:
            print(
                f"frprmn iteration {iteration:3d}: "
                f"J={J_new:.12e}, "
                f"alpha={alpha:.4e}, "
                f"beta={beta:.4e}, "
                f"||grad||inf="
                f"{np.linalg.norm(grad_new, ord=np.inf):.4e}"
            )

        theta = theta_new
        J = J_new
        grad_J = grad_new
        direction = direction_new

        if (
          relative_change < ftol
          and np.linalg.norm(grad_new, ord=np.inf) < gtol
          ):
            converged = True
            stopped_reason = (
                "objective tolerance"
            )
            break

    info = {
        "converged": converged,
        "stopped_reason": stopped_reason,
        "history": history,
        "gradient_norm": np.linalg.norm(
            grad_J,
            ord=np.inf,
        ),
        "iterations": len(history) - 1,
    }

    return theta, J, info

def optimiser(
    A_max,
    step,
    L_domain,
    D_diff,
    w_freq,
    gamma_rtp,
    N_nu_val,
    nu_height,
    parameterization_kind="imaginary",
    theta_seed=None
):
    a_indices_full, U_base = initialise_modes(
        A=A_max,
        L=L_domain,
        plot_initial=False,
    )

    n_stages = A_max // step
    theta_opt = None
    previous_A = None
    stage_results = []

    for k in range(n_stages):
        stage = k + 1
        A_k = (k + 1) * step

        a_indices_stage = np.arange(
            -A_k,
            A_k + 1,
        )

        if parameterization_kind == "imaginary":
            dim_k = A_k
        elif parameterization_kind == "full":
            dim_k = 2 * A_k - 1
        else:
            raise ValueError(
                "Invalid parameterization_kind."
            )

        if theta_opt is None:
            if theta_seed is None:
                theta_init = initial_theta_from_base(
                    U_base=U_base,
                    a_indices_full=a_indices_full,
                    A_k=A_k,
                    kind=parameterization_kind,
                )

            elif parameterization_kind == "imaginary":
                theta_init = theta_seed[:A_k].copy()

            else:
                raise NotImplementedError(
                    "theta_seed continuation is currently implemented "
                    "only for parameterization_kind='imaginary'."
                )

        else:
            theta_init = pad_theta(
                theta_old=theta_opt,
                old_A=previous_A,
                new_A=A_k,
                kind=parameterization_kind,
    )

        def J_and_grad(theta_vec):
            U_modes_now, dU_now = (
                derivative_modes_from_theta(
                    theta=theta_vec,
                    A_k=A_k,
                    a_indices_full=a_indices_stage,
                    kind=parameterization_kind,
                )
            )

            details = current_series_and_gradient(
                a_indices=a_indices_stage,
                U_modes=U_modes_now,
                dU_dtheta=dU_now,
                A_k=A_k,
                D=D_diff,
                w=w_freq,
                L=L_domain,
                gamma=gamma_rtp,
                N_nu=N_nu_val,
                nu=nu_height,
                return_details=True,
            )

            return (
                details["current"],
                details["gradient"],
            )

        J_initial, grad_initial = (
            J_and_grad(theta_init)
        )

        print(
            f"\nStarting stage "
            f"{stage}/{n_stages}"
        )

        print(
            f"A_k = {A_k}"
        )

        print(
            f"Number of parameters = {dim_k}"
        )

        print(
            f"Initial current = "
            f"{J_initial:.12e}"
        )

        print(
            "Initial gradient norm = "
            f"{np.linalg.norm(grad_initial, ord=np.inf):.6e}"
        )

        theta_candidate, J_candidate, info = (
            frprmn_maximise(
                theta0=theta_init,
                J_and_grad=J_and_grad,
                max_iter=500,
                gtol=1e-6,
                ftol=1e-10,
                initial_step=1.0,
                beta_max=1.0,
                ascent_eta=1e-4,
                armijo_c1=1e-4,
                verbose=True,
            )
        )

        U_candidate, dU_candidate = (
            derivative_modes_from_theta(
                theta=theta_candidate,
                A_k=A_k,
                a_indices_full=a_indices_stage,
                kind=parameterization_kind,
            )
        )

        candidate_details = current_series_and_gradient(
            a_indices=a_indices_stage,
            U_modes=U_candidate,
            dU_dtheta=dU_candidate,
            A_k=A_k,
            D=D_diff,
            w=w_freq,
            L=L_domain,
            gamma=gamma_rtp,
            N_nu=N_nu_val,
            nu=nu_height,
            return_details=True,
        )

        J_candidate = candidate_details["current"]
        coeffs_candidate = candidate_details["coeffs"]
        radius_info = candidate_details["radius"]

        improves = (
            J_candidate
            >= J_initial
        )

        finite_candidate = (
            np.isfinite(J_candidate)
            and np.all(
                np.isfinite(
                    coeffs_candidate
                )
            )
        )

        candidate_stable = (
            finite_candidate
            and improves
            and radius_info["inside_radius"]
        )

        if candidate_stable:
            theta_opt = theta_candidate.copy()
            J_best = J_candidate
            accepted = True
            status = "accepted"
        else:
            print(
                "Rejecting candidate: "
                f"J={J_candidate:.6e}, "
                f"radius="
                f"{radius_info['radius_used']:.6e}, "
                f"|nu|={abs(nu_height):.6e}, "
                f"inside_radius="
                f"{radius_info['inside_radius']}, "
                f"improves={improves}"
            )

            theta_opt = theta_init.copy()
            J_best = J_initial
            accepted = False
            status = (
                "rejected: outside radius "
                "or no improvement"
            )

        J_final, grad_final = J_and_grad(theta_opt)

        result = {
            "stage": stage,
            "A_k": A_k,
            "theta_initial": theta_init.copy(),
            "theta_opt": theta_opt.copy(),
            "J_initial": J_initial,
            "J_best": J_best,
            "J_final": J_final,
            "gradient_norm": np.linalg.norm(
                grad_final,
                ord=np.inf,
            ),
            "frprmn_info": info,
            "radius_current": radius_info[
                "radius_current"
            ],
            "radius_derivative": radius_info[
                "radius_derivative"
            ],
            "radius_used": radius_info[
                "radius_used"
            ],
            "inside_radius": radius_info[
                "inside_radius"
            ],
            "accepted": accepted,
            "status": status,
        }

        stage_results.append(result)

        print(
            f"Finished stage {stage}: "
            f"A_k={A_k}, "
            f"J_initial={J_initial:.12e}, "
            f"J_final={J_final:.12e}, "
            f"iterations={info['iterations']}, "
            f"reason={info['stopped_reason']}, "
            f"radius={radius_info['radius_used']:.6e}, "
            f"inside={radius_info['inside_radius']}, "
            f"status={status}"
        )

        previous_A = A_k

    return (
        theta_opt,
        a_indices_full,
        U_base,
        stage_results,
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