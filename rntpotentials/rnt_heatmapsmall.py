from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rnt_current import current_series_and_gradient
from rnt_modes import derivative_modes_from_theta
from rnt_optimise import optimiser


def evaluate_final_solution(
    theta,
    Pe,
    Qe,
    A_max,
    N_nu_val,
    parameterization_kind="imaginary",
):
    """
    Evaluate current, gradient, coefficients, and radius information
    for an A=A_max optimised potential.

    We use D = L = 1, hence:
        w = Pe
        gamma = Qe.
    """

    D_diff = 1.0
    L_domain = 1.0
    w_freq = Pe
    gamma_rtp = Qe

    a_indices = np.arange(-A_max, A_max + 1)

    U_modes, dU_dtheta = derivative_modes_from_theta(
        theta=theta,
        A_k=A_max,
        a_indices_full=a_indices,
        kind=parameterization_kind,
    )

    return current_series_and_gradient(
        a_indices=a_indices,
        U_modes=U_modes,
        dU_dtheta=dU_dtheta,
        A_k=A_max,
        D=D_diff,
        w=w_freq,
        L=L_domain,
        gamma=gamma_rtp,
        N_nu=N_nu_val,
        nu=1.0,
        return_details=True,
    )


def run_one_point(
    Pe,
    Qe,
    A_max,
    step,
    N_nu_val,
    theta_seed=None,
    parameterization_kind="imaginary",
    verbose=False,
):
    """
    Run the complete A=step, 2*step, ..., A_max continuation
    for one point in (Pe, Qe) space.

    theta_seed:
        The final A=A_max theta from a nearby previously solved point.
        If None, the optimiser starts from the standard sawtooth.
    """

    theta_opt, _, _, stage_results = optimiser(
        A_max=A_max,
        step=step,
        L_domain=1.0,
        D_diff=1.0,
        w_freq=Pe,
        gamma_rtp=Qe,
        N_nu_val=N_nu_val,
        nu_height=1.0,
        parameterization_kind=parameterization_kind,
        theta_seed=theta_seed,
    )

    final_details = evaluate_final_solution(
        theta=theta_opt,
        Pe=Pe,
        Qe=Qe,
        A_max=A_max,
        N_nu_val=N_nu_val,
        parameterization_kind=parameterization_kind,
    )

    final_stage = stage_results[-1]
    gradient_norm = np.linalg.norm(
        final_details["gradient"],
        ord=np.inf,
    )

    return {
        "theta": theta_opt,
        "current": final_details["current"],
        "gradient_norm": gradient_norm,
        "coeffs": final_details["coeffs"],
        "radius": final_details["radius"],
        "stage_results": stage_results,
        "accepted": final_stage["accepted"],
        "status": final_stage["status"],
        "iterations": final_stage["frprmn_info"]["iterations"],
        "stopped_reason": (
            final_stage["frprmn_info"]["stopped_reason"]
        ),
    }


def make_heatmap(
    results_table,
    Pe_values,
    Qe_values,
):
    """
    Convert a table of results into a 2D current array.

    Only points whose final stage was accepted and whose final
    gradient is sufficiently small are displayed.
    """

    current_grid = np.full(
        (len(Qe_values), len(Pe_values)),
        np.nan,
        dtype=float,
    )

    for _, row in results_table.iterrows():
        i = int(row["Pe_index"])
        j = int(row["Qe_index"])

        valid = (
            bool(row["accepted"])
            and bool(row["inside_direct_radius"])
            and np.isfinite(row["current"])
        )

        if valid:
            current_grid[j, i] = row["current"]

    return current_grid


def main():
    # --------------------------------------------------
    # Numerical settings used at every (Pe, Qe) point
    # --------------------------------------------------

    A_max = 50
    step = 5
    N_nu_val = 50

    parameterization_kind = "imaginary"

    gradient_tolerance = 1e-6

    # --------------------------------------------------
    # Parameter-space grid
    #
    # Initially used np.array([0.8, 1.0, 1.25]) for both
    # --------------------------------------------------

    Pe_values = np.geomspace(0.1, 10.0, 21)
    Qe_values = np.array([1.0])

    # first point solved from scratch
    Pe_anchor = 1.0
    Qe_anchor = 1.0

    # --------------------------------------------------
    # Output directory
    # --------------------------------------------------

    results_dir = Path("heatmap_results")
    modes_dir = results_dir / "modes"

    results_dir.mkdir(exist_ok=True)
    modes_dir.mkdir(exist_ok=True)

    # --------------------------------------------------
    # Store all results here.
    #
    # Each element is a plain dictionary suitable for a
    # DataFrame and CSV export.
    # --------------------------------------------------

    rows = []
    solved_thetas = {}

    # --------------------------------------------------
    # 1. Solve the anchor point from the sawtooth.
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print(
        f"Anchor optimisation: "
        f"Pe={Pe_anchor:.6g}, Qe={Qe_anchor:.6g}"
    )
    print("=" * 70)

    start = time.time()

    anchor_result = run_one_point(
        Pe=Pe_anchor,
        Qe=Qe_anchor,
        A_max=A_max,
        step=step,
        N_nu_val=N_nu_val,
        theta_seed=None,
        parameterization_kind=parameterization_kind,
    )

    elapsed = time.time() - start

    anchor_gradient_norm = anchor_result["gradient_norm"]

    solved_thetas[(Pe_anchor, Qe_anchor)] = (
        anchor_result["theta"].copy()
    )

    anchor_Pe_index = int(
        np.where(np.isclose(Pe_values, Pe_anchor))[0][0]
    )

    anchor_Qe_index = int(
        np.where(np.isclose(Qe_values, Qe_anchor))[0][0]
    )

    np.save(
        modes_dir / "theta_Pe_1_Qe_1.npy",
        anchor_result["theta"],
    )

    rows.append(
        {
            "Pe": Pe_anchor,
            "Qe": Qe_anchor,
            "Pe_index": anchor_Pe_index,
            "Qe_index": anchor_Qe_index,
            "current": anchor_result["current"],
            "gradient_norm": anchor_gradient_norm,
            "stationary": (
                anchor_gradient_norm < gradient_tolerance
            ),
            "accepted": anchor_result["accepted"],
            "radius_used": (
                anchor_result["radius"]["radius_used"]
            ),
            "inside_direct_radius": (
                anchor_result["radius"]["inside_radius"]
            ),
            "iterations": anchor_result["iterations"],
            "stopped_reason": (
                anchor_result["stopped_reason"]
            ),
            "status": anchor_result["status"],
            "parent_Pe": np.nan,
            "parent_Qe": np.nan,
            "elapsed_seconds": elapsed,
        }
    )

    print(
        f"Anchor current = "
        f"{anchor_result['current']:.12e}"
    )
    print(
        f"Anchor gradient norm = "
        f"{anchor_gradient_norm:.6e}"
    )
    print(
        f"Anchor radius = "
        f"{anchor_result['radius']['radius_used']:.6e}"
    )

    # --------------------------------------------------
    # 2. Construct a simple snake traversal.
    #
    # Start each row from a point solved in the previous
    # row, then move left-to-right or right-to-left.
    # --------------------------------------------------
    previous_theta = None
    previous_Pe = np.nan
    previous_Qe = np.nan
    for j, Qe in enumerate(Qe_values):
        if j % 2 == 0:
            Pe_indices = list(range(len(Pe_values)))
        else:
            Pe_indices = list(
                range(len(Pe_values) - 1, -1, -1)
            )

        for i in Pe_indices:
            Pe = Pe_values[i]

            # The anchor is already solved.
            if (
                np.isclose(Pe, Pe_anchor)
                and np.isclose(Qe, Qe_anchor)
            ):
                continue

            # NOT NEAREST NEIGHBOUR ALGORITHM - JUST GOES ALONG A LINE
            if previous_theta is None:
                theta_seed = None
                parent_Pe = np.nan
                parent_Qe = np.nan
            else:
                theta_seed = previous_theta.copy()
                parent_Pe = previous_Pe
                parent_Qe = previous_Qe

            print("\n" + "-" * 70)
            print(
                f"Optimising "
                f"Pe={Pe:.6g}, Qe={Qe:.6g}"
            )
            print(
                f"Initialised from "
                f"Pe={parent_Pe:.6g}, "
                f"Qe={parent_Qe:.6g}"
            )
            print("-" * 70)

            start = time.time()

            result = run_one_point(
                Pe=Pe,
                Qe=Qe,
                A_max=A_max,
                step=step,
                N_nu_val=N_nu_val,
                theta_seed=theta_seed,
                parameterization_kind=parameterization_kind,
            )

            elapsed = time.time() - start

            gradient_norm = result["gradient_norm"]

            solved_thetas[(Pe, Qe)] = (
                result["theta"].copy()
            )

            np.save(
                modes_dir
                / f"theta_Pe_{Pe:.6g}_Qe_{Qe:.6g}.npy",
                result["theta"],
            )
            previous_theta = result["theta"].copy()
            previous_Pe = Pe
            previous_Qe = Qe

            rows.append(
                {
                    "Pe": Pe,
                    "Qe": Qe,
                    "Pe_index": i,
                    "Qe_index": j,
                    "current": result["current"],
                    "gradient_norm": gradient_norm,
                    "stationary": (
                        gradient_norm < gradient_tolerance
                    ),
                    "accepted": result["accepted"],
                    "radius_used": (
                        result["radius"]["radius_used"]
                    ),
                    "inside_direct_radius": (
                        result["radius"]["inside_radius"]
                    ),
                    "iterations": result["iterations"],
                    "stopped_reason": (
                        result["stopped_reason"]
                    ),
                    "status": result["status"],
                    "parent_Pe": parent_Pe,
                    "parent_Qe": parent_Qe,
                    "elapsed_seconds": elapsed,
                }
            )

            print(
                f"Current = "
                f"{result['current']:.12e}"
            )
            print(
                f"Gradient norm = "
                f"{gradient_norm:.6e}"
            )
            print(
                f"Radius = "
                f"{result['radius']['radius_used']:.6e}"
            )
            print(
                f"Inside direct radius = "
                f"{result['radius']['inside_radius']}"
            )
            print(
                f"Accepted = {result['accepted']}"
            )
            print(
                f"Elapsed = {elapsed:.2f} s"
            )

            # Save after every point, so an interrupted run
            # still leaves usable results.
            partial_table = pd.DataFrame(rows)

            partial_table.to_csv(
                results_dir / "heatmap_results_partial.csv",
                index=False,
            )

# Final results table

    results_table = pd.DataFrame(rows)

    results_table = results_table.sort_values(
        ["Qe_index", "Pe_index"]
    ).reset_index(drop=True)

    results_table.to_csv(
        results_dir / "heatmap_results.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("Final heatmap table")
    print("=" * 70)

    print(
        results_table.to_string(
            index=False,
            formatters={
                "Pe": "{:.6g}".format,
                "Qe": "{:.6g}".format,
                "current": "{:.8e}".format,
                "gradient_norm": "{:.3e}".format,
                "radius_used": "{:.6e}".format,
                "elapsed_seconds": "{:.2f}".format,
            },
        )
    )

# Build heatmap

    current_grid = make_heatmap(
        results_table=results_table,
        Pe_values=Pe_values,
        Qe_values=Qe_values,
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    image = ax.pcolormesh(
        Pe_values,
        Qe_values,
        current_grid,
        shading="auto",
        cmap="viridis",
    )

    ax.set_xlabel("Peclet number Pe")
    ax.set_ylabel("Qeclet number Qe")
    ax.set_title(
        "Validated optimised current "
        r"$J L^2 / D$"
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    colourbar = fig.colorbar(
        image,
        ax=ax,
    )

    colourbar.set_label(
        r"Maximum current $J L^2 / D$"
    )

    fig.tight_layout()

    fig.savefig(
        results_dir / "current_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    # --------------------------------------------------
    # 5. Plot the numerical status of every point.
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 5))

    for _, row in results_table.iterrows():
        valid = (
            bool(row["accepted"])
            and bool(row["inside_direct_radius"])
            and np.isfinite(row["current"])
        )

        if valid:
            marker = "o"
            colour = "tab:green"
            label = "validated"
        else:
            marker = "x"
            colour = "tab:red"
            label = "not validated"

        ax.scatter(
            row["Pe"],
            row["Qe"],
            marker=marker,
            color=colour,
            s=80,
            label=label,
        )

    handles, labels = ax.get_legend_handles_labels()

    unique = dict(zip(labels, handles))

    ax.legend(
        unique.values(),
        unique.keys(),
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Peclet number Pe")
    ax.set_ylabel("Qeclet number Qe")
    ax.set_title("Heatmap-point optimisation status")
    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        results_dir / "heatmap_status.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    main()