"""
SEIR Outbreak Simulator
========================

A compartmental epidemic model (Kermack-McKendrick SEIR/SIR) integrated
numerically with scipy's solve_ivp, extended with:

  * A piecewise-constant contact-rate reduction to represent an
    intervention (e.g. social distancing) that starts on a given day.
  * An optional vaccination compartment (SEIRV) where a fraction of the
    population is immunized before the outbreak begins, according to a
    vaccine coverage and efficacy.

The module can be used as a library (see `simulate`) or as a CLI
(see `main` / `python seir_simulator.py --help`).
"""

from __future__ import annotations

import argparse
import dataclasses
from typing import List, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Model parameters and results
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ModelParams:
    """Epidemiological parameters for the SEIR(V) model."""

    population: float = 1_000_000.0
    r0: float = 2.5
    incubation_period: float = 5.2   # days, 1/sigma. Ignored for model="sir".
    infectious_period: float = 7.0   # days, 1/gamma
    initial_infected: float = 1.0
    initial_exposed: float = 0.0     # only used for model="seir"/"seirv"
    model: str = "seir"              # "sir", "seir", or "seirv"

    # Intervention: a single step-change reduction in the contact rate,
    # starting on `intervention_day` and held for the rest of the run.
    intervention_day: float | None = None
    intervention_reduction: float = 0.0   # fraction in [0, 1)

    # Vaccination (SEIRV only): a fraction `vaccination_coverage` of the
    # population is vaccinated before t=0. Each vaccinated person is
    # protected with probability `vaccination_efficacy`; protected people
    # start in the immune V compartment, the rest re-enter S.
    vaccination_coverage: float = 0.0     # fraction in [0, 1]
    vaccination_efficacy: float = 1.0     # fraction in [0, 1]

    def __post_init__(self) -> None:
        if self.model not in ("sir", "seir", "seirv"):
            raise ValueError(f"model must be 'sir', 'seir' or 'seirv', got {self.model!r}")
        if self.population <= 0:
            raise ValueError("population must be positive")
        if self.r0 < 0:
            raise ValueError("r0 must be non-negative")
        if self.incubation_period <= 0:
            raise ValueError("incubation_period must be positive")
        if self.infectious_period <= 0:
            raise ValueError("infectious_period must be positive")
        if not (0.0 <= self.intervention_reduction < 1.0):
            raise ValueError("intervention_reduction must be in [0, 1)")
        if not (0.0 <= self.vaccination_coverage <= 1.0):
            raise ValueError("vaccination_coverage must be in [0, 1]")
        if not (0.0 <= self.vaccination_efficacy <= 1.0):
            raise ValueError("vaccination_efficacy must be in [0, 1]")
        if self.initial_infected < 0 or self.initial_exposed < 0:
            raise ValueError("initial_infected/initial_exposed must be non-negative")
        if self.initial_infected + self.initial_exposed > self.population:
            raise ValueError("initial_infected + initial_exposed cannot exceed population")

    @property
    def sigma(self) -> float:
        """Rate of progression from exposed to infectious (1/incubation period)."""
        return 1.0 / self.incubation_period

    @property
    def gamma(self) -> float:
        """Recovery rate (1/infectious period)."""
        return 1.0 / self.infectious_period

    @property
    def beta0(self) -> float:
        """Baseline transmission rate, beta = R0 * gamma."""
        return self.r0 * self.gamma


@dataclasses.dataclass
class SimulationResult:
    """Time series output of a simulation run plus summary statistics."""

    t: np.ndarray
    S: np.ndarray
    E: np.ndarray
    I: np.ndarray
    R: np.ndarray
    V: np.ndarray
    params: ModelParams

    @property
    def peak_infected(self) -> float:
        """Peak simultaneous number of infectious individuals."""
        return float(np.max(self.I))

    @property
    def peak_day(self) -> float:
        """Day on which the infectious compartment peaks."""
        return float(self.t[int(np.argmax(self.I))])

    @property
    def total_attack_rate(self) -> float:
        """Fraction of the population that is ultimately infected (R_final / N)."""
        return float(self.R[-1] / self.params.population)

    @property
    def final_susceptible_fraction(self) -> float:
        """Fraction of the population that escaped infection (S_final / N)."""
        return float(self.S[-1] / self.params.population)


# ---------------------------------------------------------------------------
# Contact-rate intervention
# ---------------------------------------------------------------------------

def contact_rate_multiplier(t: float, intervention_day: float | None, reduction: float) -> float:
    """Piecewise-constant multiplier applied to beta.

    Returns 1.0 before `intervention_day` (no intervention in effect) and
    `1 - reduction` from `intervention_day` onward, modeling a step-change
    in the contact rate such as social distancing or a lockdown.
    """
    if intervention_day is None or t < intervention_day:
        return 1.0
    return 1.0 - reduction


# ---------------------------------------------------------------------------
# ODE right-hand sides
# ---------------------------------------------------------------------------

def _seir_rhs(t: float, y: np.ndarray, params: ModelParams) -> List[float]:
    S, E, I, R = y
    N = params.population
    beta = params.beta0 * contact_rate_multiplier(t, params.intervention_day, params.intervention_reduction)
    sigma = params.sigma
    gamma = params.gamma

    new_infections = beta * S * I / N
    dS = -new_infections
    dE = new_infections - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I
    return [dS, dE, dI, dR]


def _sir_rhs(t: float, y: np.ndarray, params: ModelParams) -> List[float]:
    S, I, R = y
    N = params.population
    beta = params.beta0 * contact_rate_multiplier(t, params.intervention_day, params.intervention_reduction)
    gamma = params.gamma

    new_infections = beta * S * I / N
    dS = -new_infections
    dI = new_infections - gamma * I
    dR = gamma * I
    return [dS, dI, dR]


def _seirv_rhs(t: float, y: np.ndarray, params: ModelParams) -> List[float]:
    S, E, I, R, V = y
    N = params.population
    beta = params.beta0 * contact_rate_multiplier(t, params.intervention_day, params.intervention_reduction)
    sigma = params.sigma
    gamma = params.gamma

    # V is fully immune and does not transmit or acquire infection: it only
    # shrinks the initial susceptible pool (see `_initial_state`).
    new_infections = beta * S * I / N
    dS = -new_infections
    dE = new_infections - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I
    dV = 0.0
    return [dS, dE, dI, dR, dV]


# ---------------------------------------------------------------------------
# Simulation driver
# ---------------------------------------------------------------------------

def _initial_state(params: ModelParams) -> Tuple[np.ndarray, Sequence[str]]:
    N = params.population
    I0 = params.initial_infected
    E0 = params.initial_exposed if params.model != "sir" else 0.0

    if params.model == "seirv":
        vaccinated = params.vaccination_coverage * N
        immune = vaccinated * params.vaccination_efficacy
        V0 = immune
        S0 = N - I0 - E0 - V0
        if S0 < 0:
            raise ValueError("vaccination_coverage * efficacy leaves no room for initial S")
        y0 = np.array([S0, E0, I0, 0.0, V0], dtype=float)
        compartments = ("S", "E", "I", "R", "V")
    elif params.model == "seir":
        S0 = N - I0 - E0
        y0 = np.array([S0, E0, I0, 0.0], dtype=float)
        compartments = ("S", "E", "I", "R")
    else:  # sir
        S0 = N - I0
        y0 = np.array([S0, I0, 0.0], dtype=float)
        compartments = ("S", "I", "R")
    return y0, compartments


def simulate(params: ModelParams, days: float = 180.0, points_per_day: int = 4) -> SimulationResult:
    """Integrate the SEIR/SIR/SEIRV ODE system over [0, days].

    Uses scipy's `solve_ivp` (adaptive-step Runge-Kutta, RK45) with dense
    output evaluated on a uniform grid for plotting and peak detection.
    """
    if days <= 0:
        raise ValueError("days must be positive")

    y0, compartments = _initial_state(params)
    rhs = {"sir": _sir_rhs, "seir": _seir_rhs, "seirv": _seirv_rhs}[params.model]

    n_eval = max(int(days * points_per_day), 2)
    t_eval = np.linspace(0.0, days, n_eval)

    solution = solve_ivp(
        fun=rhs,
        t_span=(0.0, days),
        y0=y0,
        args=(params,),
        method="RK45",
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-6,
        max_step=1.0,
    )
    if not solution.success:
        raise RuntimeError(f"ODE integration failed: {solution.message}")

    values = dict(zip(compartments, solution.y))
    zeros = np.zeros_like(solution.t)
    return SimulationResult(
        t=solution.t,
        S=values["S"],
        E=values.get("E", zeros),
        I=values["I"],
        R=values["R"],
        V=values.get("V", zeros),
        params=params,
    )


# ---------------------------------------------------------------------------
# Scenario comparison and plotting
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Scenario:
    name: str
    params: ModelParams


def build_scenarios(base: ModelParams) -> List[Scenario]:
    """Build a baseline scenario plus one scenario per active intervention.

    Always includes a no-intervention baseline so results are comparable.
    Adds a social-distancing scenario if an intervention was configured,
    a vaccination scenario if vaccination coverage was configured (forcing
    model to "seirv"), and a combined scenario if both are configured.
    """
    scenarios: List[Scenario] = []

    baseline = dataclasses.replace(
        base, intervention_day=None, intervention_reduction=0.0,
        vaccination_coverage=0.0,
    )
    scenarios.append(Scenario("Baseline (no intervention)", baseline))

    has_intervention = base.intervention_day is not None and base.intervention_reduction > 0.0
    has_vaccination = base.vaccination_coverage > 0.0

    if has_intervention:
        distancing = dataclasses.replace(base, vaccination_coverage=0.0)
        scenarios.append(Scenario(
            f"Social distancing (day {base.intervention_day:g}, "
            f"-{base.intervention_reduction:.0%} contact)",
            distancing,
        ))

    if has_vaccination:
        vaccination = dataclasses.replace(
            base, intervention_day=None, intervention_reduction=0.0,
            model="seirv",
        )
        scenarios.append(Scenario(
            f"Vaccination ({base.vaccination_coverage:.0%} coverage, "
            f"{base.vaccination_efficacy:.0%} efficacy)",
            vaccination,
        ))

    if has_intervention and has_vaccination:
        combined = dataclasses.replace(base, model="seirv")
        scenarios.append(Scenario("Social distancing + vaccination", combined))

    return scenarios


def run_scenarios(scenarios: Sequence[Scenario], days: float) -> List[Tuple[str, SimulationResult]]:
    return [(s.name, simulate(s.params, days=days)) for s in scenarios]


def plot_scenarios(results: Sequence[Tuple[str, SimulationResult]], output_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, result in results:
        pct_infected = 100.0 * result.I / result.params.population
        ax.plot(result.t, pct_infected, label=name, linewidth=2)

    ax.set_xlabel("Day")
    ax.set_ylabel("Infectious population (%)")
    ax.set_title("Epidemic curve: scenario comparison")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "SEIR Outbreak Simulator: numerically integrate a "
            "Kermack-McKendrick SEIR/SIR compartmental model with optional "
            "social-distancing and vaccination scenarios."
        )
    )
    parser.add_argument("--population", type=float, default=1_000_000.0,
                         help="Total population size N (default: 1,000,000)")
    parser.add_argument("--r0", type=float, default=2.5,
                         help="Basic reproduction number R0 (default: 2.5)")
    parser.add_argument("--incubation-period", type=float, default=5.2,
                         help="Mean incubation period in days, 1/sigma (default: 5.2). Ignored for --model sir")
    parser.add_argument("--infectious-period", type=float, default=7.0,
                         help="Mean infectious period in days, 1/gamma (default: 7.0)")
    parser.add_argument("--initial-infected", type=float, default=1.0,
                         help="Number of initially infectious individuals (default: 1)")
    parser.add_argument("--initial-exposed", type=float, default=0.0,
                         help="Number of initially exposed individuals (default: 0)")
    parser.add_argument("--model", choices=["sir", "seir"], default="seir",
                         help="Compartmental model to use as the base model (default: seir). "
                              "Automatically upgraded to seirv when vaccination coverage > 0.")
    parser.add_argument("--days", type=float, default=180.0,
                         help="Number of days to simulate (default: 180)")

    parser.add_argument("--intervention-day", type=float, default=None,
                         help="Day on which a contact-rate reduction (e.g. social distancing) begins")
    parser.add_argument("--intervention-reduction", type=float, default=0.0,
                         help="Fractional reduction in contact rate from intervention-day onward, "
                              "e.g. 0.5 for a 50%% reduction (default: 0.0)")

    parser.add_argument("--vaccination-coverage", type=float, default=0.0,
                         help="Fraction of the population vaccinated before the outbreak (default: 0.0)")
    parser.add_argument("--vaccination-efficacy", type=float, default=0.9,
                         help="Vaccine efficacy, fraction of vaccinated people who become immune (default: 0.9)")

    parser.add_argument("--output", type=str, default="epidemic_curve.png",
                         help="Path to save the comparison plot (default: epidemic_curve.png)")
    parser.add_argument("--no-plot", action="store_true",
                         help="Skip generating the plot")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    base_params = ModelParams(
        population=args.population,
        r0=args.r0,
        incubation_period=args.incubation_period,
        infectious_period=args.infectious_period,
        initial_infected=args.initial_infected,
        initial_exposed=args.initial_exposed,
        model=args.model,
        intervention_day=args.intervention_day,
        intervention_reduction=args.intervention_reduction,
        vaccination_coverage=args.vaccination_coverage,
        vaccination_efficacy=args.vaccination_efficacy,
    )

    scenarios = build_scenarios(base_params)
    results = run_scenarios(scenarios, days=args.days)

    header = f"{'Scenario':<45}{'Peak infected':>16}{'Peak day':>12}{'Attack rate':>14}"
    print(header)
    print("-" * len(header))
    for name, result in results:
        print(
            f"{name:<45}"
            f"{result.peak_infected:>16,.0f}"
            f"{result.peak_day:>12.1f}"
            f"{result.total_attack_rate:>13.1%}"
        )

    if not args.no_plot:
        plot_scenarios(results, args.output)
        print(f"\nSaved comparison plot to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
