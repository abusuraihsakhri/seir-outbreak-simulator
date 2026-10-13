"""Tests for seir_simulator.py, run with plain assert statements.

Run with: python test_seir_simulator.py
"""

import numpy as np

from seir_simulator import (
    ModelParams,
    build_scenarios,
    contact_rate_multiplier,
    run_scenarios,
    simulate,
)


def test_population_is_conserved():
    """S + E + I + R must equal N at every time step (no births/deaths)."""
    params = ModelParams(population=100_000, r0=3.0, incubation_period=5.0,
                          infectious_period=6.0, initial_infected=20, model="seir")
    result = simulate(params, days=120)
    total = result.S + result.E + result.I + result.R
    assert np.allclose(total, params.population, rtol=1e-6), (
        f"population not conserved: min={total.min()}, max={total.max()}, expected={params.population}"
    )


def test_outbreak_grows_when_r0_above_one_and_dies_when_below_one():
    """A classic SIR sanity check: R0 > 1 produces a substantial epidemic,
    R0 < 1 keeps infections near the seed value and the outbreak fades out."""
    growing = ModelParams(population=100_000, r0=2.5, infectious_period=7.0,
                           initial_infected=10, model="sir")
    result_growing = simulate(growing, days=200)
    assert result_growing.peak_infected > 10 * growing.initial_infected, (
        "expected substantial growth in infections when R0 > 1"
    )
    assert result_growing.total_attack_rate > 0.5, (
        f"expected a large attack rate for R0=2.5, got {result_growing.total_attack_rate:.3f}"
    )

    dying = ModelParams(population=100_000, r0=0.5, infectious_period=7.0,
                         initial_infected=10, model="sir")
    result_dying = simulate(dying, days=200)
    assert result_dying.peak_infected <= dying.initial_infected + 1.0, (
        f"expected infections to stay near the seed value for R0=0.5, "
        f"got peak={result_dying.peak_infected}"
    )
    assert result_dying.total_attack_rate < 0.05, (
        f"expected a small attack rate for R0=0.5, got {result_dying.total_attack_rate:.3f}"
    )


def test_intervention_reduces_peak_and_delays_it():
    """A mid-outbreak contact-rate reduction should flatten and delay the peak
    relative to an identical run with no intervention."""
    baseline = ModelParams(population=500_000, r0=3.0, incubation_period=5.2,
                            infectious_period=7.0, initial_infected=50, model="seir")
    intervened = ModelParams(population=500_000, r0=3.0, incubation_period=5.2,
                              infectious_period=7.0, initial_infected=50, model="seir",
                              intervention_day=30, intervention_reduction=0.6)

    result_baseline = simulate(baseline, days=250)
    result_intervened = simulate(intervened, days=250)

    assert result_intervened.peak_infected < result_baseline.peak_infected, (
        "intervention should reduce the peak number of infectious individuals"
    )
    assert result_intervened.peak_day >= result_baseline.peak_day, (
        "intervention should delay (or not advance) the peak"
    )
    assert result_intervened.total_attack_rate < result_baseline.total_attack_rate, (
        "intervention should reduce the total attack rate"
    )


def test_contact_rate_multiplier_step_change():
    """The intervention multiplier is 1.0 before the intervention day and
    (1 - reduction) from that day onward."""
    assert contact_rate_multiplier(0.0, intervention_day=None, reduction=0.5) == 1.0
    assert contact_rate_multiplier(100.0, intervention_day=None, reduction=0.5) == 1.0
    assert contact_rate_multiplier(9.99, intervention_day=10.0, reduction=0.4) == 1.0
    assert contact_rate_multiplier(10.0, intervention_day=10.0, reduction=0.4) == 0.6
    assert contact_rate_multiplier(50.0, intervention_day=10.0, reduction=0.4) == 0.6


def test_vaccination_reduces_attack_rate_and_conserves_population():
    """Vaccinating a large, effective fraction of the population before the
    outbreak should sharply cut the attack rate, and S+E+I+R+V must equal N."""
    unvaccinated = ModelParams(population=200_000, r0=3.0, incubation_period=5.0,
                                infectious_period=7.0, initial_infected=25, model="seirv",
                                vaccination_coverage=0.0)
    vaccinated = ModelParams(population=200_000, r0=3.0, incubation_period=5.0,
                              infectious_period=7.0, initial_infected=25, model="seirv",
                              vaccination_coverage=0.8, vaccination_efficacy=0.9)

    result_unvaccinated = simulate(unvaccinated, days=200)
    result_vaccinated = simulate(vaccinated, days=200)

    total_vacc = result_vaccinated.S + result_vaccinated.E + result_vaccinated.I \
        + result_vaccinated.R + result_vaccinated.V
    assert np.allclose(total_vacc, vaccinated.population, rtol=1e-6), (
        "S+E+I+R+V must equal N in the SEIRV model"
    )

    assert result_vaccinated.total_attack_rate < result_unvaccinated.total_attack_rate, (
        "vaccination should reduce the total attack rate"
    )
    # 0.8 coverage * 0.9 efficacy = 72% of the population starts immune, so the
    # remaining susceptible pool is small enough that the outbreak should be minor.
    assert result_vaccinated.total_attack_rate < 0.2, (
        f"expected a small attack rate with high vaccination coverage, "
        f"got {result_vaccinated.total_attack_rate:.3f}"
    )


def test_peak_and_attack_rate_are_internally_consistent():
    """Peak values and cumulative infected fraction must match compartments."""
    params = ModelParams(population=50_000, r0=2.0, infectious_period=5.0,
                          initial_infected=5, model="sir")
    result = simulate(params, days=150)

    assert result.peak_infected == np.max(result.I)
    assert result.peak_day == result.t[int(np.argmax(result.I))]
    assert abs(result.total_attack_rate -
               (result.E[-1] + result.I[-1] + result.R[-1]) / params.population) < 1e-12


def test_build_scenarios_includes_baseline_and_requested_interventions():
    base = ModelParams(population=10_000, r0=2.5, initial_infected=5, model="seir",
                        intervention_day=20, intervention_reduction=0.5,
                        vaccination_coverage=0.3, vaccination_efficacy=0.8)
    scenarios = build_scenarios(base)
    names = [s.name for s in scenarios]

    assert any("Baseline" in n for n in names)
    assert any("distancing" in n for n in names)
    assert any("Vaccination" in n for n in names)
    assert any("+" in n for n in names)  # combined scenario
    assert len(scenarios) == 4

    results = run_scenarios(scenarios, days=100)
    assert len(results) == 4
    for name, result in results:
        assert result.t[0] == 0.0
        assert result.t[-1] == 100.0


def test_invalid_params_raise():
    try:
        ModelParams(population=-5)
        assert False, "expected ValueError for negative population"
    except ValueError:
        pass

    try:
        ModelParams(model="bogus")
        assert False, "expected ValueError for invalid model"
    except ValueError:
        pass

    try:
        ModelParams(intervention_reduction=1.5)
        assert False, "expected ValueError for reduction >= 1"
    except ValueError:
        pass


if __name__ == "__main__":
    tests = [
        test_population_is_conserved,
        test_outbreak_grows_when_r0_above_one_and_dies_when_below_one,
        test_intervention_reduces_peak_and_delays_it,
        test_contact_rate_multiplier_step_change,
        test_vaccination_reduces_attack_rate_and_conserves_population,
        test_peak_and_attack_rate_are_internally_consistent,
        test_build_scenarios_includes_baseline_and_requested_interventions,
        test_invalid_params_raise,
    ]
    for test in tests:
        test()
        print(f"PASSED: {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")

def test_unresolved_infections_are_included_in_attack_rate():
    params = ModelParams(population=1_000, r0=0.0, model="sir", initial_infected=10)
    result = simulate(params, days=1)
    assert abs(result.total_attack_rate - 0.01) < 1e-10
    assert result.I[-1] > 0


def test_intervention_on_day_zero_matches_constant_reduced_r0():
    reduced = ModelParams(population=20_000, model="seir", initial_infected=30,
                          r0=3.0, intervention_day=0, intervention_reduction=0.5)
    constant = ModelParams(population=20_000, model="seir", initial_infected=30, r0=1.5)
    a = simulate(reduced, days=60)
    b = simulate(constant, days=60)
    assert np.allclose(a.I, b.I, atol=1e-4, rtol=1e-5)


def test_input_validation_rejects_nan_inf_and_invalid_grid():
    import pytest
    for val in [float("nan"), float("inf"), -float("inf")]:
        with pytest.raises(ValueError):
            ModelParams(r0=val)
        with pytest.raises(ValueError):
            ModelParams(intervention_day=val)
        with pytest.raises(ValueError):
            simulate(ModelParams(), days=val)
    with pytest.raises(ValueError):
        simulate(ModelParams(), days=20, points_per_day=0)
    with pytest.raises(ValueError):
        simulate(ModelParams(), days=100_000, points_per_day=100)
