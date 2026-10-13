import test from "node:test";
import assert from "node:assert/strict";
import { simulate, buildScenarios, validate } from "./app.mjs";

const base = {
  model: "seir", population: 100000, r0: 2.5,
  incubationPeriod: 5.2, infectiousPeriod: 7,
  initialInfected: 10, initialExposed: 0,
  interventionDay: null, interventionReduction: 0,
  vaccinationCoverage: 0, vaccinationEfficacy: 0.9
};

test("SIR epidemic grows above threshold and dies out below it", () => {
  const grows = simulate({ ...base, model: "sir" }, 180);
  const dies = simulate({ ...base, model: "sir", r0: 0.5 }, 180);
  assert.ok(grows.peakInfected > 100);
  assert.ok(grows.attackRate > 0.5);
  assert.ok(dies.peakInfected <= 10.01);
  assert.ok(dies.attackRate < 0.01);
});

test("each SEIRV compartment is nonnegative and population is conserved", () => {
  const p = { ...base, model: "seirv", vaccinationCoverage: 0.4 };
  for (const x of simulate(p, 180).series) {
    const values = [x.S, x.E, x.I, x.R, x.V];
    assert.ok(values.every(v => v >= -1e-6));
    assert.ok(Math.abs(values.reduce((a, b) => a + b, 0) - p.population) < 1e-5);
  }
});

test("contact-rate reduction decreases cumulative infections", () => {
  const baseline = simulate(base, 180);
  const intervention = simulate({
    ...base, interventionDay: 20, interventionReduction: 0.6
  }, 180);
  assert.ok(intervention.attackRate < baseline.attackRate);
  assert.ok(intervention.peakInfected < baseline.peakInfected);
});

test("intervention from day zero equals a reduced baseline R0", () => {
  const a = simulate({ ...base, interventionDay: 0, interventionReduction: 0.5 }, 120);
  const b = simulate({ ...base, r0: 1.25 }, 120);
  assert.ok(Math.abs(a.attackRate - b.attackRate) < 1e-8);
});

test("vaccination protection reduces infection burden", () => {
  const a = simulate({ ...base, model: "seirv" }, 180);
  const b = simulate({ ...base, model: "seirv", vaccinationCoverage: 0.8 }, 180);
  assert.ok(b.attackRate < a.attackRate);
});

test("attack rate includes unresolved infections at the horizon", () => {
  const out = simulate({ ...base, r0: 0 }, 1);
  assert.ok(Math.abs(out.attackRate - 10 / base.population) < 1e-10);
  assert.ok(out.series[1].I > 0);
});

test("buildScenarios creates requested combinations", () => {
  assert.equal(buildScenarios(base).length, 1);
  const p = { ...base, vaccinationCoverage: 0.3,
    interventionDay: 30, interventionReduction: 0.5 };
  assert.deepEqual(buildScenarios(p).map(s => s.name),
    ["Baseline", "Contact reduction", "Vaccination", "Combined"]);
});

test("invalid numerical inputs and impossible initial state are rejected", () => {
  assert.throws(() => validate({ ...base, r0: NaN }, 180), RangeError);
  assert.throws(() => simulate({ ...base, population: Infinity }), RangeError);
  assert.throws(() => simulate({ ...base, interventionDay: -1 }), RangeError);
  assert.throws(() => simulate({ ...base, model: "seirv", vaccinationCoverage: 1 }), RangeError);
  assert.throws(() => simulate(base, 1.5), RangeError);
  assert.throws(() => simulate(base, 731), RangeError);
});
