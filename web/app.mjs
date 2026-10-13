// Deterministic SIR/SEIR/SEIRV simulation, executed entirely in the browser.
// The native JavaScript RK4 solver is independent of the Python SciPy solver.
const number = (v, label, min, max, inclusiveMax = true) => {
  if (typeof v !== "number" || !Number.isFinite(v) || v < min ||
      (inclusiveMax ? v > max : v >= max)) {
    throw new RangeError(`${label} must be a finite number between ${min} and ${max}${inclusiveMax ? "" : " (exclusive)"}.`);
  }
  return v;
};

export function validate(p, days) {
  if (!["sir", "seir", "seirv"].includes(p.model)) throw new RangeError("Select a valid model.");
  number(p.population, "Population", 1, 1e9);
  number(p.r0, "R₀", 0, 8);
  number(p.incubationPeriod, "Latent period", 1, 100);
  number(p.infectiousPeriod, "Infectious period", 1, 100);
  number(p.initialInfected, "Initial infectious", 0, p.population);
  number(p.initialExposed, "Initial exposed", 0, p.population);
  number(p.interventionReduction, "Contact reduction", 0, 1, false);
  number(p.vaccinationCoverage, "Vaccination coverage", 0, 1);
  number(p.vaccinationEfficacy, "Vaccination efficacy", 0, 1);
  if (p.interventionDay !== null) number(p.interventionDay, "Intervention day", 0, 730);
  number(days, "Simulation days", 1, 730);
  if (!Number.isInteger(days)) throw new RangeError("Simulation days must be an integer.");
  const exposed = p.model === "sir" ? 0 : p.initialExposed;
  const protectedCount = p.model === "seirv" ?
    p.population * p.vaccinationCoverage * p.vaccinationEfficacy : 0;
  if (exposed + p.initialInfected + protectedCount > p.population) {
    throw new RangeError("Initial exposed, infectious and protected people exceed the population.");
  }
}

const derivatives = (p, t, y) => {
  const [s, e, i] = y;
  const beta = p.r0 / p.infectiousPeriod *
    (p.interventionDay !== null && t >= p.interventionDay ? 1 - p.interventionReduction : 1);
  const incidence = beta * s * i / p.population;
  const progress = p.model === "sir" ? 0 : e / p.incubationPeriod;
  const recoveries = i / p.infectiousPeriod;
  return [-incidence, incidence - progress, progress - recoveries, recoveries, 0];
};

function rk4(p, t, y, h) {
  const a = derivatives(p, t, y);
  const b = derivatives(p, t + h / 2, y.map((v, i) => v + h * a[i] / 2));
  const c = derivatives(p, t + h / 2, y.map((v, i) => v + h * b[i] / 2));
  const d = derivatives(p, t + h, y.map((v, i) => v + h * c[i]));
  return y.map((v, i) => v + h * (a[i] + 2 * b[i] + 2 * c[i] + d[i]) / 6);
}

export function simulate(p, days = 180) {
  validate(p, days);
  const latent = p.model === "sir" ? 0 : p.initialExposed;
  const immune = p.model === "seirv" ?
    p.population * p.vaccinationCoverage * p.vaccinationEfficacy : 0;
  let y = [p.population - latent - p.initialInfected - immune,
    latent, p.initialInfected, 0, immune];
  let t = 0;
  const series = [{ day: 0, S: y[0], E: y[1], I: y[2], R: y[3], V: y[4] }];
  const maxRate = Math.max(p.r0 / p.infectiousPeriod,
    1 / p.infectiousPeriod, p.model === "sir" ? 0 : 1 / p.incubationPeriod);
  const stepLimit = Math.min(0.25, 0.1 / maxRate);
  for (let day = 1; day <= days; day++) {
    while (t < day - 1e-10) {
      let dt = Math.min(stepLimit, day - t);
      if (p.interventionDay !== null && p.interventionReduction > 0 &&
          t < p.interventionDay && t + dt > p.interventionDay) {
        dt = p.interventionDay - t;
      }
      // At the boundary RK4 must use the pre-intervention RHS for the
      // final left-hand step; otherwise its last stage applies the new rate.
      let next;
      if (p.interventionDay !== null && p.interventionReduction > 0 &&
          t < p.interventionDay && Math.abs(t + dt - p.interventionDay) < 1e-9) {
        next = rk4({ ...p, interventionDay: null }, t, y, dt);
      } else {
        next = rk4(p, t, y, dt);
      }
      if (next.some(v => !Number.isFinite(v) || v < -1e-6)) {
        throw new Error("Numerical integration produced an invalid compartment.");
      }
      y = next;
      t = Math.min(day, t + dt);
    }
    series.push({ day, S: y[0], E: y[1], I: y[2], R: y[3], V: y[4] });
  }
  const peak = series.reduce((best, s) => s.I > best.I ? s : best, series[0]);
  const final = series[series.length - 1];
  return {
    series, peakInfected: peak.I, peakDay: peak.day,
    attackRate: (final.E + final.I + final.R) / p.population,
    finalSusceptible: final.S / p.population
  };
}

export function buildScenarios(base) {
  const noIntervention = { ...base, interventionDay: null, interventionReduction: 0 };
  const noVaccination = { ...noIntervention, vaccinationCoverage: 0 };
  const result = [{ name: "Baseline", params: noVaccination }];
  const distancing = base.interventionDay !== null && base.interventionReduction > 0;
  const vaccination = base.vaccinationCoverage > 0;
  if (distancing) result.push({
    name: "Contact reduction",
    params: { ...base, vaccinationCoverage: 0 }
  });
  if (vaccination) result.push({
    name: "Vaccination",
    params: { ...noIntervention, model: "seirv" }
  });
  if (distancing && vaccination) result.push({
    name: "Combined",
    params: { ...base, model: "seirv" }
  });
  return result;
}

const svgNS = "http://www.w3.org/2000/svg";
const plotColors = ["#60a5fa", "#f59e0b", "#34d399", "#f472b6"];
let lastResults = [];

function draw(results, population, days) {
  const svg = document.getElementById("chart");
  svg.replaceChildren();
  const W = 940, H = 350, left = 67, right = 22, top = 20, bottom = 48;
  const cw = W - left - right, ch = H - top - bottom;
  const maxY = Math.max(0.001, ...results.flatMap(r => r.output.series.map(s => 100 * s.I / population))) * 1.1;
  const x = day => left + cw * day / days;
  const y = pct => top + ch * (1 - pct / maxY);
  function add(tag, attributes, text) {
    const el = document.createElementNS(svgNS, tag);
    for (const [key, value] of Object.entries(attributes)) el.setAttribute(key, String(value));
    if (text !== undefined) el.textContent = text;
    svg.appendChild(el);
    return el;
  }
  add("title", {}, "Infectious population percentage over time for selected scenarios");
  for (let i = 0; i <= 4; i++) {
    const pct = maxY * i / 4, yy = y(pct);
    add("line", { x1: left, y1: yy, x2: W - right, y2: yy, stroke: "#334155" });
    add("text", { x: left - 10, y: yy + 5, "text-anchor": "end", fill: "#cbd5e1", "font-size": 13 }, `${pct.toFixed(1)}%`);
  }
  for (let i = 0; i <= 4; i++) {
    const d = days * i / 4;
    add("text", { x: x(d), y: H - 19, "text-anchor": "middle", fill: "#cbd5e1", "font-size": 13 }, String(Math.round(d)));
  }
  add("text", { x: W / 2, y: H - 1, "text-anchor": "middle", fill: "#cbd5e1", "font-size": 13 }, "Day");
  for (const [i, result] of results.entries()) {
    const path = result.output.series.map((point, j) =>
      `${j ? "L" : "M"}${x(point.day).toFixed(2)},${y(100 * point.I / population).toFixed(2)}`).join(" ");
    add("path", { d: path, fill: "none", stroke: plotColors[i % plotColors.length],
      "stroke-width": 2.5, "stroke-linejoin": "round" });
  }
}

function getParams(form) {
  const val = name => Number(form.elements.namedItem(name).value);
  return {
    model: form.elements.namedItem("model").value,
    population: val("population"), r0: val("r0"),
    incubationPeriod: val("incubationPeriod"),
    infectiousPeriod: val("infectiousPeriod"),
    initialInfected: val("initialInfected"),
    initialExposed: val("initialExposed"),
    interventionDay: form.elements.namedItem("interventionDay").value === "" ?
      null : val("interventionDay"),
    interventionReduction: val("interventionReduction") / 100,
    vaccinationCoverage: val("vaccinationCoverage") / 100,
    vaccinationEfficacy: val("vaccinationEfficacy") / 100
  };
}

function render() {
  const form = document.getElementById("controls");
  const status = document.getElementById("status");
  const resultsBody = document.getElementById("results");
  try {
    const p = getParams(form);
    const days = Number(form.elements.namedItem("days").value);
    validate(p, days);
    const scenarios = buildScenarios(p);
    lastResults = scenarios.map(s => ({
      name: s.name, output: simulate(s.params, days)
    }));
    draw(lastResults, p.population, days);
    resultsBody.replaceChildren();
    for (const [i, entry] of lastResults.entries()) {
      const tr = document.createElement("tr");
      const cells = [entry.name, entry.output.peakInfected.toLocaleString(undefined, { maximumFractionDigits: 0 }),
        String(entry.output.peakDay), (100 * entry.output.attackRate).toFixed(2) + "%"];
      for (const [j, val] of cells.entries()) {
        const td = document.createElement("td");
        td.textContent = val;
        if (j === 0) td.style.setProperty("--series-color", plotColors[i]);
        if (j === 0) td.classList.add("scenario");
        tr.appendChild(td);
      }
      resultsBody.appendChild(tr);
    }
    const warning = p.model === "sir" && p.vaccinationCoverage > 0 ?
      " Note: vaccination scenarios use SEIRV; their latency differs from the SIR baseline." : "";
    status.textContent = `Completed ${lastResults.length} scenario(s) locally. No data was uploaded.${warning}`;
    status.className = "status success";
    document.getElementById("export").disabled = false;
  } catch (error) {
    lastResults = [];
    document.getElementById("export").disabled = true;
    document.getElementById("chart").replaceChildren();
    resultsBody.replaceChildren();
    status.textContent = error.message;
    status.className = "status error";
  }
}

function exportCsv() {
  if (lastResults.length === 0) return;
  const rows = ["scenario,day,S,E,I,R,V"];
  for (const { name, output } of lastResults) {
    for (const s of output.series) {
      rows.push([name, s.day, s.S, s.E, s.I, s.R, s.V].map(v =>
        `"${String(v).replaceAll('"', '""')}"`).join(","));
    }
  }
  const url = URL.createObjectURL(new Blob([rows.join("\n") + "\n"], { type: "text/csv;charset=utf-8" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "seir-scenarios.csv";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

if (typeof document !== "undefined") {
  document.getElementById("controls")?.addEventListener("submit", event => {
    event.preventDefault();
    render();
  });
  document.getElementById("export")?.addEventListener("click", exportCsv);
  if (document.getElementById("controls")) render();
}
