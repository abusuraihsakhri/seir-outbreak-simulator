# SEIR Outbreak Simulator

Deterministic SIR, SEIR and SEIRV epidemic simulations with contact-reduction and pre-outbreak vaccination scenarios. The repository provides an interactive browser application and a Python/SciPy numerical implementation.

## Browser application

Serve the static files locally:

```bash
python -m http.server 8000 --directory web
```

Open **http://localhost:8000/**. Change the parameters, select **Run simulation**, inspect the infectious-prevalence curves and summary table, or export daily compartment counts as CSV. The browser version uses a built-in fourth-order Runge–Kutta solver in JavaScript; it does not load Python, Pyodide, or a backend. A GitHub Pages deployment workflow is included; its public URL should only be linked after GitHub Pages is enabled and the deployed application is verified.

The browser limits the parameter ranges to keep interactive calculations responsive. Computations occur locally; simulation inputs are not transmitted to a server. No external libraries or fonts are needed.

## Python simulation

Python 3.10+ is supported. Install dependencies and run the model:

```bash
python -m pip install -r requirements.txt
python seir_simulator.py --model seir --population 100000 --r0 2.5 --days 180 \
  --intervention-day 30 --intervention-reduction 0.5 \
  --vaccination-coverage 0.4 --vaccination-efficacy 0.9
```

To suppress the PNG plot, use `--no-plot`. Import `ModelParams`, `simulate`, `build_scenarios`, and `run_scenarios` from `seir_simulator.py` for programmatic use. Python integration uses SciPy `solve_ivp` with separate integration segments at an intervention discontinuity.

Both implementations use a closed, homogeneous-mixing population. The SEIR models use an exposed but non-infectious compartment, permanent recovery, and no births or deaths. Vaccination assumes a protected fraction equal to coverage × efficacy at time zero. **Cumulative infected** means `(E + I + R) / population` at the final simulated day, including infections that have not yet recovered. The interactive browser plots infectious prevalence; it does not predict observed case reports.

A vaccination scenario uses SEIRV. When the base model is SIR, comparison with vaccination therefore also changes the compartment structure, so that difference must not be interpreted as a pure vaccine effect.

## Additional diagnostic API and CLI

The pre-existing generic worker/audit components remain available for compatibility; **they are separate from the SEIR mathematical model** and do not provide calibrated epidemiological or clinical assessments.

```bash
python cli.py audit --primary 12 --secondary 4
python cli.py batch -i records.csv -o results.csv
python cli.py serve --host 127.0.0.1 --port 8000
```

The FastAPI endpoints include `/health`, `/metrics`, `/api/audit`, `/api/chat`, and `/api/audit/logs`. The batch CSV accepts `task_id`, `target_identifier`, `primary_metric`, `secondary_metric`, `status_descriptor`, and `is_critical_flag` (true/false or 1/0). The API and generic diagnostic components are not intended to process patient data. They have **no user authentication** and should not be exposed publicly without deployment hardening.

The audit trail is held in process memory and signed with HMAC-SHA256. Set `AUDIT_SECRET_KEY` securely when a stable key is required; otherwise a random process-local key is generated at startup. The pattern-based identifier check is not a substitute for systematic de-identification or HIPAA compliance.

## Tests

```bash
python -m pip install -r requirements.txt pytest
python -m pytest -q
node --test web/app.test.mjs
```

The web tests use Node.js 22+ with no npm dependencies. CI checks Python 3.10–3.12 and the JavaScript numerical solver. Python and browser solvers use different numerical methods, so small differences are expected.

## Limitations and privacy

This is an educational/research simulator, not a disease-specific forecast or a validated public-health decision system. There is no age structure, stochastic transmission, imported infections, waning immunity, reinfection, calibration, or uncertainty estimation. The static browser application processes parameters and produces exports locally. The optional Python API accepts and processes requests on the machine where it is deployed; avoid entering sensitive information.

## Deployment and license

The `.github/workflows/pages.yml` workflow deploys the `web/` directory through GitHub Pages once Pages is configured to use GitHub Actions. `Dockerfile` and `docker-compose.yml` support local deployment of the separate FastAPI diagnostic service; the compose configuration binds to localhost by default.

Released under the [MIT License](LICENSE).
