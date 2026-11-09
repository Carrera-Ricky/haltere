# Haltere Project Handoff

## Project

Local repository:

`D:\Dev\FlightSim`

Origin repository:

`https://github.com/skulitom/haltere`

Haltere is a research project using a connectome-constrained recurrent neural network based on a selected 30,000-neuron fruit-fly connectome to control a simulated FPV quadcopter.

The system is not a complete biological fly-brain simulation. It is a rate-based neural controller whose structure is constrained by the fruit-fly connectome. The controller receives telemetry-derived sensory inputs and produces motor-control outputs.

## User context

The user is learning the repository from a beginner graphics background. Explain changes conceptually before making them. Avoid assuming familiarity with PyTorch, drone simulation, neural networks, or Liftoff.

The user does not currently have drone hardware or Liftoff installed. Work should therefore remain in the headless Haltere simulator unless the user later installs Liftoff.

## Environment setup

The machine uses Windows PowerShell and AMD hardware. CUDA is not appropriate.

Python 3.13 was installed and managed through `uv`. The project environment is:

`D:\Dev\FlightSim\.venv`

The CPU PyTorch package was installed with:

```powershell
uv pip install --no-cache --python .venv/Scripts/python.exe `
  torch==2.11.0 `
  --index-url https://download.pytorch.org/whl/cpu
```

Project dependencies were installed with:

```powershell
uv pip install --no-cache --python .venv/Scripts/python.exe -e ".[dev,vision]"
```

uv encountered Windows access-denied problems in its normal cache. The working workaround was `--no-cache`; a project-local cache can also be selected with:

```powershell
$env:UV_CACHE_DIR = "D:\Dev\FlightSim\.uv-cache"
$env:UV_LINK_MODE = "copy"
```

The user successfully ran the original CPU smoke test with `artifacts\ftSmooth_best.pt`.

## Original smoke test

From `D:\Dev\FlightSim`:

```powershell
.venv\Scripts\python.exe -m haltere.cli eval `
  artifacts\ftSmooth_best.pt `
  --device cpu `
  --batch 1 `
  --steps 30
```

This loads the checkpoint and runs a short headless simulator check. The sparse-tensor warning is a PyTorch warning, not a failure.

## Current goal

The first custom behavior is a two-post figure-eight maneuver:

1. Navigate to Post A.
2. Orbit Post A.
3. Transition to Post B.
4. Orbit Post B in the opposite direction.
5. Complete the route without contacting either post.

The first version intentionally uses known post coordinates. Vision and image recognition are deferred.

## Code changes made

### Route generation

Added:

`haltere/liftoff/figure_eight.py`

This generates two post centers, circular orbit waypoints, and transitions between the two posts. It produces a route dictionary containing post locations and waypoints.

### Route CLI command

Updated:

- `haltere/cli.py`
- `haltere/liftoff/commands.py`

New command:

```powershell
.venv\Scripts\python.exe -m haltere.cli liftoff figure-eight `
  --out configs\two_post_figure8.yaml
```

This command only generates the YAML route. It does not launch Liftoff or fly the drone.

The generated route currently exists at:

`D:\Dev\FlightSim\configs\two_post_figure8.yaml`

### Headless figure-eight task

Added:

`haltere/sim/figure_eight.py`

This loads the route YAML and creates a headless task around the existing quadcopter simulator. It reuses the existing neural brain and vehicle dynamics. It does not use images.

The task advances a moving target along the known route and records the drone's clearance from the two post centers.

### Evaluation CLI command

Updated:

`haltere/cli.py`

New command:

```powershell
.venv\Scripts\python.exe -m haltere.cli eval-figure-eight `
  artifacts\ftSmooth_best.pt `
  --route configs\two_post_figure8.yaml `
  --device cpu `
  --steps 3600
```

This runs approximately 36 seconds of headless simulation.

### Tests

Added:

`tests/test_figure_eight.py`

The tests validate basic route geometry and reject overlapping orbit settings.

## Metric meanings

The current figure-eight evaluator reports:

- `dist_mean`: average distance from the moving target over the run.
- `within_0.5m`: fraction of simulation time spent within 0.5 meters of the moving target.
- `speed`: average drone speed over the run.
- `final_dist`: distance from the target at the end.
- `final_speed`: drone speed at the end.
- `target_speed`: target movement speed along the route.
- `route_progress_m`: final target progress in meters.
- `route_length_m`: total route length in meters.
- `min_post_clearance_m`: closest horizontal distance to either post center.
- `hit_post`: whether the abstract collision radius was breached.
- `cost`: combined simulator penalty; lower is generally better, but it is not a percentage or universal pass/fail score.
- `crashed_ever`: whether the simulated quadcopter crashed at any point.

The current abstract post collision radius is 1 meter. The posts are not yet rendered T-post meshes.

## Latest result

Command used:

```powershell
.venv\Scripts\python.exe -m haltere.cli eval-figure-eight `
  artifacts\ftSmooth_best.pt `
  --route configs\two_post_figure8.yaml `
  --device cpu `
  --steps 3600
```

Result:

```json
{
  "dist_mean": 1.7710717916488647,
  "within_0.5m": 0.13944444060325623,
  "crashed": 0.0,
  "speed": 1.1583707332611084,
  "target_speed": 1.5,
  "final_dist": 0.48843029141426086,
  "final_speed": 0.38237249851226807,
  "route_progress_m": 43.0428466796875,
  "route_length_m": 43.0428466796875,
  "min_post_clearance_m": 1.6441214084625244,
  "hit_post": 0.0,
  "cost": 3.914942026305198,
  "crashed_ever": 0.0
}
```

Interpretation:

- The route completed: 43.04 m of 43.04 m.
- No crash occurred.
- No post contact occurred.
- Minimum post clearance was 1.64 m, above the 1 m abstract collision radius.
- Average speed was 1.16 m/s against a 1.5 m/s target.
- Average route tracking error was 1.77 m.
- The drone was within 0.5 m of the target for about 14% of the run.
- The final distance was about 0.49 m because the target stopped at the end and the drone settled near it.

This is a successful safety/completion proof-of-concept, but not yet a tight or fast route-following controller.

## Important architecture

```text
route definition
    -> target position
    -> existing brain receives goal information
    -> brain produces motor commands
    -> quadcopter simulator updates position
    -> metrics measure tracking and safety
```

The route layer decides where the drone should go. The brain controls movement. The simulator models physics. The metrics evaluate the result.

## Suggested next steps

1. Review the generated YAML and the new route/task code with the user.
2. Improve the headless route controller or target presentation if tighter tracking is desired.
3. Add a more realistic T-post collision envelope: vertical pole plus horizontal crossbar.
4. Add a rendered simulator only after the headless behavior is understood.
5. If Liftoff is installed later, generate/install a course with two visible T-post-like objects.
6. Defer camera recognition until the known-coordinate route is reliable.

Do not claim visual recognition, arbitrary-course navigation, or real-drone control from the current result.
