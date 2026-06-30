# experiment_runner

Manage [Stable-Baselines3](https://github.com/maymac00/mo-stable-baselines3) experiments with
[Optuna](https://optuna.org/) hyperparameter tuning. You subclass one abstract manager, declare a
search space in YAML, and the library handles vectorised env construction (single- and
multi-objective), monitoring, optional reward normalisation, TensorBoard logging, per-trial result
directories, and an Optuna study backed by SQLite (or any Optuna storage URL).

Depends on the **`mo-stable-baselines3`** fork (pinned by commit in `pyproject.toml`), which adds the
multi-objective vec-env stack (`MoVecEnv`, `MoDummyVecEnv`, `MoVecMonitor`, `MoVecNormalize`,
`MoMonitor`).

## Install

The SB3 fork is a git dependency, so install from the project root (it is resolved automatically):

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
```

or with pip:

```bash
pip install -e .
```

Requires Python ≥ 3.10, torch ≥ 2.6.0.

## Core API

Subclass `ExperimentManager` and implement three abstract methods; optionally override
`get_callbacks`.

```python
from typing import Dict, Any
import gymnasium as gym
from stable_baselines3 import PPO
from experiment_runner import ExperimentManager


class CartpoleHPT(ExperimentManager):
    def build_model(self, env, model_args: Dict[str, Any]):
        model_args["batch_size"] = model_args["n_steps"] * 5
        return PPO("MlpPolicy", env, device="cpu", verbose=1, **model_args)

    def build_env(self, args: Dict[str, Any]):
        return gym.make("CartPole-v1")

    def evaluate(self, model, env, exp_args: Dict[str, Any]) -> float:
        obs, _ = env.reset()
        rew = 0.0
        for _ in range(exp_args["n_eval_episodes"]):
            for _ in range(500):
                action, _ = model.predict(obs)
                obs, reward, tr, tm, _ = env.step(action)
                rew += reward
                if tr or tm:
                    obs, _ = env.reset()
                    break
        return rew / exp_args["n_eval_episodes"]


if __name__ == "__main__":
    em = CartpoleHPT("test", "cartpole_example", hp_path="hp_search.yaml",
                     tb_log=True, normalize_reward=True)
    em.optimize(10)
```

### `ExperimentManager(...)`

| arg | meaning |
|-----|---------|
| `name` | study / experiment name |
| `save_dir` | directory for per-trial outputs and the SQLite DB |
| `storage` | Optuna storage URL; defaults to `sqlite:///{cwd}/{save_dir}/{name}.db` |
| `hp_path` | YAML search-space file |
| `prune` | reserved for Optuna pruning (see TODO — currently a no-op) |
| `save_models` | save `final_model` (and `vec_normalize.pkl`) per trial |
| `tb_log` | enable TensorBoard logging into the trial dir |
| `normalize_reward` | wrap env in `VecNormalize`/`MoVecNormalize` (reward only) |
| `n_objectives` | >1 switches to the `Mo*` vec-env stack |
| `reload` | re-read the YAML on every trial (live-edit the search space) |

Abstract methods: `build_model(env, model_args)`, `build_env(args)`, `evaluate(model, env, exp_args) -> float`.
Optional: `get_callbacks(args) -> list`. Drive the loop with `optimize(n_trials)`; read results via
`get_best_params()` / `get_best_value()`. The study direction is `maximize`.

### Search-space YAML

Four sections — `model`, `policy`, `env`, `experiment` — passed to your callbacks as nested dicts.
A value is either **fixed** (used verbatim) or a **distribution** (a dict with a `type`):

```yaml
model:
  learning_rate: {type: float, low: 1e-5, high: 1e-3}   # see TODO re: log-uniform
  n_steps:       {type: int,   low: 2000, high: 6000, step: 2000}
  gamma:         {type: float, low: 0.7,  high: 0.9999}
  ent_coef:      {type: float, low: 0.001, high: 0.2}

policy:
  net_arch: {pi: [32, 32], vf: [32, 32]}                # fixed value

experiment:
  n_timesteps: 1000000     # required
  n_eval_episodes: 20
  n_envs: 5                # >1 -> SubprocVecEnv / MoVecEnv
  log_interval: 5
```

Supported `type`s: `float`, `int`, `categorical` (needs `choices`). `experiment.n_timesteps` is
mandatory. `n_envs > 1` selects subprocess (or multi-objective) vector envs.

## Storage & inspection

By default an SQLite study DB is written under `save_dir`. Inspect it live with:

```bash
optuna-dashboard sqlite:///cartpole_example/test.db
```

---

## TODO / Roadmap

Grounded in the current source. Sorted by difficulty.

### Easy

- [ ] **Honour `log` sampling.** The YAML advertises `log: true` for `learning_rate`, but
  `HyperparameterManager.define_search_space` never forwards it — LR is sampled uniformly, not
  log-uniform. Parse `log` and pass it to `suggest_float`/`suggest_int`.
- [ ] **Keep int spaces integral.** `suggest_int` is called with `float(low/high)` and a float `step`;
  pass them as `int` so the distribution is correct.
- [ ] **Make trial cwd switching safe.** `objective` does `os.chdir(trial_path)` and only restores cwd
  on the success path. Wrap it in `try/finally` so a raised trial doesn't leave later trials writing
  into the wrong directory.
- [ ] **Document the schema in-repo** (this README covers it) and validate unknown keys/sections early.
- [ ] **Smoke test in CI:** run the CartPole example for one trial / a few thousand steps.

### Medium

- [ ] **Finish pruning.** `prune` and the `TrialPruned` handler in `optimize` are wired, but `objective`
  never calls `trial.report(...)` or raises (the pruning callback is a `# TODO` at
  `ExperimentManager.py:122`). Add an SB3 eval callback that reports intermediate scores and prunes.
- [ ] **Parallelise trials.** Replace the manual `ask`/`tell` Python loop with `study.optimize(..., n_jobs=k)`
  or multiple worker processes sharing one storage, so HPO uses more than one core.
- [ ] **Distributed-storage recipe.** Document/support Postgres or MySQL; SQLite locks under concurrent
  workers. The `storage` arg already accepts any URL.
- [ ] **Robust user-attrs.** The attr-logging loop silently swallows failures; JSON-serialise non-scalar
  values (e.g. `net_arch`) so they survive into the study.
- [ ] **Checkpoint/resume a single long trial** (SB3 `save`/`load` + `VecNormalize` stats) so a preempted
  run continues instead of restarting from zero.

### Hard

- [ ] **True multi-objective studies.** The `Mo*` env stack is in place, but the study is scalar
  (`direction="maximize"`). Support `directions=[...]`, let `evaluate` return a vector, and expose the
  Pareto front / hypervolume.
- [ ] **SLURM dispatch (submitit).** Launch N workers against shared Optuna storage with requeue on
  preemption — turns this into a cluster HPO harness.
- [ ] **Trial deduplication.** Skip trials whose resolved hyperparameters already completed, queried from
  the study, to make sweeps idempotent and resumable.
- [ ] **Pluggable sampler/pruner from YAML** (TPE / CMA-ES / ASHA-Hyperband) with budget-aware scheduling.
- [ ] **Live metrics sink.** Beyond TensorBoard, stream eval/objective curves to InfluxDB for a Grafana
  dashboard aggregating concurrent workers.
- [ ] **Allow ordering of wrappers.** Right now this repo enforces the Monitor wrappers to be exactly one level above the final MoVec wrapper. Being able to move the Monitor wrapper to custom positions of the stack could allow tracking of normalised rewards, scalarise/unscalarised rewards. For instance, tracking MO rewards for PPO runs.

## License

GPL-3.0.
