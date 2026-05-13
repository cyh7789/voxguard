# Scenario Map

Scenarios mirror the purple-agent package names under `src/`.

| Agent Package | Scenario Directory | Files |
|---------------|--------------------|-------|
| `src/purple_car_bench_agent/` | `scenarios/purple_car_bench_agent/` | `local.toml`, `smoke.toml`, `docker-local.toml`, `ghcr.toml` |
| `src/purple_car_bench_agent_codex/` | `scenarios/purple_car_bench_agent_codex/` | `smoke.toml`, `docker-local.toml` |
| `src/purple_car_bench_agent_codex_planner/` | `scenarios/purple_car_bench_agent_codex_planner/` | `smoke.toml`, `docker-local.toml` |
| `src/purple_car_bench_agent_codex_python/` | `scenarios/purple_car_bench_agent_codex_python/` | `smoke.toml`, `docker-local.toml` |

Use `smoke.toml` for quick local checks, `local.toml` for the fuller baseline
local run, `docker-local.toml` for local Docker builds, and `ghcr.toml` for the
published baseline image smoke.
