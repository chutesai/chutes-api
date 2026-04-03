# Agent Constraints

**Read this file before making any changes.** These constraints apply to all AI-assisted work in this repository. This file is the permanent **constraints** layer; task-specific **Goal / Constraints / Output Format / Failure Conditions** belong in copies of the templates under `[docs/specs/templates/](docs/specs/templates/)`.

## Project Identity

This repository is the **chutes.ai** platform **API and validator** code: HTTP API, related services, Docker/deployment assets, and integration with the broader Chutes ecosystem (see [README.md](README.md)).

## Stack (Non-Negotiable)

- **Language**: Python `>=3.10,<3.13` (3.12 is typical for local work)
- **Package manager / runs**: **[uv](https://github.com/astral-sh/uv)** — install with `uv sync`; run tools with `uv run …` (e.g. `uv run pytest`)
- **Build**: **hatchling** ([pyproject.toml](pyproject.toml)); installable packages: `**api`**, `**metasync**`
- **HTTP API**: **FastAPI** + **Uvicorn**, default **ORJSONResponse** ([api/main.py](api/main.py))
- **Data**: **SQLAlchemy 2.x** + **asyncpg**, **Pydantic** / **pydantic-settings**; SQL migrations under `[api/migrations/](api/migrations/)` (timestamped `.sql`)
- **Ops / deps** (non-exhaustive): **Redis**, **loguru**, **httpx** / **aiohttp**, **Socket.IO** client, **aioboto3**, domain packages (`chutes`, Bittensor-related libs, attestation tooling as pulled in by pyproject)

Do not replace this stack with alternate frameworks or ORMs unless explicitly agreed. Do not introduce extra dependencies without discussion.

## Hard Rules

- **Never add a new dependency** without explicit approval
- **Configuration**: use `**api.config.settings`** (pydantic-settings) and environment variables — **no hardcoded secrets**, connection strings, or API keys
- **Database schema changes**: add a new file under `[api/migrations/](api/migrations/)` **and** keep ORM models in `[api/database/orms.py](api/database/orms.py)` in sync; describe the migration plan in PRs/specs
- **Lint/format**: **Ruff** only — `make lint` and `make reformat` ([makefiles/lint.mk](makefiles/lint.mk)); there is **no enforced coverage percentage** in CI — still add or update tests when behavior changes
- `**nv-attest/`**: excluded from Ruff in pyproject — do not “fix” it via repo-wide lint refactors unless scoped to that subtree
- **Crypto / attestation-sensitive code**: follow existing patterns in the relevant modules (e.g. server/attestation paths); never hardcode keys or measurements

## Patterns

- **Async-first** in request paths: `**async def`** handlers, **async** SQLAlchemy sessions (`**get_session`** and related helpers in `[api/database/](api/database/)`); avoid blocking I/O in handlers
- **Domain layout**: routes in `**api/<domain>/router.py`**, shared logic often in `**api/<domain>/util.py**` (match neighboring domains)
- **Models and settings**: ORM in `**api/database/orms.py`**; app settings via `**api.config**`
- **Tests**: under `**tests/unit/`** and `**tests/integration/**`; use `**uv run pytest**`. **Match test style** to the file you edit (this repo uses both plain `**def test_*`** and `**class Test***` — stay consistent with surrounding tests)
- **Naming and structure**: follow existing modules in the same package; prefer small, focused changes over large unsolicited refactors

## Architecture Overview


| Area                                                 | Purpose                                                                                                       |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **[api/main.py](api/main.py)**                       | Main FastAPI app: lifespan, mounts domain routers (users, chutes, instances, invocations, payments, miner, …) |
| `**api/<domain>/`**                                  | Feature modules: `router.py`, `util.py`, and related schemas/helpers per domain                               |
| **[api/database/](api/database/)**                   | Async engine/session helpers; **[api/database/orms.py](api/database/orms.py)** ORM models                     |
| **[api/migrations/](api/migrations/)**               | Ordered SQL migrations consumed at startup (see `lifespan` / `tasks.py`)                                      |
| **[metasync/](metasync/)**                           | Metagraph / sync utilities (separate package in the same repo)                                                |
| `**api/payment/`**, `**api/socket_server.py**`, etc. | Auxiliary services or entrypoints alongside the main app — follow local patterns when touching them           |
| **[nv-attest/](nv-attest/)**                         | Attestation-related subtree (own tooling; Ruff-excluded at repo root)                                         |
| `**docker/`**, **[dev/dev.md](dev/dev.md)**          | Local Docker Compose and dev bootstrap                                                                        |


## Development Commands

```bash
uv sync --extra dev    # Install project + dev dependencies (from repo root)
uv run pytest          # Run tests (add paths or -k as needed)
make lint              # Ruff check + format check
make reformat          # Ruff format (line length per makefile)
make infrastructure    # Docker compose up for test infra (see makefiles/development.mk)
```

Local full stack: see **[dev/dev.md](dev/dev.md)** (Docker network, `docker compose`, optional GPU compose files).

**Session handshake**: Before starting substantive work, confirm you have read this file and will follow it; for non-trivial features, bugfixes, or refactors, consider filling a copy of the appropriate template under `[docs/specs/templates/](docs/specs/templates/)`.