### 🔄 Project Awareness & Context

-   **Always read `PLANNING.md`** at the start of a new conversation to understand the project's architecture, goals, style, and constraints.
-   **Check `TASK.md`** before starting a new task. If the task isn’t listed, add it with a brief description and today's date.
-   **Use consistent naming conventions, file structure, and architecture patterns** as described in `PLANNING.md`.
-   **Use venv_linux** (the virtual environment) whenever executing Python commands, including for unit tests.

### 🧱 Code Structure & Modularity

-   **File size rule**: Keep any single source file **≤ 500 lines**. If approaching the limit, split into cohesive modules.
-   **Feature-first structure**: Group by feature (navigation, restaurant, shopping, safety) rather than by layer when possible.

#### Backend (Python / FastAPI)

-   **Top-level layout**
    -   `app/main.py` – FastAPI app factory & startup wiring
    -   `app/api/` – **routers** per feature: `navigation.py`, `restaurant.py`, `shopping.py`, `safety.py`, `health.py`
    -   `app/models/` – **Pydantic schemas** and **SQLModel ORM** models (separate folders `schemas/`, `entities/`)
    -   `app/services/` – business logic (e.g., `asr.py`, `ocr.py`, `mt.py`, `tts.py`, `allergens.py`, `maps.py`)
    -   `app/core/` – cross-cutting concerns (`config.py` with Pydantic Settings, `logging.py`, `deps.py` DI wiring, `errors.py`)
    -   `app/adapters/` – external integrations (e.g., Google/Apple/Azure/Mapbox SDK clients)
    -   `app/ml/` – edge/cloud model utilities (`loader.py`, `quantization.py`, `runners/` for TFLite/ONNX/whisper.cpp)
    -   `app/utils/` – small pure helpers (no side effects)
    -   `migrations/` – SQLModel/SQLAlchemy migrations (Alembic)
    -   `tests/` – mirrors `app/` with unit/integration/latency tests
-   **Routers**: thin, async endpoints → delegate to `services/*`.
-   **Services**: stateless where possible; inject adapters via function args or FastAPI Depends.
-   **Models**: Pydantic for I/O, SQLModel for persistence. Keep validation in schemas, not in routers.
-   **Config**: Use Pydantic Settings. Load from env; `.env` only for local dev. No secrets in code. Support `APP_ENV` profiles.
-   **Imports**: prefer absolute imports within `app.*`. Avoid circular deps by layering: `api` → `services` → `adapters`/`ml`.
-   **I18n**: centralize locale tables and message templates under `app/i18n/`.
-   **Edge assets**: store model blobs under `app/ml/assets/` with manifest checksums; never commit large binaries—use LFS or release artifacts.

#### Mobile Frontend (React Native **or** Flutter)

-   **React Native** (if chosen)
    -   `src/` with **feature folders**: `features/navigation/`, `features/restaurant/`, `features/shopping/`, `features/safety/`
    -   Inside each: `components/`, `hooks/`, `screens/`, `services/` (API clients), `i18n/`
    -   Shared libs: `src/lib/` (analytics, storage, theming), `src/ui/` (atoms/molecules)
    -   State: Redux Toolkit or Zustand per feature; avoid global state unless necessary
    -   Native modules: `native-modules/` for camera/ASR/OCR bindings
-   **Flutter** (if chosen)
    -   `lib/` feature-first folders mirroring RN; use `bloc` or `riverpod` for state
    -   Platform channels for camera/ASR/OCR bindings; isolate platform code under `platform/`

#### Cross-cutting conventions

-   **Environment variables** via `python-dotenv` in dev; validate with Pydantic. Mobile uses `.env.*` files with build flavors.
-   **Secrets**: use platform keychains/keystores; never commit.
-   **Logging**: structured logs (JSON) with request correlation IDs. No PII in logs by default.
-   **Feature flags**: keep in `app/core/flags.py`; control rollouts and A/B.
-   **Telemetry**: privacy-preserving metrics only (latency, crashes). Opt-in for content logs.
-   **Imports & names**: snake_case for files, PascalCase for classes, lower_snake_case for functions/vars.
-   **Dependency boundaries**: UI ↔ API contracts defined by OpenAPI schemas; generate clients where possible.

### 🧪 Testing & Reliability

-   **Use Pytest for backend Python code.** Write tests for all new functions, classes, API routes, and edge AI integration logic.
-   **Add latency benchmarks** for AI tasks (OCR, ASR, MT, TTS) to ensure performance budgets are met.
-   **Include multilingual test cases** covering Latin, CJK, and Arabic scripts.
-   For mobile frontend (React Native/Flutter):
    -   **Write integration tests** (Jest/Detox for React Native, widget/integration tests for Flutter).
-   **Tests should live in `/tests` folder** mirroring the app structure, with coverage for:
    -   Expected use
    -   Edge cases (e.g., low-light OCR, noisy ASR, poor network, offline mode)
    -   Failure cases (e.g., timeouts, unsupported languages, model load errors)
-   **Automate tests in CI** with coverage reports and device/emulator runs for mobile.

### ✅ Task Completion

-   **Mark completed tasks in `TASK.md`** immediately after finishing them.
-   Add new sub-tasks or TODOs discovered during development to `TASK.md` under a “Discovered During Work” section.

### 📎 Style & Conventions

-   **Python is the backend primary language.**
-   **Follow PEP8**, use type hints, and enforce formatting with `black`.
-   **Use `pydantic` for data validation**, **`FastAPI` for backend APIs**, and **`SQLModel` for ORM**.
-   **Frontend/mobile code** should follow **React Native or Flutter coding standards** depending on chosen technology.
-   **Write Google-style docstrings** for all functions and classes.
-   **Ensure multilingual support:** code must be Unicode-safe and i18n/l10n ready.
-   **For AI/ML integration:** optimize for edge (efficient model loading, quantization), and document resource usage.

### 📚 Documentation & Explainability

-   **Update `README.md`** when new features are added, dependencies change, or setup steps are modified.
-   **Comment non-obvious code** and ensure everything is understandable to a mid-level developer.
-   When writing complex logic, **add an inline `# Reason:` comment** explaining the why, not just the what.

### 🧠 AI Behavior Rules

-   **Never assume missing context. Ask questions if uncertain.**
-   **Never hallucinate libraries or functions** – only use known, verified Python packages.
-   **Always confirm file paths and module names** exist before referencing them in code or tests.
-   **Never delete or overwrite existing code** unless explicitly instructed to or if part of a task from `TASK.md`.
