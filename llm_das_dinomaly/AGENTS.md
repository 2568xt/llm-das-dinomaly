# Package Instructions

## Module Boundaries

- Keep core reusable code in this package; scripts should stay thin entrypoints.
- Put Dinomaly interop in `integrations/` or `wrappers/` instead of importing third-party internals throughout the package.
- Keep orchestration in `pipelines/`, synthetic policy code in `synth/`, selection logic in `search/`, and feature/fusion model code in `enhancer/`.
- Prefer existing config and path helpers from `utils/` over ad hoc parsing.

## Compatibility

- Keep Dinomaly-facing code compatible with the older visual stack described in the root `AGENTS.md`.
- Keep future LLM/code-generation behavior decoupled from Dinomaly runtime code; exchange generated policies through files or explicit records.

## Tests

- Add or update focused tests under `tests/` when package behavior changes.
- For changes touching shared config, caching, records, wrappers, or pipelines, run the relevant focused pytest file before the full required checks.
