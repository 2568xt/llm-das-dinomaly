# Script Instructions

## Scope

- Keep scripts as thin entrypoints that delegate reusable behavior to `llm_das_dinomaly/`.
- Avoid hardcoded machine-specific paths; use config files, environment variables, or documented CLI arguments.
- Quote shell variables and fail early for missing required paths or files.

## Server Runner Changes

- For server-runner changes, preserve local-only path handling through `configs/server_paths.env`.
- Do not print secrets or server-local absolute paths in committed examples.
- Run the relevant dry-run or path-validation command locally when possible, in addition to the root required checks.
