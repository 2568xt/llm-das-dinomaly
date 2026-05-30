# Agent Instructions

## Git And PR Workflow

- Never push code changes directly to `main`.
- For every code or documentation change, start from the latest `main` and create a feature branch named `codex/<short-task-name>`.
- Commit only intentional project files. Do not commit server-local paths, checkpoints, datasets, caches, or outputs.
- Push the feature branch and open a pull request into `main` with `gh pr create`.
- Each PR description must include:
  - Problem or motivation.
  - Summary of changes.
  - Test commands and results.
  - Server update or run commands, when relevant.
  - Risks or follow-up notes.

## Required Local Checks

Before opening a PR, run:

```bash
python3 -m pytest
python3 -m compileall -q llm_das_dinomaly scripts tests
```

For server-runner changes, also run the relevant dry-run or path-validation
command locally when possible.

## Server Notes

- The GPU server should pull from `main` only after the PR is merged.
- Keep `configs/server_paths.env` local to the server. It is intentionally gitignored.
- Keep checkpoints under ignored local paths such as `checkpoints/`.
- Use `configs/server_paths.example.env` only as a template.

## Environment Notes

- Dinomaly execution uses the older visual stack: Python 3.8, PyTorch 1.12/CUDA 11.3 where possible, and `third_party/Dinomaly/requirements.txt`.
- LLM code generation, when added later, should live in a separate newer Python environment and exchange generated policies through files.
