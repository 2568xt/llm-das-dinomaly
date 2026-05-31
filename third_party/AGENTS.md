# Third-Party Instructions

## Vendor Boundary

- This directory contains third-party code and nested vendor repositories.
- Prefer changes in `llm_das_dinomaly/integrations/` or `llm_das_dinomaly/wrappers/` instead of editing vendor files.
- If a vendor file must change, keep the patch minimal and document the reason in the PR.

## Artifacts

- Do not add checkpoints, datasets, experiment outputs, local IDE files, or caches from this directory.
- Preserve third-party dependency stacks unless the user explicitly asks for an environment migration.
