from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

try:
    import yaml
except ImportError:  # pragma: no cover - exercised in minimal server envs.
    yaml = None


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


class ConfigError(ValueError):
    pass


def load_yaml_config(path: Union[str, Path], env: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    if yaml is None:
        raise ConfigError("PyYAML is required for YAML configs. Install with `pip install pyyaml`.")
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigError("top-level YAML config must be a mapping")
    return expand_env(raw, env=env)


def expand_env(value: Any, env: Optional[Mapping[str, str]] = None) -> Any:
    env = env or os.environ
    if isinstance(value, dict):
        return {key: expand_env(item, env=env) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env(item, env=env) for item in value]
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda match: _resolve_env(match, env), value)
    return value


def require_path(path: Union[str, Path], *, kind: str, must_be_file: bool = False) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.exists():
        raise FileNotFoundError(f"{kind} does not exist: {candidate}")
    if must_be_file and not candidate.is_file():
        raise FileNotFoundError(f"{kind} must be a file: {candidate}")
    if not must_be_file and not candidate.is_dir():
        raise FileNotFoundError(f"{kind} must be a directory: {candidate}")
    return candidate


def _resolve_env(match: re.Match[str], env: Mapping[str, str]) -> str:
    name, default = match.group(1), match.group(2)
    value = env.get(name)
    if value is not None and value != "":
        return value
    if default is not None:
        return default
    raise ConfigError(f"required environment variable is not set: {name}")
