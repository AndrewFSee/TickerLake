"""Configuration and secrets loading.

Precedence, highest first:
  1. Environment variables (TICKERLAKE_* overrides, and API keys)
  2. config/config.yaml
  3. Built-in defaults

Secrets are *only* ever read from the environment (via .env), never from YAML,
so the config file stays safe to commit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Project root = two levels up from this file (src/tickerlake/config.py -> repo root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class ConfigError(RuntimeError):
    """Raised when configuration is missing or malformed."""


@dataclass(frozen=True)
class Secrets:
    """API credentials, sourced exclusively from the environment."""

    fred_api_key: str | None = None
    finnhub_api_key: str | None = None
    sec_user_agent: str | None = None

    @classmethod
    def from_env(cls) -> Secrets:
        return cls(
            fred_api_key=_clean(os.getenv("FRED_API_KEY")),
            finnhub_api_key=_clean(os.getenv("FINNHUB_API_KEY")),
            sec_user_agent=_clean(os.getenv("SEC_USER_AGENT")),
        )

    def has(self, name: str) -> bool:
        return bool(getattr(self, name, None))

    def __repr__(self) -> str:  # never leak key material into logs
        present = [f for f in ("fred_api_key", "finnhub_api_key", "sec_user_agent") if self.has(f)]
        return f"Secrets(present={present})"


def _clean(value: str | None) -> str | None:
    """Strip whitespace and surrounding quotes; treat empty/placeholder as absent."""
    if value is None:
        return None
    v = value.strip().strip('"').strip("'").strip()
    if not v or v.lower() in {"none", "null", "changeme", "your_key_here"}:
        return None
    return v


@dataclass
class Config:
    """Parsed configuration with dotted-path access."""

    raw: dict[str, Any]
    secrets: Secrets
    data_root: Path
    config_path: Path
    project_root: Path = field(default=PROJECT_ROOT)

    def get(self, dotted: str, default: Any = None) -> Any:
        """Look up a nested key, e.g. cfg.get('options.throttle_seconds', 0.5)."""
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise ConfigError(f"Missing required config key: {dotted}")
        return value

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name, {})
        if not isinstance(value, dict):
            raise ConfigError(
                f"Config section '{name}' must be a mapping, got {type(value).__name__}"
            )
        return value

    def stage_enabled(self, stage: str) -> bool:
        """A stage runs only if enabled in config AND its required secret is present."""
        if not self.get(f"{stage}.enabled", False):
            return False
        required = _REQUIRED_SECRET.get(stage)
        if required and not self.secrets.has(required):
            return False
        return True

    def missing_secret_for(self, stage: str) -> str | None:
        """Env var name that is blocking this stage, or None."""
        required = _REQUIRED_SECRET.get(stage)
        if required and not self.secrets.has(required):
            return _ENV_NAME[required]
        return None


# Stages whose fetcher cannot function without a credential.
# SEC EDGAR technically works without a User-Agent but the SEC throttles/blocks
# anonymous clients, so we treat it as required rather than fail mid-run.
_REQUIRED_SECRET = {
    "fred": "fred_api_key",
    "finnhub": "finnhub_api_key",
    "sec_edgar": "sec_user_agent",
}

_ENV_NAME = {
    "fred_api_key": "FRED_API_KEY",
    "finnhub_api_key": "FINNHUB_API_KEY",
    "sec_user_agent": "SEC_USER_AGENT",
}


def load_config(config_path: str | Path | None = None) -> Config:
    """Load YAML config + .env secrets and resolve the data root."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise ConfigError(
            f"Config file not found: {path}\n"
            "Copy config/config.yaml from the repo, or pass --config."
        )

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Config root must be a mapping, got {type(raw).__name__}")

    _apply_env_overrides(raw)

    root_value = os.getenv("TICKERLAKE_DATA_ROOT") or raw.get("storage", {}).get("root", "data")
    data_root = Path(root_value)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root

    return Config(
        raw=raw,
        secrets=Secrets.from_env(),
        data_root=data_root.resolve(),
        config_path=path,
    )


def _apply_env_overrides(raw: dict[str, Any]) -> None:
    """Allow TICKERLAKE_OPTIONS__THROTTLE_SECONDS=1.5 style overrides.

    Double underscore separates nesting levels. Values are parsed as YAML scalars
    so `true`, `null`, and numbers come through with the right type.
    """
    prefix = "TICKERLAKE_"
    skip = {"TICKERLAKE_DATA_ROOT"}
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix) or env_key in skip:
            continue
        parts = [p.lower() for p in env_key[len(prefix) :].split("__")]
        if len(parts) < 2:
            continue
        node = raw
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        try:
            node[parts[-1]] = yaml.safe_load(env_val)
        except yaml.YAMLError:
            node[parts[-1]] = env_val
