"""Config reader (IMP-001). PyYAML and the standard library only.

Every parameter in config.yml is a leaf {value, prov, ref}. load_config()
returns the nested structure with each leaf replaced by its `value`.
"""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yml"

PROVENANCE_TAGS = frozenset(
    {"literature", "locked", "placeholder", "computed", "fitted", "unset"}
)
# Tags whose value may legitimately be null until code or a decision fills it in.
NULLABLE_TAGS = frozenset({"unset", "computed"})
# Blocks passed through untouched (plain values, not leaves).
PASSTHROUGH_KEYS = frozenset({"meta", "open_questions_for_implementation"})


class ConfigError(ValueError):
    """Raised for a malformed config leaf."""


def _leaf_value(leaf, where):
    if "value" not in leaf:
        raise ConfigError(f"{where}: leaf has 'prov' but no 'value'")
    if "prov" not in leaf:
        raise ConfigError(f"{where}: leaf has 'value' but no 'prov'")
    if "ref" not in leaf:
        raise ConfigError(f"{where}: leaf has no 'ref'")
    prov = leaf["prov"]
    if prov not in PROVENANCE_TAGS:
        raise ConfigError(f"{where}: unknown provenance tag {prov!r}")
    if not isinstance(leaf["ref"], str) or not leaf["ref"]:
        raise ConfigError(f"{where}: 'ref' must be a non-empty string")
    value = leaf["value"]
    if value is None and prov not in NULLABLE_TAGS:
        raise ConfigError(f"{where}: null value not allowed for prov {prov!r}")
    return value


def _walk(node, where):
    if isinstance(node, dict):
        if "value" in node or "prov" in node:
            return _leaf_value(node, where)
        return {k: _walk(v, f"{where}.{k}") for k, v in node.items()}
    return node


def load_config(path=None):
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    out = {}
    for key, node in raw.items():
        out[key] = node if key in PASSTHROUGH_KEYS else _walk(node, key)
    return out
