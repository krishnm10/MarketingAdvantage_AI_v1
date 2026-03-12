import os
from typing import Iterable, Optional


_TRUE_VALUES = {"1", "true", "t", "yes", "y", "on", "enabled"}
_FALSE_VALUES = {"0", "false", "f", "no", "n", "off", "disabled"}


def parse_bool(value: Optional[str], default: bool = False) -> bool:
    """
    Parse flexible boolean strings from env values.

    Supported true values:  true/1/yes/on/enabled
    Supported false values: false/0/no/off/disabled
    """
    if value is None:
        return default

    normalized = str(value).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return default


def get_env_bool(
    key: str,
    default: bool = False,
    aliases: Iterable[str] = (),
) -> bool:
    """
    Return boolean env flag from key or aliases.

    The first key found wins. If value is invalid/unset, default is returned.
    """
    for name in (key, *aliases):
        raw = os.getenv(name)
        if raw is not None:
            return parse_bool(raw, default=default)
    return default

