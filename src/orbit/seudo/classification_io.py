"""Save/load a time-course struct's classification state to disk, ported
from the `seudo` package's classification_io.py -- a Python-to-Python
pickle round trip of just the classification-relevant fields (tc +
transient_info).
"""

from __future__ import annotations

import pickle


def save_classification(tc_struct: dict, path) -> None:
    payload = {"tc": tc_struct["tc"], "transient_info": tc_struct.get("transient_info")}
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)


def load_classification(path) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def load_classification_into(se, path, which_struct: str = "default") -> dict:
    """Loads a saved classification into se's chosen tc struct, replacing
    its 'transient_info' (and 'tc', if that struct didn't already have one)."""
    payload = load_classification(path)
    tc_struct = se._resolve_tc_struct(which_struct)
    if tc_struct.get("tc") is None:
        tc_struct["tc"] = payload["tc"]
    tc_struct["transient_info"] = payload["transient_info"]
    return tc_struct
