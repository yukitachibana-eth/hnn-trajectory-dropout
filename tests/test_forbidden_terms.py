"""The published code must not mention internal experiment labels, removed features or
machine-specific details. Scans hnnuq/, configs/ and scripts/ (case-insensitive, word-ish)."""

import getpass
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("hnnuq", "configs", "scripts")
SCAN_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".sh", ".toml", ".txt"}

# Removed model families / experiments and internal wave labels.
FORBIDDEN = {
    "tdse": r"tdse",
    "film": r"film",
    "weld": r"weld",
    "mc_fixed": r"mc_fixed",
    "hvar": r"hvar",
    "h_var": r"h_var",
    "shadow": r"shadow",
    "measurement_renormalized": r"measurement_renormalized",
    "concrete_finite": r"concrete_finite",
    "L6": r"L6",
    "q2": r"q2",
    "q8": r"q8",
    "gnn": r"gnn",
    "baseline_models": r"baseline_models",
    "EstimateK": r"estimatek",
    "k_matrix estimation": r"k[-_ ]matrix[-_ ]estimation",
    "x10": r"x10[a-z]?",
    "x13": r"x13",
    "x9": r"x9",
    "v05": r"v0\.?5",
    "klfix": r"kl[-_]?fix(?:ed)?",
    "alpha_rom": r"alpha_rom",
    "persubstep_lastmask": r"persubstep_lastmask",
    "lastmask_persubstep": r"lastmask_persubstep",
    "wave": r"waves?",
    "wave/experiment id": r"x-\d+",
    # machine / cluster specifics
    "absolute home path": r"/homes?/",
    "user name": re.escape(getpass.getuser()),  # the account running the test
    "conda env": r"conda|exp-env",
    "tmux": r"tmux",
}

# (path relative to the repo root, term name) -> reason. Keep this list short.
ALLOWLIST = {}


def _files():
    for d in SCAN_DIRS:
        for path in sorted((ROOT / d).rglob("*")):
            if path.is_file() and path.suffix in SCAN_SUFFIXES and "__pycache__" not in path.parts:
                yield path


def _hits():
    compiled = {
        name: re.compile(rf"(?<![A-Za-z0-9]){pattern}(?![a-z0-9])", re.IGNORECASE)
        for name, pattern in FORBIDDEN.items()
    }
    # 'x9' style labels are matched case-insensitively but only as separate tokens
    for path in _files():
        rel = str(path.relative_to(ROOT))
        for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            for name, rx in compiled.items():
                if rx.search(line) and (rel, name) not in ALLOWLIST:
                    yield f"{rel}:{lineno}: [{name}] {line.strip()[:100]}"


def test_no_forbidden_terms():
    hits = list(_hits())
    assert not hits, "forbidden terms found:\n" + "\n".join(hits[:50])


@pytest.mark.parametrize("key", sorted(ALLOWLIST))
def test_allowlist_entries_have_reasons(key):
    assert ALLOWLIST[key].strip()
