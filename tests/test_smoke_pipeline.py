"""End-to-end smoke run (data -> train -> eval -> ledger -> tables -> figures) on tiny settings (CPU, a few minutes).

The numbers are meaningless at this size; the test checks that every stage runs and every output file is written.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_reproduce_smoke(tmp_path):
    env = dict(os.environ, PYTHON=sys.executable)
    proc = subprocess.run(["bash", str(ROOT / "scripts" / "reproduce.sh"), "--smoke", "--workdir", str(tmp_path)],
                          cwd=ROOT, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "FAILED" not in proc.stdout
    for name in ("ledger/classical_per_seed.csv", "ledger/tfim_per_seed.csv", "figures/cp_ood_by_mass_ratio.csv",
                 "figures/sc_ood_by_k.csv", "figures/tfim_by_g.csv", "extra/poisson_bracket_per_seed.csv"):
        assert (tmp_path / "results" / name).stat().st_size > 0, name
    for tid in ("T1", "T6", "T8", "T11", "T12", "T16", "T18"):
        assert (tmp_path / "out" / "tables" / f"{tid}.tex").exists(), tid
    for fig in (2, 3, 4, 5, 6, 8):
        assert (tmp_path / "out" / "figures" / f"fig{fig}.pdf").exists(), fig
