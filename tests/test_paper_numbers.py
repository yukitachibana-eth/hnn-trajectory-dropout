"""Compare every cell of the paper's data tables (tests/paper_tex/*.tex) with the tables built from results/ (out/tables/T<n>.csv).

Each paper cell is parsed into its numeric tokens (mean, sd, CI bounds, p-values, n, ...), a bold flag and a significance mark.  The tokens
must equal the built values rounded to the number of decimals (or mantissa digits for scientific notation) printed in the paper.
Cells are matched by position: (tabular index, data-row index, data-column index).
Run `python tests/test_paper_numbers.py` for a per-table report, or `pytest tests/test_paper_numbers.py`.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TEX = Path(__file__).resolve().parent / "paper_tex"
OUT = ROOT / "out/tables"
FILES = {"T1": "main_results_table", "T6": "id_nll_ic_ensemble_effect_table", "T7": "ic_noise_conventions_table", "T8": "tfim_noise_conventions_table",
         "T9": "main_results_full_table", "T10": "ood_pendulum_table", "T11": "tfim_results_table", "T12": "poisson_bracket_table",
         "T13": "mask_bank_size_table", "T14": "efficiency_table", "T15": "hidden_param_detection_table", "T16": "decomposition_table",
         "T17": "ic_noise_sweep_table", "T18": "energy_norm_table", "T19": "long_horizon_energy_table", "T20": "effect_sizes_classical_table",
         "T21": "effect_sizes_tfim_table"}
N_LABEL = {"T15": [2, 1], "T17": [3]}  # leading label columns per tabular (default 1)
TOKEN = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:e[-+]?\d+)?")


def tabulars(text: str) -> List[List[List[str]]]:
    """Per tabular, the data rows (after the header) as lists of raw cell strings."""
    out = []
    for body in re.findall(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", text, re.S):
        body = body.split("\\toprule", 1)[1].split("\\bottomrule")[0]
        rows, buf, seen_mid = [], "", False
        for ln in body.split("\n"):
            s = ln.strip()
            if not s or s.startswith(("\\cmidrule", "\\noalign")):
                continue
            if s.startswith("\\midrule"):
                seen_mid = True
                continue
            buf += " " + s
            if s.endswith("\\\\"):
                row = buf.strip()[:-2].strip()
                buf = ""
                if seen_mid and not row.startswith(("\\multicolumn{", "\\emph")) or (seen_mid and "&" in row and not row.startswith("\\multicolumn{")):
                    rows.append([c.strip() for c in re.split(r"(?<!\\)&", row)])
        out.append(rows)
    return out


def parse_cell(raw: str) -> Tuple[List[str], bool, str, str]:
    """(numeric tokens, bold, mark, plain text) of one tex cell."""
    bold = "\\textbf{" in raw
    mark = "*" if "\\textsuperscript{*}" in raw else ("dagger" if "dagger" in raw else "")
    s = re.sub(r"\\multicolumn\{\d+\}\{\w+\}", "", raw).replace("$-$", "\x00").replace("\\textsuperscript{*}", "").replace("$^{\\dagger}$", "")
    s = re.sub(r"(\d)\\times10\^\{(-?\d+)\}", r"\1e\2", s)
    s = re.sub(r"\\(textbf|makecell|footnotesize|textit)", "", s).replace("\\\\", " ").replace("\\pm", " ").replace("--", " ")
    s = s.replace("$", " ").replace("{", " ").replace("}", " ").replace("\x00", "-")
    toks = TOKEN.findall(s)
    plain = re.sub(r"\s+", " ", re.sub(r"[\\${}]", " ", raw.replace("\\textbf", ""))).strip()
    return toks, bold, mark, plain


def matches(tok: str, v: float) -> bool:
    """True if v rounds to the printed token (same decimals / mantissa digits)."""
    t = tok.lower()
    if "e" in t:
        mant, exp = t.split("e")
        d = len(mant.split(".")[1]) if "." in mant else 0
        m2, e2 = f"{v:.{d}e}".split("e")
        return float(m2) == float(mant) and int(e2) == int(exp)
    d = len(t.split(".")[1]) if "." in t else 0
    return abs(float(f"{v:.{d}f}")) == abs(float(t)) and (float(t) == 0 or float(f"{v:.{d}f}") == float(t))


def compare(tid: str) -> Tuple[int, int, List[str]]:
    """(cells matched, cells total, mismatch descriptions) for one table."""
    csv = pd.read_csv(OUT / f"{tid}.csv", keep_default_na=False)
    mine: Dict[Tuple[int, int, int], pd.Series] = {(r.part, r.row_idx, r.col_idx): r for r in csv.itertuples()}
    tabs = tabulars((TEX / f"{FILES[tid]}.tex").read_text())
    nl = N_LABEL.get(tid, [1] * len(tabs))
    total = ok = 0
    bad: List[str] = []
    for pi, rows in enumerate(tabs):
        for ri, row in enumerate(rows):
            for ci, raw in enumerate(row[nl[pi]:]):
                total += 1
                toks, bold, mark, plain = parse_cell(raw)
                m = mine.get((pi, ri, ci))
                where = f"{tid} part{pi} row{ri} ({row[0][:30]}) col{ci}: paper {raw[:60]!r}"
                if m is None:
                    bad.append(where + " | no built cell")
                    continue
                nums = [float(x) for x in str(m.nums).split(";")] if str(m.nums) else []
                errs = []
                if toks:
                    if len(toks) != len(nums):
                        errs.append(f"token count {len(toks)} vs {len(nums)}")
                    else:
                        errs += [f"{t} != {v!r}" for t, v in zip(toks, nums) if not matches(t, v)]
                elif plain.replace(" ", "").lower() != str(m.display).replace(" ", "").lower():
                    errs.append(f"text {plain!r} vs {m.display!r}")
                if bool(m.bold) != bold:
                    errs.append(f"bold paper={bold} built={bool(m.bold)}")
                if str(m.mark) != mark:
                    errs.append(f"mark paper={mark!r} built={m.mark!r}")
                if errs:
                    bad.append(where + " | " + "; ".join(errs))
                else:
                    ok += 1
    n_built = sum(1 for k in mine if k[0] < len(tabs))
    if n_built != total:
        bad.append(f"{tid}: built {n_built} cells vs {total} in the paper")
    return ok, total, bad


@pytest.fixture(scope="session", autouse=True)
def built() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts/40_build_tables.py")], check=True, cwd=ROOT)


@pytest.mark.parametrize("tid", list(FILES))
def test_table_matches_paper(tid: str) -> None:
    ok, total, bad = compare(tid)
    assert not bad, f"{tid}: {ok}/{total} cells match\n" + "\n".join(bad)


if __name__ == "__main__":
    subprocess.run([sys.executable, str(ROOT / "scripts/40_build_tables.py")], check=True, cwd=ROOT)
    tot_ok = tot = 0
    for t in FILES:
        o, n, b = compare(t)
        tot_ok, tot = tot_ok + o, tot + n
        print(f"{t}: {o}/{n}")
        for x in b:
            print("   MISMATCH", x)
    print(f"TOTAL {tot_ok}/{tot}")
