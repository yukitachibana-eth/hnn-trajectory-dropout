#!/usr/bin/env bash
# Reproduce the paper's tables and figures.
#
#   scripts/reproduce.sh --from-csv                  tables + figures from the shipped results/ (seconds)
#   scripts/reproduce.sh --smoke [--workdir DIR]     whole pipeline on tiny settings (CPU), results in DIR/results
#   scripts/reproduce.sh [stage ...] [--gpus 0,1]    full pipeline; stages: data train eval ledger tables figures (default: all)
#
# Options: --gpus IDS   one job per listed GPU (default: CPU)
#          --jobs N     parallel CPU processes (default 1; 4 with --smoke)
#          --seeds S    e.g. 42-51 (default; one seed with --smoke)
#          --systems L  comma-separated subset of cp,sc,tfim
# A full run writes data/, runs/, results/ (overwriting the shipped CSVs) and out/; see README.md for the compute estimate.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"

stages=()
smoke=0
from_csv=0
gpus=""
jobs=""
seeds=""
systems=""
workdir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) smoke=1 ;;
    --from-csv) from_csv=1 ;;
    --gpus) gpus="$2"; shift ;;
    --jobs) jobs="$2"; shift ;;
    --seeds) seeds="$2"; shift ;;
    --systems) systems="$2"; shift ;;
    --workdir) workdir="$2"; shift ;;
    data|train|eval|ledger|tables|figures) stages+=("$1") ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

DATA="$ROOT/data"; RUNS="$ROOT/runs"; RES="$ROOT/results"; OUT="$ROOT/out"
if [[ $from_csv -eq 1 ]]; then
  stages=(tables figures)
elif [[ ${#stages[@]} -eq 0 ]]; then
  stages=(data train eval ledger tables figures)
fi
common=()
lenient=()
if [[ $smoke -eq 1 ]]; then
  [[ -n "$workdir" ]] || workdir="$(mktemp -d "${TMPDIR:-/tmp}/hnnuq_smoke.XXXXXX")"
  mkdir -p "$workdir"
  DATA="$workdir/data"; RUNS="$workdir/runs"; RES="$workdir/results"; OUT="$workdir/out"
  common+=(--smoke)
  lenient=(--lenient)  # smoke-sized results do not fill every table cell
  jobs="${jobs:-4}"
  echo "smoke workdir: $workdir"
fi
[[ -n "$systems" ]] && sysopt=(--systems "$systems") || sysopt=()
[[ -n "$seeds" ]] && seedopt=(--seeds "$seeds") || seedopt=()
[[ -n "$gpus" ]] && gpuopt=(--gpus "$gpus") || gpuopt=()
[[ -n "$jobs" ]] && jobopt=(--jobs "$jobs") || jobopt=()

run_stage() {
  local name="$1"; shift
  local t0=$SECONDS
  echo "=== $name"
  "$@"
  echo "=== $name done in $((SECONDS - t0)) s"
}

for stage in "${stages[@]}"; do
  case "$stage" in
    data)    run_stage data    "$PYTHON" scripts/00_generate_data.py --out "$DATA" "${sysopt[@]}" "${common[@]}" ;;
    train)   run_stage train   "$PYTHON" scripts/10_train.py --runs "$RUNS" --data "$DATA" "${sysopt[@]}" "${seedopt[@]}" "${gpuopt[@]}" "${jobopt[@]}" "${common[@]}" ;;
    eval)    run_stage eval    "$PYTHON" scripts/20_evaluate.py --runs "$RUNS" --data "$DATA" "${sysopt[@]}" "${seedopt[@]}" "${gpuopt[@]}" "${jobopt[@]}" "${common[@]}" ;;
    ledger)  run_stage ledger  "$PYTHON" scripts/30_build_ledger.py --runs "$RUNS" --out "$RES" "${common[@]}" ;;
    tables)
      run_stage tables "$PYTHON" scripts/40_build_tables.py --results "$RES" --out "$OUT/tables" "${lenient[@]}"
      run_stage effect_sizes "$PYTHON" scripts/41_effect_sizes.py --results "$RES" --out "$OUT/tables" ;;
    figures) run_stage figures "$PYTHON" scripts/50_build_figures.py --results "$RES/figures" --out "$OUT/figures" ;;
  esac
done
echo "finished: ${stages[*]}"
