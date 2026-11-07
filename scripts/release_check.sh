#!/usr/bin/env bash
# Checks before publishing the repository: private paths / names / e-mail addresses / environment names in tracked files,
# file sizes and binaries, the forbidden-term test and the test suite; optionally the end-to-end smoke run.
#
#   scripts/release_check.sh [--smoke]
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
smoke=0
[[ "${1:-}" == "--smoke" ]] && smoke=1

fail=0
report() { echo "FAIL: $1"; fail=1; }

# Files that would be published: tracked or untracked-but-not-ignored, excluding the private dev/ tooling.
mapfile -t files < <(git ls-files --cached --others --exclude-standard | grep -v '^dev/' | while read -r f; do [[ -f "$f" ]] && echo "$f"; done)
# Files that contain the search patterns themselves.
selfs=(scripts/release_check.sh tests/test_forbidden_terms.py)
is_self() { local f; for f in "${selfs[@]}"; do [[ "$1" == "$f" ]] && return 0; done; return 1; }

text_files=()
for f in "${files[@]}"; do
  is_self "$f" && continue
  grep -qI . "$f" 2>/dev/null && text_files+=("$f")
done

echo "== private paths, user names, e-mail addresses, environment names (${#text_files[@]} text files)"
scan() {  # scan LABEL REGEX
  local hits
  hits=$(grep -nIE "$2" "${text_files[@]}" 2>/dev/null | head -20)
  [[ -n "$hits" ]] && { report "$1"; echo "$hits"; }
}
scan "absolute home path" '/homes?/'
scan "scratch path" '/tmp/claude'
scan "user name" "$(id -un)|yta""chibana"  # the account running the check and the author handle
scan "e-mail address" '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
scan "environment / session manager name" 'con''da|tm''ux|exp''-env'  # split so this file does not match itself

echo "== file sizes and binaries"
for f in "${files[@]}"; do
  size=$(stat -c %s "$f")
  (( size > 5 * 1024 * 1024 )) && report "$f is larger than 5 MB ($size bytes)"
  if [[ "$size" -gt 0 ]] && ! grep -qI . "$f" 2>/dev/null; then
    [[ "$f" == assets/*.pdf ]] || report "binary file outside assets/*.pdf: $f"
  fi
done

echo "== tests (including the forbidden-term test)"
"$PYTHON" -m pytest tests -q -m "not slow" || report "pytest"

if [[ $smoke -eq 1 ]]; then
  echo "== smoke run"
  PYTHON="$PYTHON" bash scripts/reproduce.sh --smoke || report "smoke run"
fi

if [[ $fail -eq 0 ]]; then echo "release check passed"; else echo "release check FAILED"; fi
exit $fail
