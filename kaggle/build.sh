#!/usr/bin/env bash
# Rebuild run_kaggle.sh from the project sources: header + every source file as a heredoc + footer.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=run_kaggle.sh
DELIM=__KRUM_TESTBED_EOF__
FILES=(poison_testbed/*.py run_experiment.py tests/test_testbed.py analyze.py gru_experiment.py check_probe.py deploy_xp3.py pilot_xp5.py gru_xp5.py deploy_xp5.py)

if grep -l "$DELIM" "${FILES[@]}"; then echo "delimiter found in a source file" >&2; exit 1; fi
{
  cat kaggle/header.sh
  for f in "${FILES[@]}"; do
    printf "cat > %s <<'%s'\n" "$f" "$DELIM"
    tr -d '\r' < "$f"
    printf "%s\n" "$DELIM"
  done
  cat kaggle/footer.sh
} | tr -d '\r' > "$OUT"
chmod +x "$OUT"
echo "wrote $OUT ($(wc -l < "$OUT") lines) from ${#FILES[@]} files"
