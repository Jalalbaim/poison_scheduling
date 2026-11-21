#!/usr/bin/env bash
# Rebuild run_kaggle_cpc.sh: header + every source file as a heredoc + footer (same scheme as GRU_mnist/kaggle/build.sh).
# Sources: the GRU_mnist testbed (read, never modified) and CPC_DP scripts; each is written flat into $WORK on Kaggle.
# Then check it: extract every embedded file exactly as Kaggle will, and compare it byte for byte with its source.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=run_kaggle_cpc.sh
DELIM=__CPC_DP_EOF__
G=../GRU_mnist
# "source:destination" (destination relative to $WORK on Kaggle)
PAIRS=()
for f in "$G"/poison_testbed/*.py; do PAIRS+=("$f:poison_testbed/$(basename "$f")"); done
PAIRS+=("$G/poison_testbed/alarm_xp5.json:poison_testbed/alarm_xp5.json" "$G/run_experiment.py:run_experiment.py"
        "$G/tests/test_testbed.py:tests/test_testbed.py" "onset_twins.py:onset_twins.py"
        "tests/test_onset_twins.py:tests/test_onset_twins.py")

for p in "${PAIRS[@]}"; do if grep -q "$DELIM" "${p%%:*}"; then echo "delimiter found in ${p%%:*}" >&2; exit 1; fi; done
{
  cat kaggle/header.sh
  for p in "${PAIRS[@]}"; do
    src=${p%%:*}; dst=${p#*:}
    printf "cat > %s <<'%s'\n" "$dst" "$DELIM"
    tr -d '\r' < "$src"
    [ -z "$(tail -c1 "$src")" ] || printf "\n"
    printf "%s\n" "$DELIM"
  done
  cat kaggle/footer.sh
} | tr -d '\r' > "$OUT"
chmod +x "$OUT"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/poison_testbed" "$TMP/tests"
awk -v d="$DELIM" '/^cat > .* <<'"'"'/ {on=1} on {print} $0 == d {on=0}' "$OUT" > "$TMP/extract.sh"
(cd "$TMP" && bash extract.sh)
bad=0
for p in "${PAIRS[@]}"; do
  src=${p%%:*}; dst=${p#*:}
  if ! cmp -s <(tr -d '\r' < "$src"; [ -z "$(tail -c1 "$src")" ] || printf "\n") "$TMP/$dst"; then
    echo "MISMATCH after extraction: $dst" >&2; bad=1
  fi
done
n_delim=$(grep -c "^${DELIM}\$" "$OUT")
if [ "$n_delim" -ne "${#PAIRS[@]}" ] || [ "$bad" -ne 0 ]; then
  echo "$OUT is broken: $n_delim delimiters for ${#PAIRS[@]} files" >&2; exit 1
fi
echo "wrote $OUT ($(wc -l < "$OUT") lines) from ${#PAIRS[@]} files; extraction check: all identical"
