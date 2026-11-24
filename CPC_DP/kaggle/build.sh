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
        "tests/test_onset_twins.py:tests/test_onset_twins.py" "xp7.py:xp7.py" "tests/test_xp7.py:tests/test_xp7.py"
        "$G/analyze.py:analyze.py" "$G/gru_experiment.py:gru_experiment.py" "deploy_xp7.py:deploy_xp7.py")
for f in cpc/*.py; do PAIRS+=("$f:cpc/$(basename "$f")"); done
# Trained models deployed by stage xp7_deploy (binary: a base64 tar.gz, checked below).
DEPLOY_MODELS=(g0/gru g0/A2 hard/gru hard/A2)
MODEL_FILES=()
for m in "${DEPLOY_MODELS[@]}"; do for f in report.json norm.npz config.json gru.pt cpc.pt; do
  [ -f "results/xp7_models/$m/$f" ] && MODEL_FILES+=("xp7_models/$m/$f"); done; done
MODELS_B64=$(tar -C results -czf - "${MODEL_FILES[@]}" | base64 -w 76)

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
  printf "mkdir -p results && base64 -d > xp7_models.tgz <<'%s'\n%s\n%s\n" "$DELIM" "$MODELS_B64" "$DELIM"
  printf "tar -C results -xzf xp7_models.tgz && rm xp7_models.tgz\n"
  cat kaggle/footer.sh
} | tr -d '\r' > "$OUT"
chmod +x "$OUT"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/poison_testbed" "$TMP/tests" "$TMP/cpc"
awk -v d="$DELIM" '/^(cat > |mkdir -p results && base64).* <<'"'"'/ {on=1} on {print} $0 == d {on=0} /^tar -C results/ {print}' "$OUT" > "$TMP/extract.sh"
(cd "$TMP" && bash extract.sh)
bad=0
for p in "${PAIRS[@]}"; do
  src=${p%%:*}; dst=${p#*:}
  if ! cmp -s <(tr -d '\r' < "$src"; [ -z "$(tail -c1 "$src")" ] || printf "\n") "$TMP/$dst"; then
    echo "MISMATCH after extraction: $dst" >&2; bad=1
  fi
done
for f in "${MODEL_FILES[@]}"; do
  if ! cmp -s "results/$f" "$TMP/results/$f"; then echo "MISMATCH after extraction: $f" >&2; bad=1; fi
done
n_delim=$(grep -c "^${DELIM}\$" "$OUT")
n_delim=$((n_delim - 1))  # the model archive
if [ "$n_delim" -ne "${#PAIRS[@]}" ] || [ "$bad" -ne 0 ]; then
  echo "$OUT is broken: $n_delim delimiters for ${#PAIRS[@]} files" >&2; exit 1
fi
echo "wrote $OUT ($(wc -l < "$OUT") lines, $(du -h "$OUT" | cut -f1)) from ${#PAIRS[@]} files + ${#MODEL_FILES[@]} model files; extraction check: all identical"
