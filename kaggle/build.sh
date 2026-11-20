#!/usr/bin/env bash
# Rebuild run_kaggle.sh from the project sources: header + every source file as a heredoc + footer.
# Then check it: extract every embedded file exactly as Kaggle will, and compare it byte for byte with its source.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=run_kaggle.sh
DELIM=__KRUM_TESTBED_EOF__
FILES=(poison_testbed/*.py poison_testbed/alarm_xp5.json run_experiment.py tests/test_testbed.py analyze.py gru_experiment.py check_probe.py deploy_xp3.py pilot_xp5.py gru_xp5.py deploy_xp5.py)

if grep -l "$DELIM" "${FILES[@]}"; then echo "delimiter found in a source file" >&2; exit 1; fi
{
  cat kaggle/header.sh
  for f in "${FILES[@]}"; do
    printf "cat > %s <<'%s'\n" "$f" "$DELIM"
    tr -d '\r' < "$f"
    # A source without a final newline would glue the delimiter to its last line: the heredoc would then
    # swallow the next files (xp6: alarm_xp5.json). Always end the file's text with a newline.
    [ -z "$(tail -c1 "$f")" ] || printf "\n"
    printf "%s\n" "$DELIM"
  done
  cat kaggle/footer.sh
} | tr -d '\r' > "$OUT"
chmod +x "$OUT"

# Check: run only the 'cat > file <<DELIM ... DELIM' blocks in an empty folder and compare with the sources
# (expected content = the source without \r, with a final newline).
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/poison_testbed" "$TMP/tests"
awk -v d="$DELIM" '/^cat > .* <<'"'"'/ {on=1} on {print} $0 == d {on=0}' "$OUT" > "$TMP/extract.sh"
(cd "$TMP" && bash extract.sh)
n_delim=$(grep -c "^${DELIM}\$" "$OUT")
bad=0
for f in "${FILES[@]}"; do
  if ! cmp -s <(tr -d '\r' < "$f"; [ -z "$(tail -c1 "$f")" ] || printf "\n") "$TMP/$f"; then
    echo "MISMATCH after extraction: $f" >&2; bad=1
  fi
done
if [ "$n_delim" -ne "${#FILES[@]}" ] || [ "$bad" -ne 0 ]; then
  echo "run_kaggle.sh is broken: $n_delim delimiters for ${#FILES[@]} files" >&2; exit 1
fi
echo "wrote $OUT ($(wc -l < "$OUT") lines) from ${#FILES[@]} files; extraction check: all ${#FILES[@]} files identical"
