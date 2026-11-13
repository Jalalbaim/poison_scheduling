
# ---------------------------------------------------------------------------
# 3. Tests (reproducibility, schedule dispatch, labels). Stops here on failure.
# ---------------------------------------------------------------------------
echo "== tests"
"$PY" -m pytest tests/test_testbed.py -q

# ---------------------------------------------------------------------------
# 4. Stages.
# ---------------------------------------------------------------------------
# shellcheck disable=SC2086  # EXTRA is intentionally word-split into --set arguments
run_sanity() {
  if [ ! -f calibration.json ]; then
    echo "== calibrate C"
    "$PY" -u run_experiment.py calibrate
  fi
  echo "== Step 2: sanity grid (126 runs)"
  "$PY" -u run_experiment.py sanity --jobs "$JOBS" $EXTRA
  echo "== Step 2 table"
  "$PY" -u run_experiment.py report
  (cd "$WORK" && zip -qr "$WORK/../results_sanity.zip" calibration.json runs_sanity)
  echo "== saved $(dirname "$WORK")/results_sanity.zip"
}

run_collect() {
  if [ ! -f calibration.json ]; then
    echo "== calibrate C"
    "$PY" -u run_experiment.py calibrate
  fi
  echo "== Step 3: randomized collection (720 runs)"
  "$PY" -u run_experiment.py collect --jobs "$JOBS" $EXTRA
  (cd "$WORK" && zip -qr "$WORK/../results_collect.zip" calibration.json runs)
  echo "== saved $(dirname "$WORK")/results_collect.zip"
}

SECONDS=0
case "$STAGE" in
  sanity)  run_sanity ;;
  collect) run_collect ;;
  all)     run_sanity; run_collect ;;
esac
echo "== done in $((SECONDS / 60)) min"
