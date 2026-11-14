
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

# C from xp1, pinned so that re-running an xp1 schedule reproduces the xp1 run exactly.
XP1_C=3.0552712798118593

run_counterfactual() {
  echo "== GRU check 3: true counterfactuals (4 seeds x 16 conditions x 6 schedules = 384 runs)"
  "$PY" -u run_experiment.py counterfactual --jobs "$JOBS" --set C=$XP1_C $EXTRA
  (cd "$WORK" && zip -qr "$WORK/../results_cf.zip" runs_cf)
  echo "== saved $(dirname "$WORK")/results_cf.zip"
}

SECONDS=0
case "$STAGE" in
  sanity)         run_sanity ;;
  collect)        run_collect ;;
  all)            run_sanity; run_collect ;;
  counterfactual) run_counterfactual ;;
esac
echo "== done in $((SECONDS / 60)) min"
