# ---------------------------------------------------------------------------
# 3. Tests (testbed reproducibility + onset-twin plans). Stops here on failure.
# ---------------------------------------------------------------------------
echo "== tests"
"$PY" -m pytest tests/test_testbed.py tests/test_onset_twins.py -q

# ---------------------------------------------------------------------------
# 4. Stages.
# ---------------------------------------------------------------------------
run_xp6_eval() {
  local args=() out=results/xp6_eval
  if [ "${CPC_SMOKE:-0}" = "1" ]; then args=(--smoke); out=results/xp6_eval_smoke; fi
  echo "== xp6_eval: onset twins (evaluation only; off reference + on-plans [onset+d, onset+d+k))"
  "$PY" -u onset_twins.py --jobs "$JOBS" --out "$out" "${args[@]}"
  (cd "$WORK" && rm -f "$WORK/../xp6_eval.zip" && zip -qr "$WORK/../xp6_eval.zip" "$out/all_blocks.csv" "$out/manifest.csv")
  echo "== saved $(dirname "$WORK")/xp6_eval.zip"
}

SECONDS=0
case "$STAGE" in
  xp6_eval) run_xp6_eval ;;
esac
echo "== done in $((SECONDS / 60)) min"
