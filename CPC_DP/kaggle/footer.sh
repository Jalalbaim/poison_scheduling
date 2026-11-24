# ---------------------------------------------------------------------------
# 3. Tests (testbed reproducibility + onset-twin plans). Stops here on failure.
# ---------------------------------------------------------------------------
echo "== tests"
"$PY" -m pytest tests/test_testbed.py tests/test_onset_twins.py tests/test_xp7.py -q

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

run_xp7() {  # $1 = a | b | twins
  local args=() out=results/xp7_$1 name=xp7_$1
  if [ "${CPC_SMOKE:-0}" = "1" ]; then args=(--smoke); out=results/xp7_$1_smoke; name=xp7_$1_smoke; fi
  echo "== xp7 part $1"
  "$PY" -u xp7.py --part "$1" --jobs "$JOBS" --out "$out" "${args[@]}"
  (cd "$WORK" && rm -f "$WORK/../$name.zip" && zip -qr "$WORK/../$name.zip" "$out/all_blocks.csv" "$out/manifest.csv")
  echo "== saved $(dirname "$WORK")/$name.zip"
}

run_xp7_deploy() {
  local args=() out=results/deploy_xp7 name=xp7_deploy
  if [ "${CPC_SMOKE:-0}" = "1" ]; then args=(--smoke); out=results/deploy_xp7/smoke; name=xp7_deploy_smoke; fi
  echo "== xp7 deployment: gru / CPC A2 trained on gamma 0 and on the hard alarm, as Krum controllers"
  "$PY" -u deploy_xp7.py --jobs "$JOBS" --models gru_g0=g0/gru cpc_g0=g0/A2 gru_hard=hard/gru cpc_hard=hard/A2     "${args[@]}"
  (cd "$WORK" && rm -f "$WORK/../$name.zip" && zip -qr "$WORK/../$name.zip" results/deploy_xp7)
  echo "== saved $(dirname "$WORK")/$name.zip"
}

SECONDS=0
case "$STAGE" in
  xp6_eval) run_xp6_eval ;;
  xp7_a) run_xp7 a ;;
  xp7_b) run_xp7 b ;;
  xp7_twins) run_xp7 twins ;;
  xp7_deploy) run_xp7_deploy ;;
esac
echo "== done in $((SECONDS / 60)) min"
