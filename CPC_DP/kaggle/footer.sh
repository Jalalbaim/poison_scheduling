# ---------------------------------------------------------------------------
# 3. Tests (testbed reproducibility + onset-twin plans). Stops here on failure.
# ---------------------------------------------------------------------------
echo "== tests"
"$PY" -m pytest tests/test_testbed.py tests/test_onset_twins.py tests/test_xp7.py tests/test_xp8.py tests/test_learn8.py -q

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
  local args=() out=results/xp7/deploy name=xp7_deploy
  if [ "${CPC_SMOKE:-0}" = "1" ]; then args=(--smoke); out=results/xp7/deploy/smoke; name=xp7_deploy_smoke; fi
  echo "== xp7 deployment: gru / CPC A2 trained on gamma 0 and on the hard alarm, as Krum controllers"
  "$PY" -u deploy_xp7.py --jobs "$JOBS" --models gru_g0=g0/gru cpc_g0=g0/A2 gru_hard=hard/gru cpc_hard=hard/A2     "${args[@]}"
  (cd "$WORK" && rm -f "$WORK/../$name.zip" && zip -qr "$WORK/../$name.zip" results/xp7/deploy)
  echo "== saved $(dirname "$WORK")/$name.zip"
}

run_xp8() {  # $1 = 1 | 2
  local args=() out=results/xp8/part$1 name=xp8_part$1
  if [ "${CPC_SMOKE:-0}" = "1" ]; then args=(--smoke); out=results/xp8/part$1_smoke; name=xp8_part$1_smoke; fi
  echo "== xp8 part $1: trunks (Krum always off / always on) + paired branches, decision at step t uses step t"
  "$PY" -u xp8.py --part "$1" --jobs "$JOBS" --budget-hours "${XP8_BUDGET_H:-11}" "${args[@]}"
  (cd "$WORK" && rm -f "$WORK/../$name.zip" && zip -qr "$WORK/../$name.zip" "$out/all_blocks.parquet" "$out/manifest.csv" "$out/jobs.csv" "$out/summary.json")
  echo "== saved $(dirname "$WORK")/$name.zip"
}

run_xp8_learn() {
  local args=() out=learn name=xp8_learn
  if [ "${CPC_SMOKE:-0}" = "1" ]; then args=(--smoke); out=learn_smoke; name=xp8_learn_smoke; fi
  # data: part1/ and part2/ (all_blocks.parquet + manifest.csv) from a notebook input dataset, or XP8_DATA
  local src="${XP8_DATA:-}" f=""
  if [ -z "$src" ]; then
    f=$(find -L /kaggle/input -path '*part1/all_blocks.parquet' 2>/dev/null | head -1 || true)
    if [ -n "$f" ]; then src=$(dirname "$(dirname "$f")"); fi
  fi
  for p in part1 part2; do
    if [ -z "$src" ] || [ ! -f "$src/$p/all_blocks.parquet" ] || [ ! -f "$src/$p/manifest.csv" ]; then
      echo "xp8 data not found (need part1/ and part2/ with all_blocks.parquet + manifest.csv): add the xp8_data dataset to the notebook, or set XP8_DATA" >&2
      exit 1
    fi
  done
  echo "== xp8 data: $src"
  for p in part1 part2; do
    mkdir -p results/xp8/$p
    for g in all_blocks.parquet manifest.csv; do ln -sf "$src/$p/$g" results/xp8/$p/$g; done
  done
  echo "== xp8 learn: models A (direct gain) and B (outcome Causal CPC), 3 seeds each, offline test"
  "$PY" -u learn8.py --res results/xp8 "${args[@]}"
  echo "== xp8 live: best seed of A and of B decide Krum at every step of the test setups"
  "$PY" -u live8.py --res results/xp8 --jobs "$JOBS" "${args[@]}"
  (cd "$WORK/results/xp8" && rm -f "$WORK/../$name.zip" && zip -qr "$WORK/../$name.zip" "$out" -x "*/live/runs/*")
  echo "== saved $(dirname "$WORK")/$name.zip"
}

SECONDS=0
case "$STAGE" in
  xp6_eval) run_xp6_eval ;;
  xp7_a) run_xp7 a ;;
  xp7_b) run_xp7 b ;;
  xp7_twins) run_xp7 twins ;;
  xp7_deploy) run_xp7_deploy ;;
  xp8_1) run_xp8 1 ;;
  xp8_2) run_xp8 2 ;;
  xp8_learn) run_xp8_learn ;;
esac
echo "== done in $((SECONDS / 60)) min"
