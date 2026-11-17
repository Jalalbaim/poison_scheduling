
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

run_xp2() {
  echo "== xp2: windowed backdoor + Krum, decisions every 2 rounds, schedules with kills (260 runs)"
  "$PY" -u run_experiment.py xp2 --jobs "$JOBS" $EXTRA
  (cd "$WORK" && zip -qr "$WORK/../results_xp2.zip" runs_xp2)
  echo "== saved $(dirname "$WORK")/results_xp2.zip"
}

run_xp3() {
  echo "== xp3: stealthy windowed backdoors (1/8, 2/8, 8/8) + no attack, probe schedules, 80 paired twins (400 runs)"
  "$PY" -u run_experiment.py xp3 --jobs "$JOBS" $EXTRA
  (cd "$WORK" && zip -qr "$WORK/../results_xp3.zip" runs_xp3)
  echo "== saved $(dirname "$WORK")/results_xp3.zip"
}

run_xp4() {
  echo "== xp4: xp3 design + signals v2, onset-focused schedules, 80 paired twins (400 runs)"
  "$PY" -u run_experiment.py xp4 --jobs "$JOBS" $EXTRA
  (cd "$WORK" && zip -qr "$WORK/../results_xp4.zip" runs_xp4)
  echo "== saved $(dirname "$WORK")/results_xp4.zip"
}

run_xp5_pilot() {
  echo "== xp5 pilot: Fashion-MNIST; damage grid (9 attack settings x 3 defenses x on/off x 2 seeds) + erasure windows (162 runs)"
  "$PY" -u pilot_xp5.py --jobs "$JOBS" | tee pilot_xp5_log.txt
  cp pilot_xp5_log.txt results/pilot_xp5/
  (cd "$WORK" && zip -qr "$WORK/../xp5_pilot.zip" results/pilot_xp5)
  echo "== saved $(dirname "$WORK")/xp5_pilot.zip"
}

# GRU candidates for xp3: name hidden lr weight_decay. xp1/xp2 settings (64, 2e-3, 0) overfit xp3 after 2 epochs.
XP3_SWEEP=("a 64 5e-4 0" "b 32 5e-4 1e-3" "c 64 2e-4 1e-2" "d 128 5e-4 1e-2")

find_xp3_data() {
  if [ -n "${XP3_DATA:-}" ]; then echo "$XP3_DATA"; return; fi
  local f
  f=$(find /kaggle/input -path '*runs_xp3/all_blocks.csv' 2>/dev/null | head -1)
  [ -z "$f" ] && f=$(find /kaggle/input -name all_blocks.csv 2>/dev/null | head -1)
  if [ -z "$f" ]; then  # a zip that Kaggle did not extract
    local z; z=$(find /kaggle/input -name 'results_xp3*.zip' 2>/dev/null | head -1)
    if [ -n "$z" ]; then unzip -qo "$z" 'runs_xp3/all_blocks.csv' -d "$WORK/xp3_unzipped" && f="$WORK/xp3_unzipped/runs_xp3/all_blocks.csv"; fi
  fi
  echo "$f"
}

run_xp3_deploy() {
  local data; data=$(find_xp3_data)
  if [ -z "$data" ] || [ ! -f "$data" ]; then
    echo "xp3 data not found: attach results_xp3.zip as a dataset (Add Input) or set XP3_DATA=/path/all_blocks.csv"; exit 1
  fi
  mkdir -p results/xp3/runs_xp3 results/xp3/gru_sweep
  cp "$data" results/xp3/runs_xp3/all_blocks.csv
  "$PY" - <<'PYCHECK'
import pandas as pd
d = pd.read_csv("results/xp3/runs_xp3/all_blocks.csv")
n = d.run_id.nunique(); rows = d.groupby("run_id").size().unique()
print(f"xp3 data: {n} runs, rows per run {rows}")
if n != 400 or list(rows) != [251]:
    raise SystemExit("unexpected xp3 data: expected 400 runs x 251 rows")
PYCHECK
  echo "== xp3_deploy 1/3: GRU sweep on xp3 (${#XP3_SWEEP[@]} models, one at a time)"
  for cfg in "${XP3_SWEEP[@]}"; do
    set -- $cfg
    if [ -f "results/xp3/gru_sweep/$1/report.json" ]; then echo "   model $1 already trained"; continue; fi
    rm -rf "results/xp3/gru_sweep/$1"
    echo "   model $1: hidden $2, lr $3, weight decay $4"
    "$PY" -u gru_experiment.py --runs results/xp3/runs_xp3/all_blocks.csv --out "results/xp3/gru_sweep/$1" \
      --hidden "$2" --lr "$3" --wd "$4" --patience 40 > "results/xp3/gru_sweep/$1.log" 2>&1 \
      || { echo "   model $1 failed:"; tail -20 "results/xp3/gru_sweep/$1.log"; exit 1; }
    grep -E "^model:" "results/xp3/gru_sweep/$1.log"
  done
  echo "== xp3_deploy 2/3: model selection on validation seeds (check P: hold vs keep probing)"
  "$PY" -u check_probe.py --select results/xp3/gru_sweep/a results/xp3/gru_sweep/b results/xp3/gru_sweep/c results/xp3/gru_sweep/d
  local sel; sel=$("$PY" -c "import json; print(json.load(open('results/xp3/gru_selected.json'))['selected'])")
  echo "== xp3_deploy 3/3: deployment with $sel (6 policies x 4 conditions x 4 seeds = 96 runs)"
  "$PY" -u deploy_xp3.py --gru "$sel" --jobs "$JOBS" | tee results/deploy_xp3_log.txt
  (cd "$WORK" && zip -qr "$WORK/../xp3_deploy.zip" results/xp3/gru_sweep results/xp3/gru_selected.json \
     results/deploy_xp3 results/deploy_xp3_log.txt)
  echo "== saved $(dirname "$WORK")/xp3_deploy.zip"
}

SECONDS=0
case "$STAGE" in
  sanity)         run_sanity ;;
  collect)        run_collect ;;
  all)            run_sanity; run_collect ;;
  counterfactual) run_counterfactual ;;
  xp2)            run_xp2 ;;
  xp3)            run_xp3 ;;
  xp4)            run_xp4 ;;
  xp5_pilot)      run_xp5_pilot ;;
  xp3_deploy)     run_xp3_deploy ;;
esac
echo "== done in $((SECONDS / 60)) min"
