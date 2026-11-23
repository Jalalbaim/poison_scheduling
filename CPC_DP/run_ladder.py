"""Train the ablation ladder on xp6, locally: A0-A3 at hidden 32 / 64 / 128, then A4 (feedback) at the hidden size
with the lowest A3 validation MSE. One training at a time (two ran out of RAM locally). Finished models are skipped on a rerun.

    ../.venv/Scripts/python run_ladder.py            # -> results/cpc/<preset>_h<hidden>[_fb]/, logs in results/cpc/logs/
"""

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "cpc"
LOGS = OUT / "logs"
SIZES = [32, 64, 128]
JOBS = 1  # 2 ran out of RAM on the laptop


def folder(preset: str, hidden: int) -> Path:
    return OUT / f"{preset}_h{hidden}{'_fb' if preset == 'A4' else ''}"


def train(job: tuple[str, int]) -> tuple[str, int, float | None]:
    preset, hidden = job
    f = folder(preset, hidden)
    if (f / "report.json").exists():
        print(f"skip {f.name} (done)", flush=True)
    else:
        print(f"start {f.name}", flush=True)
        with open(LOGS / f"{f.name}.log", "w") as log:
            r = subprocess.run([sys.executable, "-u", str(HERE / "train_cpc.py"), "--preset", preset,
                                "--hidden", str(hidden), "--out", str(f)], stdout=log, stderr=subprocess.STDOUT,
                               cwd=HERE)
        if r.returncode != 0:
            print(f"FAILED {f.name} (see {LOGS / (f.name + '.log')})", flush=True)
            return preset, hidden, None
    val = json.loads((f / "report.json").read_text())["best_val_mse"]
    print(f"done  {f.name}: best val MSE {val:.5f}", flush=True)
    return preset, hidden, val


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    jobs = [(p, h) for p in ("A0", "A1", "A2", "A3") for h in SIZES]
    with ThreadPoolExecutor(JOBS) as pool:
        results = list(pool.map(train, jobs))
    a3 = [(v, h) for p, h, v in results if p == "A3" and v is not None]
    if not a3:
        raise SystemExit("no A3 model finished: A4 not started")
    best_h = min(a3)[1]
    print(f"A3 best hidden = {best_h}: training A4 (feedback) at that size", flush=True)
    results.append(train(("A4", best_h)))
    summary = [{"preset": p, "hidden": h, "best_val_mse": v, "folder": folder(p, h).name} for p, h, v in results]
    (OUT / "ladder.json").write_text(json.dumps({"a4_hidden": best_h, "models": summary}, indent=1))
    print("\n".join(f"{s['folder']:>12}: {s['best_val_mse']}" for s in summary))


if __name__ == "__main__":
    main()
