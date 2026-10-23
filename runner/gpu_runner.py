"""GPU runner sederhana untuk menjalankan pekerjaan proyek di laptop.

Cara kerja:
- Memantau folder jobs/pending/ untuk file *.json.
- Tiap job hanya boleh menjalankan notebook di root/notebooks/, script di scripts/,
  atau script di jobs/tasks/ (path di luar folder proyek ditolak).
- Output ditulis ke jobs/logs/<job>.log, status akhir ke jobs/done/<job>.json.
- jobs/heartbeat.json diperbarui tiap 30 detik (waktu, status, job aktif, GPU).
- Buat file jobs/STOP untuk menghentikan runner setelah job aktif selesai,
  atau file jobs/KILL untuk menghentikan job aktif saat itu juga (file KILL
  otomatis dihapus). Ctrl+C juga menghentikan runner.

Format job (contoh jobs/pending/010_env.json):
    {"target": "scripts/env_check.py", "args": [], "timeout_min": 10}

Hanya memakai library standar Python.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JOBS = ROOT / "jobs"
PENDING, RUNNING, DONE, LOGS = (JOBS / d for d in ("pending", "running", "done", "logs"))
for d in (PENDING, RUNNING, DONE, LOGS):
    d.mkdir(parents=True, exist_ok=True)
ALLOWED_DIRS = [(ROOT / "notebooks").resolve(), (ROOT / "scripts").resolve(), (ROOT / "jobs" / "tasks").resolve()]
# notebook di root folder proyek juga diizinkan (notebook tunggal)
PY = sys.executable

state = {"status": "idle", "job": None, "job_started": None}


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


def gpu_info():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or out.stderr.strip()
    except Exception as e:  # nvidia-smi tidak ada
        return f"nvidia-smi tidak tersedia: {e}"


def heartbeat_loop():
    while True:
        hb = {"time": now(), "pid": os.getpid(), "python": PY, **state, "gpu": gpu_info(),
              "pending": sorted(p.name for p in PENDING.glob("*.json"))}
        tmp = JOBS / "heartbeat.tmp"
        tmp.write_text(json.dumps(hb, indent=2), encoding="utf-8")
        tmp.replace(JOBS / "heartbeat.json")
        time.sleep(30)


def kill_tree(proc):
    """Hentikan proses beserta anak-anaknya (kernel Jupyter, worker paralel)."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        proc.kill()
    proc.wait(timeout=30)


def resolve_target(target: str) -> Path:
    p = (ROOT / target).resolve()
    at_root_nb = p.parent == ROOT.resolve() and p.suffix == ".ipynb"
    if not at_root_nb and not any(str(p).startswith(str(a) + os.sep) for a in ALLOWED_DIRS):
        raise ValueError(f"Target di luar notebooks/ atau scripts/: {target}")
    if not p.exists():
        raise FileNotFoundError(target)
    if p.suffix not in (".ipynb", ".py"):
        raise ValueError("Hanya .ipynb atau .py")
    return p


def run_job(job_file: Path):
    name = job_file.stem
    running = RUNNING / job_file.name
    shutil.move(str(job_file), running)
    spec = json.loads(running.read_text(encoding="utf-8"))
    log_path = LOGS / f"{name}.log"
    result = {"job": name, "spec": spec, "started": now()}
    state.update(status="running", job=name, job_started=result["started"])
    t0 = time.time()
    try:
        target = resolve_target(spec["target"])
        args = [str(a) for a in spec.get("args", [])]
        if target.suffix == ".ipynb":
            cmd = [PY, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                   "--ExecutePreprocessor.timeout=-1", str(target)]
        else:
            cmd = [PY, "-u", str(target), *args]
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        with open(log_path, "w", encoding="utf-8") as log:
            log.write(f"[{now()}] CMD: {' '.join(cmd)}\n")
            log.flush()
            proc = subprocess.Popen(cmd, cwd=str(ROOT if target.suffix == ".py" else target.parent), stdout=log, stderr=subprocess.STDOUT, env=env)
            deadline = time.time() + spec.get("timeout_min", 600) * 60
            while proc.poll() is None:
                if (JOBS / "KILL").exists() or time.time() > deadline:
                    reason = "killed" if (JOBS / "KILL").exists() else "timeout"
                    (JOBS / "KILL").unlink(missing_ok=True)
                    kill_tree(proc)
                    log.write(f"\n[runner] job dihentikan: {reason}\n")
                    result["exit_code"] = reason
                    break
                time.sleep(5)
        result.setdefault("exit_code", proc.returncode)
    except Exception as e:
        result["exit_code"] = "error"
        result["error"] = repr(e)
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n[runner] ERROR: {e!r}\n")
    result["finished"] = now()
    result["duration_min"] = round((time.time() - t0) / 60, 2)
    (DONE / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    running.unlink(missing_ok=True)
    state.update(status="idle", job=None, job_started=None)
    print(f"[{now()}] selesai {name}: exit={result['exit_code']} ({result['duration_min']} menit)", flush=True)


def main():
    print(f"[{now()}] GPU runner aktif di {ROOT}")
    print("Biarkan jendela ini terbuka. Tekan Ctrl+C atau buat file jobs/STOP untuk berhenti.", flush=True)
    # job yang tertinggal di running/ (misal laptop mati) dikembalikan ke pending
    for f in RUNNING.glob("*.json"):
        shutil.move(str(f), PENDING / f.name)
    threading.Thread(target=heartbeat_loop, daemon=True).start()
    while not (JOBS / "STOP").exists():
        jobs = sorted(PENDING.glob("*.json"))
        if jobs:
            print(f"[{now()}] mulai {jobs[0].stem}", flush=True)
            run_job(jobs[0])
        else:
            time.sleep(10)
    print("STOP ditemukan, runner berhenti.")


if __name__ == "__main__":
    main()
