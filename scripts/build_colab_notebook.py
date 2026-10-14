"""Generate self-contained Colab notebooks (code embedded as a hash-checked zip; no upload).

    python scripts/build_colab_notebook.py setup     # -> notebooks/COLAB_G4_SETUP.ipynb

The notebook mounts Google Drive, verifies and extracts the embedded code, downloads PTB-XL (100 Hz)
from PhysioNet's S3 mirror on the runtime, rebuilds the prepared arrays and checks that they are
byte-identical to the local ones, then runs the task. Results go to My Drive/<drive_dir>.
Rebuild the notebook after every code change; the bundle hash identifies the code version.
"""

import _paths  # noqa: F401

import argparse
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

from bamo.config import PROJECT_ROOT
from bamo.utils import read_json

BUNDLE_DIRS = ["src", "scripts", "configs", "tests"]
BUNDLE_FILES = ["pyproject.toml", "requirements-lock.txt"]
PIP_PINS = "wfdb==4.3.1 optuna==5.0.0 dehb==0.1.1 ConfigSpace==1.2.2 pytest"


def build_bundle() -> tuple[bytes, list[str]]:
    paths = []
    for d in BUNDLE_DIRS:
        paths += [p for p in sorted((PROJECT_ROOT / d).rglob("*"))
                  if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc",)]
    paths += [PROJECT_ROOT / f for f in BUNDLE_FILES]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            info = zipfile.ZipInfo(p.relative_to(PROJECT_ROOT).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, p.read_bytes())
    return buf.getvalue(), [p.relative_to(PROJECT_ROOT).as_posix() for p in paths]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def common_cells(drive_dir: str, title: str, intro: str, prepare_reference_data: bool = True) -> list[dict]:
    bundle, names = build_bundle()
    digest = hashlib.sha256(bundle).hexdigest()
    expected = (read_json(PROJECT_ROOT / "data" / "processed" / "data_report.json")["array_sha256"]
                if prepare_reference_data else {})
    cells = [
        md(f"""
# {title}

{intro}

**In VS Code:** choose **Select Kernel → Colab → New Colab Server → G4**, then **Run All**. If the first cell reports that Google Drive is not mounted, run the Command Palette command **Colab: Mount Google Drive to Server...**, run the cell it adds, then **Run All** again.

Results are written to **My Drive/{drive_dir}**. Disconnect the runtime when finished: G4 bills while idle.
"""),
        code(f"""
import os
OUTPUT = '/content/drive/MyDrive/{drive_dir}'
if not os.path.ismount('/content/drive'):
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except Exception as error:
        raise RuntimeError(
            'Google Drive is not mounted. In VS Code, open the Command Palette and run '
            '"Colab: Mount Google Drive to Server...", run the cell it adds, then rerun this cell. '
            f'({{error}})')
assert os.path.ismount('/content/drive') and os.path.isdir('/content/drive/MyDrive'), 'Drive mount failed'
os.makedirs(OUTPUT, exist_ok=True)
print('Results will be saved in Google Drive:', OUTPUT.replace('/content/drive/', ''))
"""),
        md(f"Embedded, hash-checked code bundle ({len(names)} files: `src/`, `scripts/`, `configs/`, `tests/`). Do not edit; rebuild with `scripts/build_colab_notebook.py`."),
        code(f"BUNDLE_B64 = '{base64.b64encode(bundle).decode()}'"),
        code(f"""
import base64, hashlib, io, json, pathlib, subprocess, sys, zipfile
bundle = base64.b64decode(BUNDLE_B64)
BUNDLE_SHA = '{digest}'
assert hashlib.sha256(bundle).hexdigest() == BUNDLE_SHA, 'embedded code bundle is corrupt'
root = pathlib.Path('/content/bamo')
root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(bundle)) as source:
    for name in source.namelist():
        assert not name.startswith('/') and '..' not in pathlib.PurePosixPath(name).parts
    source.extractall(root)
os.chdir(root)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *'{PIP_PINS}'.split()], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.', '--no-deps'], check=True)
sys.path.insert(0, str(root / 'src'))
import torch
assert torch.cuda.is_available(), 'Select a Colab GPU kernel (G4) before running'
environment = pathlib.Path(OUTPUT) / 'environment'
environment.mkdir(exist_ok=True)
(environment / 'nvidia_smi.txt').write_text(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
(environment / 'pip_freeze.txt').write_text(subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True).stdout)
(environment / 'bundle_sha256.txt').write_text(BUNDLE_SHA)
memory_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30
print(f'Code bundle verified ({{BUNDLE_SHA[:12]}}...). GPU: {{torch.cuda.get_device_name()}}, '
      f'{{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f}} GiB; '
      f'{{os.cpu_count()}} CPUs; {{memory_gb:.0f}} GiB RAM; torch {{torch.__version__}}')
"""),
        md("## Data\nDownload PTB-XL v1.0.3 (metadata + 100 Hz records) from PhysioNet's S3 mirror, verify every file against PhysioNet's SHA256SUMS, rebuild the prepared arrays, and check they are byte-identical to the local preparation."),
        code(f"""
import time
EXPECTED = {json.dumps(expected)}
started = time.time()
for step in (['scripts/download_ptbxl.py', '--workers', '64'], ['scripts/prepare_data.py']):
    result = subprocess.run([sys.executable, *step], capture_output=True, text=True)
    print(result.stdout[-1500:])
    assert result.returncode == 0, result.stderr[-3000:]
from bamo.utils import sha256_file
for name, digest in EXPECTED.items():
    assert sha256_file(f'data/processed/{{name}}') == digest, f'{{name}} differs from the local preparation'
print(f'Data ready and identical to the local arrays in {{time.time() - started:.0f}} s')
"""),
    ]
    if not prepare_reference_data:
        cells[-2:] = [
            md("## Raw data for development folds\nDownload and checksum raw files. The decision scripts open waveforms only from folds 1–7."),
            code("""
result = subprocess.run([sys.executable, 'scripts/download_ptbxl.py', '--workers', '64'], capture_output=True, text=True)
print(result.stdout[-2000:])
assert result.returncode == 0, result.stderr[-3000:]
""")]
    return cells


def setup_notebook() -> dict:
    cells = common_cells(
        "bamo_runs", "BAMO — G4 session 1: setup check, speed and concurrency probe",
        "Checks the code and data on the Colab G4, runs the test suite on the GPU, measures training speed "
        "(T_ref estimate) and how many training processes one G4 can run at once. About 20–30 minutes "
        "(≈ 3–5 compute units). At the end, copy the printed **SUMMARY** back to Claude.")
    cells += [
        md("## Tests"),
        code("""
result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:warnings'], capture_output=True, text=True)
print(result.stdout[-2000:])
assert result.returncode == 0, 'tests failed'
"""),
        md("## Speed check (single process)"),
        code("""
result = subprocess.run([sys.executable, 'scripts/run_pilot.py', 'speed', '--out-root', OUTPUT, '--cu-per-hour', '8.9'],
                        capture_output=True, text=True)
print(result.stdout[-4000:])
assert result.returncode == 0, result.stderr[-3000:]
"""),
        md("## Concurrency probe (K training processes sharing the G4)"),
        code("""
result = subprocess.run([sys.executable, 'scripts/concurrency_probe.py', '--levels', '1', '4', '8', '16', '24',
                         '--out', f'{OUTPUT}/pilot/concurrency.json'], capture_output=True, text=True)
print(result.stdout[-4000:])
assert result.returncode == 0, result.stderr[-3000:]
"""),
        md("## Summary (copy this back)"),
        code("""
speed = json.loads(pathlib.Path(f'{OUTPUT}/pilot/p1_speed.json').read_text())
conc = json.loads(pathlib.Path(f'{OUTPUT}/pilot/concurrency.json').read_text())
s = speed['summary']
print('SUMMARY')
print('GPU:', speed['hardware'].get('gpu'), '| torch', speed['hardware'].get('torch'), '| bundle', BUNDLE_SHA[:12])
print(f"single process: median {s['median_s_per_epoch']:.2f} s/epoch (min {s['min']:.2f}, max {s['max']:.2f}); "
      f"T_ref ~ {s['t_ref_estimate_s']:.0f} s")
for design, p in s['projection'].items():
    print(f"  {design}: {p['gpu_hours']:.0f} GPU-h sequential, ~{p['compute_units']:.0f} CU")
for probe, rows in conc['probes'].items():
    print(probe, ' '.join(f"K{r['k']}:x{r['throughput_gain']:.1f}" for r in rows))
"""),
    ]
    return cells


def pilot_notebook() -> list[dict]:
    cells = common_cells(
        "bamo_runs", "BAMO — G4 pilot (docs/06): T_ref, noise, comparator check, tau/kappa sweep",
        "Runs the whole pilot as one background job on pilot data (folds 1–6 → 7): P1 feasibility, P2 T_ref, P3 seed "
        "noise, proposed frozen values, P6 short check of all 7 methods, P4 tau/kappa sweep (10 controller runs), then "
        "the pre-specified selection rule. **About 7 hours on G4 (≈ 60–65 compute units).** It keeps running if the "
        "VS Code connection drops but the runtime stays alive; after a runtime reset, **Run All** resumes at the last "
        "unfinished step. **Session 2** (`SESSION = 2`, a few minutes, on a *different day or runtime*) is the T_ref drift check.")
    cells += [
        md("## Start the pilot in the background"),
        code("""
SESSION = 1   # 1 = full pilot; 2 = T_ref drift check (run later, in a new runtime)
pid_file = root / f'pilot_session{SESSION}.pid'
log_path = pathlib.Path(OUTPUT) / 'pilot' / 'pilot_log.txt'
def log_lines():
    return log_path.read_text(encoding='utf-8').splitlines() if log_path.exists() else []
running = pid_file.exists() and pathlib.Path(f'/proc/{pid_file.read_text().strip()}').exists()
if running:
    start_line = max([i for i, l in enumerate(log_lines()) if 'PILOT START' in l or 'SESSION 2 START' in l], default=0)
    print('Pilot already running, PID', pid_file.read_text().strip())
else:
    start_line = len(log_lines())  # only lines written by this launch count for progress and completion
    command = [sys.executable, 'scripts/pilot_driver.py', '--out-root', OUTPUT, '--session', str(SESSION)]
    job = subprocess.Popen(command, cwd=root, stdout=open(root / 'pilot_stdout.txt', 'a'),
                           stderr=subprocess.STDOUT, start_new_session=True)
    pid_file.write_text(str(job.pid))
    print('Pilot started, PID', job.pid)
"""),
        md("## Progress (interrupting this cell does not stop the pilot)"),
        code("""
import time
shown = start_line
end_markers = ('PILOT COMPLETE', 'PILOT STOPPED', 'PILOT SESSION 2 COMPLETE')
while True:
    lines = log_lines()
    for line in lines[shown:]:
        print(line, flush=True)
    shown = len(lines)
    if any(m in line for line in lines[start_line:] for m in end_markers):
        break
    alive = pathlib.Path(f'/proc/{pid_file.read_text().strip()}').exists()
    if not alive:
        print('Pilot process is not running. See pilot_stdout.txt below.')
        print((root / 'pilot_stdout.txt').read_text()[-3000:])
        break
    usage = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader'],
                           capture_output=True, text=True).stdout.strip()
    print(f'  ... {time.strftime("%H:%M:%S")} running, GPU {usage}', flush=True)
    time.sleep(60)
"""),
        md("## Export (download this zip to the workstation)"),
        code("""
output = pathlib.Path(OUTPUT)
zip_path = output.parent / 'bamo_pilot_results.zip'
hashes = {}
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive_zip:
    for folder in ('pilot', 'pilot_check', 'pilot_sweep', 'environment'):
        for path in sorted((output / folder).rglob('*')):
            if path.is_file():
                rel = path.relative_to(output).as_posix()
                archive_zip.write(path, arcname=rel)
                hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    archive_zip.writestr('export_manifest.json', json.dumps({'bundle_sha256': BUNDLE_SHA, 'files_sha256': hashes}, indent=2))
print(f'Ready in Google Drive: My Drive/{zip_path.name} ({zip_path.stat().st_size / 2**20:.1f} MiB, {len(hashes)} files)')
print('Disconnect the runtime now: G4 bills while idle.')
"""),
    ]
    return cells


def decision_notebook():
    cells = common_cells("bamo_decision", "BAMO — assessor decision phase on G4",
                         "Run in a new G4 runtime. The drift check compares with session 1; the headroom pilot "
                         "uses development folds and a registered plan. No confirmatory search or test evaluation "
                         "is launched. Results persist on Drive and completed headroom trainings are skipped.",
                         prepare_reference_data=False)
    cells += [md("## Verify the implementation"), code("""
result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '--disable-warnings'], capture_output=True, text=True)
print(result.stdout[-2000:])
assert result.returncode == 0, result.stderr[-3000:]

def run_logged(args, log_name):
    # The VS Code Colab kernel does not forward a child process's stdout, so relay it line by line
    # (visible in the cell and in the saved notebook) and keep a copy on Drive for the export.
    log_path = pathlib.Path(OUTPUT) / 'logs' / log_name
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, 'a') as log, subprocess.Popen([sys.executable, '-u', *args], stdout=subprocess.PIPE,
                                                     stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
        for line in proc.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, args)
"""), md("## Timing drift in a fresh G4 session"), code("""
from bamo.hardware import fingerprint
from bamo.utils import sha256_file
session_host = fingerprint()['host']
drift_path = pathlib.Path(OUTPUT) / 'timing_drift' / f'{session_host}.json'
if not drift_path.exists():
    run_logged(['scripts/timing_drift.py', '--out', str(drift_path)], 'timing_drift.txt')
drift = json.loads(drift_path.read_text())
assert drift['hardware']['host'] == session_host, 'Drift artifact belongs to a different runtime'
assert drift['baseline_sha256'] == sha256_file(root / 'configs/decision_timing_baseline.json'), 'Drift baseline changed'
print(json.dumps({k: drift[k] for k in ('median_ratio', 'within_10pct')}, indent=2))
assert drift['within_10pct'], 'Investigate session timing before calibration/freeze'
"""), md("## Headroom (run only if G4 replication is needed)"), code("""
RUN_HEADROOM = False  # local decision outputs are sufficient for landscape planning; G4 timing is separate
if RUN_HEADROOM:
    run_logged(['scripts/decision_pilot.py', '--tasks', 'form', 'rhythm', '--out-root', f'{OUTPUT}/headroom'],
               'headroom.txt')
"""), md("## Task calibration, paired precision and matched-control smoke\nRuns common-workload task calibration (1-6/7), separate precision-workload timing/noise, all four matched stopping arms, paired P/DEHB search, refit and development assessment. Select the task recommended by the local headroom decision. This is several hours on G4, depending on the task."), code("""
RUN_PRECISION = True  # next authorized development experiment; False runs drift only
PRECISION_TASK = 'rhythm'  # stronger fresh-seed headroom candidate; full benchmark remains gated
if RUN_PRECISION:
    for task in ('superclass', PRECISION_TASK):
        run_logged(['scripts/calibrate_task.py', '--task', task, '--out-root', f'{OUTPUT}/task_calibration'],
                   f'calibrate_{task}.txt')
    run_logged(['scripts/precision_pilot.py', '--task', PRECISION_TASK, '--out-root', f'{OUTPUT}/precision'],
               f'precision_{PRECISION_TASK}.txt')
"""), md("## Export evidence"), code("""
output = pathlib.Path(OUTPUT)
zip_path = output.parent / 'bamo_decision_results.zip'
hashes = {}
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as result_zip:
    for path in sorted(output.rglob('*')):
        if path.is_file():
            rel = path.relative_to(output).as_posix()
            result_zip.write(path, arcname=rel)
            hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    result_zip.writestr('export_manifest.json', json.dumps({'bundle_sha256': BUNDLE_SHA, 'files_sha256': hashes}, indent=2))
print('Evidence saved:', zip_path)
""")]
    return cells


def matched_followup_notebook():
    cells = common_cells("bamo_matched_followup_2026_10_14", "BAMO — matched-cost development follow-up",
                         "Verify and reuse the completed 14 October decision run. Run 15 new matched searches "
                         "(at most 5.81 search hours), assess DEHB at P's cost and two fixed 40% prefixes. "
                         "Up to 30 additional refits plus setup, drift and analysis are extra. The seven rhythm "
                         "labels and original settings are retained. Uses folds 1–7; full study remains gated.",
                         prepare_reference_data=False)
    cells[-2:-2] = [md("## Verify the previous evidence before new experiments\nThe original Drive folder is preferred. If it is absent, the previous exported ZIP is used. Edit SOURCE only if you stored the original evidence elsewhere."), code("""
def run_logged(args, log_name):
    log_path = pathlib.Path(OUTPUT) / 'logs' / log_name
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, 'a') as log, subprocess.Popen([sys.executable, '-u', *args], stdout=subprocess.PIPE,
                                                     stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
        for line in proc.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, args)

SOURCE = '/content/drive/MyDrive/bamo_decision'
if not (pathlib.Path(SOURCE) / 'precision/rhythm/plan.json').exists():
    prior_zip = pathlib.Path('/content/drive/MyDrive/bamo_decision_results.zip')
    assert prior_zip.exists(), 'Original evidence missing: restore bamo_decision or its exported ZIP to My Drive'
    SOURCE = '/content/bamo_decision_source'
    with zipfile.ZipFile(prior_zip) as archive:
        for name in archive.namelist():
            assert not pathlib.PurePosixPath(name).is_absolute() and '..' not in pathlib.PurePosixPath(name).parts and '\\\\' not in name
        archive.extractall(SOURCE)
run_logged(['scripts/matched_followup.py', '--source-root', SOURCE, '--out-root', OUTPUT, '--plan-only'],
           'preflight.txt')
""")]
    cells += [md("## Implementation checks"), code("""
run_logged(['-m', 'pytest', '-q', '--disable-warnings'], 'tests.txt')
"""), md("## Timing drift in this G4 session"), code("""
from bamo.hardware import fingerprint
from bamo.utils import sha256_file
session_host = fingerprint()['host']
drift_path = pathlib.Path(OUTPUT) / 'timing_drift' / f'{session_host}.json'
if not drift_path.exists():
    run_logged(['scripts/timing_drift.py', '--out', str(drift_path)], 'timing_drift.txt')
drift = json.loads(drift_path.read_text())
assert drift['hardware']['host'] == session_host, 'Different runtime drift artifact'
assert drift['baseline_sha256'] == sha256_file(root / 'configs/decision_timing_baseline.json')
print(json.dumps({k: drift[k] for k in ('median_ratio', 'within_10pct')}, indent=2))
assert drift['within_10pct'], 'Session speed changed; investigate before matched comparisons'
"""), md("## Searches, preselected refits and paired assessment\nCompleted jobs resume. Original results remain unchanged. Every configuration is selected from search logs before assessment predictions are opened."), code("""
run_logged(['scripts/matched_followup.py', '--source-root', SOURCE, '--out-root', OUTPUT,
            '--drift', str(drift_path)], 'matched_followup.txt')
summary = json.loads((pathlib.Path(OUTPUT) / 'followup_summary.json').read_text())
print('Development assessment AUROC:', json.dumps(summary['mean_auroc'], indent=2))
print('P versus DEHB at P cost:', json.dumps(summary['contrasts']['adaptive_utility-dehb_at_p_cost'], indent=2))
"""), md("## Export evidence\nBring this ZIP and the executed notebook back to the workspace."), code("""
output = pathlib.Path(OUTPUT)
zip_path = output.parent / 'bamo_matched_followup_results.zip'
hashes = {}
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as result_zip:
    for path in sorted(output.rglob('*')):
        if path.is_file():
            rel = path.relative_to(output).as_posix()
            result_zip.write(path, arcname=rel)
            hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    result_zip.writestr('export_manifest.json', json.dumps({'bundle_sha256': BUNDLE_SHA, 'files_sha256': hashes}, indent=2))
print('Evidence saved:', zip_path)
""")]
    return cells


NOTEBOOKS = {"setup": ("COLAB_G4_SETUP.ipynb", setup_notebook), "pilot": ("COLAB_G4_PILOT.ipynb", pilot_notebook),
             "decision": ("COLAB_G4_DECISION.ipynb", decision_notebook),
             "matched_followup": ("COLAB_G4_MATCHED_FOLLOWUP.ipynb", matched_followup_notebook)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=list(NOTEBOOKS))
    args = ap.parse_args()
    filename, builder = NOTEBOOKS[args.which]
    cells = builder()
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    out = PROJECT_ROOT / "notebooks" / filename
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {out.relative_to(PROJECT_ROOT)} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
