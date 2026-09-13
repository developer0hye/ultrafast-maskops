import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import xml.etree.ElementTree as ET

ROOT = Path('/home/yonghye/ultrafast-vision-build')
PREFIX = 'mask-shared-linux-v2'
SOURCE = ROOT / (PREFIX + '-source')
OUT = ROOT / (PREFIX + '-spawn-stress')
PYTHON = ROOT / (PREFIX + '-clean/bin/python')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    qualification_path = ROOT / (PREFIX + '-qualification.json')
    qualification = json.loads(qualification_path.read_text())
    assert qualification['complete'] and qualification['installed_suite_passed']
    source = json.loads((ROOT / (PREFIX + '-source.json')).read_text())
    for name, digest in source['files'].items():
        assert sha(SOURCE / name) == digest, name
    OUT.mkdir(exist_ok=False)
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'PYTHONHOME'):
        env.pop(key, None)
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', YOLO_OFFLINE='true')
    plan = list(itertools.product(range(3), ('setup', 'first-batch', 'full-epoch'), (False, True)))
    report = {'complete': False, 'passed': False, 'controller_pid': os.getpid(),
              'script_sha256': sha(Path(__file__)), 'qualification_sha256': sha(qualification_path),
              'source_commit': source['source_commit'], 'planned_fresh_processes': len(plan),
              'scope': 'Three fresh-process repetitions of each reset point and overlap mode, two spawned workers; original reference computation with test-only normalized packet transport versus native persistent adapter. Exact post-reset batch parity and strict zero exits for both old and replacement workers. Four-image synthetic fixture, no performance or universal reliability claim.',
              'environment_overrides': qualification['environment_overrides'], 'records': []}
    record_path = OUT / 'series.json'

    def save():
        record_path.write_text(json.dumps(report, indent=2) + '\n')

    save()
    for index, (round_, reset, overlap) in enumerate(plan):
        case = f'tests/test_persistent_format.py::test_close_mosaic_retains_format_through_real_worker_reset[{reset}-{overlap}-2]'
        xml = OUT / f'{index}.xml'
        log = OUT / f'{index}.log'
        cmd = [str(PYTHON), '-m', 'pytest', '-q', case, '--junitxml=' + str(xml)]
        rec = {'index': index, 'round': round_, 'reset': reset, 'overlap': overlap,
               'command': cmd, 'started_at_ns': time.time_ns(), 'cwd': str(SOURCE)}
        report['records'].append(rec)
        save()
        with log.open('x') as stream:
            proc = subprocess.Popen(cmd, cwd=SOURCE, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            rec['pid'] = proc.pid
            save()
            try:
                code = proc.wait(timeout=180)
            except subprocess.TimeoutExpired:
                rec['timed_out'] = True
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    code = proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    code = proc.wait()
        rec.update(returncode=code, finished_at_ns=time.time_ns(), log_sha256=sha(log))
        if xml.exists():
            cases = ET.parse(xml).findall('.//testcase')
            rec.update(xml_sha256=sha(xml), cases=len(cases),
                       case_passed=len(cases) == 1 and all(c.find(tag) is None for c in cases for tag in ('failure', 'error', 'skipped')))
        passed = code == 0 and rec.get('case_passed', False) and not rec.get('timed_out', False)
        save()
        print(index, reset, overlap, 'passed' if passed else 'FAILED', flush=True)
        if not passed:
            report.update(complete=True, stopped_after_failure=True)
            save()
            raise SystemExit(1)
    report.update(complete=True, passed=True, finished_at_ns=time.time_ns())
    save()


if __name__ == '__main__':
    main()
