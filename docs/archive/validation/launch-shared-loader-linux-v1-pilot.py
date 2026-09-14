import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path('/home/yonghye/ultrafast-vision-build')
PREFIX = 'mask-shared-loader-linux-v1'
SOURCE = ROOT / (PREFIX + '-source')
PYTHON = ROOT / 'mask-shared-linux-v2-clean/bin/python'
CORPUS = ROOT / 'coco-val2017-yolo/segment'
STATE_PATH = ROOT / (PREFIX + '-pilot-state.json')
STATE = {'complete': False, 'controller_pid': os.getpid(), 'started_at_ns': time.time_ns(),
         'scope': 'Full 5000-image functional pilot; one reference/native pair at workers 0/2/8 for each mask mode. Twelve fresh measured processes plus two fresh-reference verification processes. No five-pair aggregate or production qualification.',
         'commands': []}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save():
    STATE_PATH.write_text(json.dumps(STATE, indent=2) + '\n')


def run(label, arguments):
    log = ROOT / (PREFIX + '-' + label + '.log')
    command = [str(PYTHON), *map(str, arguments)]
    rec = {'label': label, 'command': command, 'cwd': str(SOURCE), 'started_at_ns': time.time_ns(), 'log': str(log)}
    STATE['commands'].append(rec)
    save()
    print(label, 'started', flush=True)
    with log.open('x') as stream:
        process = subprocess.Popen(command, cwd=SOURCE, env=ENV, stdout=stream, stderr=subprocess.STDOUT)
        rec['pid'] = process.pid
        save()
        code = process.wait()
    rec.update(returncode=code, finished_at_ns=time.time_ns(), log_sha256=sha(log))
    save()
    print(label, 'returncode', code, flush=True)
    if code:
        print(log.read_text()[-10000:], flush=True)
        raise RuntimeError(f'{label} failed; preserve the original run')


def main():
    assert not STATE_PATH.exists()
    source = json.loads((ROOT / (PREFIX + '-source.json')).read_text())
    for name, digest in source['files'].items():
        assert sha(SOURCE / name) == digest, name
    qualification = ROOT / 'mask-shared-linux-v2-qualification.json'
    stress = ROOT / 'mask-shared-linux-v2-spawn-stress/series.json'
    assert json.loads(qualification.read_text())['installed_suite_passed']
    assert json.loads(stress.read_text())['passed']
    STATE.update(script_sha256=sha(Path(__file__)), harness_commit=source['harness_commit'],
                 runtime_source_commit=source['runtime_source_commit'], source_archive_sha256=source['archive_sha256'],
                 qualification_sha256=sha(qualification), stress_sha256=sha(stress),
                 environment_overrides={k: ENV[k] for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'YOLO_OFFLINE')})
    save()
    for overlap in ('yes', 'no'):
        output = ROOT / (PREFIX + '-pilot-' + overlap + '.json')
        assert not output.exists()
        run('pilot-' + overlap, ['bench/coco_loader.py', '--corpus', CORPUS, '--out', output,
                                '--workers', '0', '2', '8', '--rounds', '1', '--overlap', overlap, '--persistent-mask'])
        report = json.loads(output.read_text())
        assert report['complete'] and len(report['results']) == 6
        assert {(r['workers'], r['round'], r['backend']) for r in report['results']} == {
            (w, 0, b) for w in (0, 2, 8) for b in ('reference', 'native')}
        assert all(r['images'] == r['verification_images'] == 5000 and r['verification_batches'] == 625 for r in report['results'])
        STATE.setdefault('completed_reports', {})[overlap] = sha(output)
        save()
    # Both timing modes are terminal before any generated-cache rebuild.
    for overlap in ('yes', 'no'):
        benchmark = ROOT / (PREFIX + '-pilot-' + overlap + '.json')
        output = ROOT / (PREFIX + '-fresh-' + overlap + '.json')
        run('fresh-' + overlap, ['bench/verify_coco_loader.py', '--corpus', CORPUS, '--benchmark', benchmark, '--out', output])
        assert json.loads(output.read_text())['all_benchmark_outputs_match_fresh_reference']
        STATE.setdefault('fresh_reports', {})[overlap] = sha(output)
        save()
    STATE.update(complete=True, pilot_passed=True, finished_at_ns=time.time_ns(), production_qualified=False)
    save()


ENV = os.environ.copy()
for key in ('PYTHONPATH', 'PYTHONHOME'):
    ENV.pop(key, None)
ENV.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', YOLO_OFFLINE='true')

if __name__ == '__main__':
    try:
        main()
    except BaseException:
        STATE.update(complete=True, pilot_passed=False, error=traceback.format_exc(), finished_at_ns=time.time_ns())
        save()
        raise
