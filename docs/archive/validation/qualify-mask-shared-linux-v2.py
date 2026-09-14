import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET


ROOT = Path('/home/yonghye/ultrafast-vision-build')
PREFIX = 'mask-shared-linux-v2'
SOURCE = ROOT / (PREFIX + '-source')
DIST = ROOT / (PREFIX + '-dist')
CLEAN = ROOT / (PREFIX + '-clean')
PYTHON = CLEAN / 'bin/python'
UV = '/home/yonghye/.local/bin/uv'
ENV = os.environ.copy()
for name in ('PYTHONPATH', 'PYTHONHOME'):
    ENV.pop(name, None)
ENV.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', YOLO_OFFLINE='true')
STATE = {'complete': False, 'controller_pid': os.getpid(), 'started_at_ns': time.time_ns(), 'commands': []}


def path(suffix):
    return ROOT / (PREFIX + '-' + suffix)


def sha(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def save():
    path('qualification.json').write_text(json.dumps(STATE, indent=2) + '\n')


def run(label, cmd, timeout=600):
    log = path(label + '.log')
    rec = {'label': label, 'command': list(map(str, cmd)), 'cwd': str(SOURCE),
           'started_at_ns': time.time_ns(), 'log': str(log)}
    STATE['commands'].append(rec)
    save()
    print(label, 'started', flush=True)
    with log.open('x') as stream:
        result = subprocess.run(rec['command'], cwd=SOURCE, env=ENV,
                                stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
    rec.update(returncode=result.returncode, finished_at_ns=time.time_ns(), log_sha256=sha(log))
    save()
    print(label, 'returncode', result.returncode, flush=True)
    if result.returncode:
        print(log.read_text()[-12000:], flush=True)
    return result.returncode


def pins(lines):
    result = {}
    for line in lines:
        if not line or line.startswith('#'):
            continue
        name = re.split(r'==| @ ', line, maxsplit=1)[0]
        norm = re.sub(r'[-_.]+', '-', name).lower()
        if norm not in ('pip', 'ultrafast-maskops'):
            assert norm not in result
            result[norm] = line[len(name):]
    return result


def main():
    assert not path('qualification.json').exists() and not CLEAN.exists()
    build = json.loads(path('build.json').read_text())
    assert build['complete'] and build['returncode'] == 0
    source = json.loads(path('source.json').read_text())
    STATE.update(source_commit=source['source_commit'], source_archive_sha256=source['archive_sha256'],
                 environment_overrides={k: ENV[k] for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'YOLO_OFFLINE')})
    assert sha(path('source.tar')) == source['archive_sha256']
    for name, digest in source['files'].items():
        assert sha(SOURCE / name) == digest, name
    wheel, = DIST.glob('*.whl')
    sdist, = DIST.glob('*.tar.gz')
    assert sha(wheel) == build['artifacts'][wheel.name]
    assert sha(sdist) == build['artifacts'][sdist.name]
    save()
    assert run('standalone-venv', [UV, 'venv', '--seed', '--python', ROOT / 'venv312-cuda/bin/python', CLEAN]) == 0
    assert 'include-system-site-packages = false' in (CLEAN / 'pyvenv.cfg').read_text()
    assert run('standalone-numpy', [UV, 'pip', 'install', '--python', PYTHON, '--no-deps', 'numpy==2.4.4', 'pip==26.2.1']) == 0
    assert run('standalone-wheel', [UV, 'pip', 'install', '--python', PYTHON, '--no-deps', wheel]) == 0
    assert not (SOURCE / 'dist').exists()
    (SOURCE / 'dist').symlink_to(DIST, target_is_directory=True)
    assert run('standalone-audit', [PYTHON, 'bench/ci_wheel_check.py']) == 0
    audit = DIST / 'wheel-validation.json'
    path('standalone-wheel-validation.json').write_bytes(audit.read_bytes())
    result = json.loads(audit.read_text())
    assert result['notice_files_checked'] == 9 and result['clean']
    STATE['standalone_passed'] = True
    STATE['standalone_audit_sha256'] = sha(audit)
    save()

    baseline = ROOT / 'mask-unit-scale-linux-freeze-v1.txt'
    lines = baseline.read_text().splitlines()
    excluded = [line for line in lines if line.startswith('ultrafast-maskops @ ')]
    assert len(excluded) == 1
    requirements = path('requirements.txt')
    requirements.write_text('\n'.join(line for line in lines if line not in excluded) + '\n')
    STATE.update(baseline_freeze_sha256=sha(baseline), requirements_sha256=sha(requirements), excluded_old_wheel=excluded)
    assert run('dependencies', [UV, 'pip', 'install', '--python', PYTHON, '--no-deps', '-r', requirements]) == 0
    assert run('pip-check', [PYTHON, '-m', 'pip', 'check']) == 0
    freeze = subprocess.check_output([str(PYTHON), '-m', 'pip', 'freeze'], cwd=SOURCE, env=ENV, text=True)
    path('freeze.txt').write_text(freeze)
    assert pins(lines) == pins(freeze.splitlines())
    assert len(pins(lines)) == 79
    STATE.update(normalized_dependency_pins_match=True, freeze_sha256=sha(path('freeze.txt')))
    installed = Path(json.loads(subprocess.check_output([str(PYTHON), '-c',
        'import sysconfig,json; print(json.dumps(sysconfig.get_path("platlib")))'], env=ENV, text=True)))
    for name, digest in source['files'].items():
        assert sha(SOURCE / name) == digest, name
        if name.startswith('python/'):
            assert sha(installed / name.removeprefix('python/')) == digest, name
    STATE['source_and_installed_runtime_bytes_match'] = True
    save()
    code = run('tests', [PYTHON, '-m', 'pytest', '-q', 'tests', '--junitxml=' + str(path('tests.xml'))])
    xml = ET.parse(path('tests.xml'))
    cases = xml.findall('.//testcase')
    failed = [c.attrib for c in cases if c.find('failure') is not None or c.find('error') is not None]
    skipped = [c.attrib for c in cases if c.find('skipped') is not None]
    STATE['tests'] = {'total': len(cases), 'passed': len(cases)-len(failed)-len(skipped), 'failed': failed,
                      'skipped': skipped, 'xml_sha256': sha(path('tests.xml')), 'returncode': code}
    STATE['complete'] = True
    STATE['installed_suite_passed'] = code == 0 and len(cases) == 257 and not failed and not skipped
    STATE['finished_at_ns'] = time.time_ns()
    save()
    print(json.dumps(STATE['tests']), flush=True)
    assert STATE['installed_suite_passed']


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        STATE['error'] = traceback.format_exc()
        save()
        raise
