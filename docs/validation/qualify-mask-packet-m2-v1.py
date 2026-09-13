"""Single-host, fail-stop qualification of a frozen macOS packet wheel."""

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET


ROOT = Path('/Volumes/T7/ultrafast-vision-build')
PREFIX = 'mask-packet-m2-v1'
SOURCE = Path('/Users/yhkwon/Documents/ultrafast-maskops-m2-packet-v1-source')
UV = '/Users/yhkwon/.local/bin/uv'
BASE = ROOT / 'mask-resize-roi-clean-v2/bin/python'
BUILD = ROOT / (PREFIX + '-build-env')
CLEAN = ROOT / (PREFIX + '-clean')
DIST = ROOT / (PREFIX + '-dist')
PYTHON = CLEAN / 'bin/python'
ENV = os.environ.copy()
for key in ('PYTHONPATH', 'PYTHONHOME'):
    ENV.pop(key, None)
OVERRIDES = dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                 YOLO_OFFLINE='true', CMAKE_BUILD_PARALLEL_LEVEL='2', MACOSX_DEPLOYMENT_TARGET='11.0')
ENV.update(OVERRIDES)
STATE = dict(complete=False, passed=False, controller_pid=os.getpid(), started_at_ns=time.time_ns(),
             platform=platform.platform(), machine=platform.machine(), environment_overrides=OVERRIDES,
             commands=[], scope='Installed macOS arm64 wheel and pinned framework tests; no macOS 11 execution, speedup, named-storage lifetime or full real-data training claim.')


def path(suffix):
    return ROOT / (PREFIX + '-' + suffix)


def sha(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def save():
    tmp = path('qualification.tmp')
    tmp.write_text(json.dumps(STATE, indent=2) + '\n')
    tmp.replace(path('qualification.json'))


def run(label, cmd, timeout=600):
    log = path(label + '.log')
    rec = dict(label=label, command=list(map(str, cmd)), cwd=str(SOURCE),
               log=str(log), started_at_ns=time.time_ns())
    STATE['commands'].append(rec)
    save()
    print(label, 'started', flush=True)
    with log.open('x') as stream:
        child = subprocess.Popen(rec['command'], cwd=SOURCE, env=ENV, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        rec['pid'] = child.pid
        save()
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            rec['timed_out'] = True
            sample = path(label + '-timeout-sample.txt')
            with sample.open('x') as output:
                observation = subprocess.run(['sample', str(child.pid), '1', '10'],
                                             stdout=output, stderr=subprocess.STDOUT, timeout=30)
            rec['timeout_sample'] = dict(path=str(sample), sha256=sha(sample), returncode=observation.returncode)
            os.killpg(child.pid, signal.SIGTERM)
            try:
                code = child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait(timeout=10)
    rec.update(returncode=code, finished_at_ns=time.time_ns(), log_sha256=sha(log))
    save()
    print(label, 'returncode', code, flush=True)
    return code


def pins(lines):
    result = {}
    for line in lines:
        if not line or line.startswith('#'):
            continue
        name = re.split(r'==| @ ', line, maxsplit=1)[0]
        normalized = re.sub(r'[-_.]+', '-', name).lower()
        if normalized not in ('pip', 'ultrafast-maskops'):
            assert normalized not in result
            result[normalized] = line[len(name):]
    return result


def verify_sources(source):
    for name, digest in source['files'].items():
        assert sha(SOURCE / name) == digest, name


def main():
    assert platform.system() == 'Darwin' and platform.machine() == 'arm64'
    assert not path('qualification.json').exists()
    assert not BUILD.exists() and not CLEAN.exists() and not DIST.exists()
    source = json.loads(path('source.json').read_text())
    assert sha(path('source.tar')) == source['archive_sha256']
    verify_sources(source)
    STATE.update(source_commit=source['source_commit'], source_archive_sha256=source['archive_sha256'],
                 qualifier_sha256=sha(Path(__file__)))
    save()
    assert run('build-venv', [UV, 'venv', '--seed', '--python', BASE, BUILD]) == 0
    bp = BUILD / 'bin/python'
    assert run('build-deps', [UV, 'pip', 'install', '--python', bp, 'build==1.3.0',
                             'scikit-build-core==0.11.6', 'pybind11==3.0.1', 'ninja==1.13.2', 'numpy==2.4.4']) == 0
    assert run('compiler', ['clang++', '--version']) == 0
    assert run('cmake', ['cmake', '--version']) == 0
    assert run('build-freeze', [UV, 'pip', 'freeze', '--python', bp]) == 0
    assert run('build', [bp, '-m', 'build', '--wheel', '--sdist', '--no-isolation', '--outdir', DIST,
                         '-Cbuild-dir=' + str(ROOT / (PREFIX + '-build'))], timeout=1800) == 0
    wheel, = DIST.glob('*.whl')
    sdist, = DIST.glob('*.tar.gz')
    STATE['artifacts'] = {file.name: dict(bytes=file.stat().st_size, sha256=sha(file)) for file in (wheel, sdist)}
    verify_sources(source)
    assert run('clean-venv', [UV, 'venv', '--seed', '--python', BASE, CLEAN]) == 0
    assert 'include-system-site-packages = false' in (CLEAN / 'pyvenv.cfg').read_text()
    assert run('standalone-install', [UV, 'pip', 'install', '--python', PYTHON, '--no-deps',
                                      'numpy==2.4.4', 'pip==26.2.1', wheel]) == 0
    assert not (SOURCE / 'dist').exists()
    (SOURCE / 'dist').symlink_to(DIST, target_is_directory=True)
    assert run('standalone-audit', [PYTHON, 'bench/ci_wheel_check.py']) == 0
    audit = DIST / 'wheel-validation.json'
    data = json.loads(audit.read_text())
    assert data['clean'] and data['notice_files_checked'] == 9
    path('standalone-audit.json').write_bytes(audit.read_bytes())
    STATE.update(standalone_passed=True, standalone_audit_sha256=sha(audit))
    baseline = path('baseline-freeze.txt')
    lines = baseline.read_text().splitlines()
    old = [line for line in lines if line.startswith('ultrafast-maskops @ ')]
    assert len(old) == 1
    requirements = path('requirements.txt')
    requirements.write_text('\n'.join(line for line in lines if line not in old) + '\n')
    local_dependencies = {}
    for line in lines:
        if line not in old and ' @ file://' in line:
            dep = Path(line.split(' @ file://', 1)[1])
            local_dependencies[line.split(' @ ', 1)[0]] = dict(path=str(dep), sha256=sha(dep))
    STATE.update(baseline_freeze_sha256=sha(baseline), requirements_sha256=sha(requirements),
                 excluded_old_wheel=old, local_dependency_artifacts=local_dependencies)
    save()
    assert run('framework-deps', [UV, 'pip', 'install', '--python', PYTHON, '--no-deps', '-r', requirements]) == 0
    assert run('pip-check', [PYTHON, '-m', 'pip', 'check']) == 0
    assert run('freeze', [PYTHON, '-m', 'pip', 'freeze']) == 0
    assert pins(lines) == pins(path('freeze.log').read_text().splitlines())
    STATE['normalized_dependency_pins_match'] = True
    STATE['dependency_pin_count'] = len(pins(lines))
    installed = Path(subprocess.check_output([str(PYTHON), '-c', 'import sysconfig; print(sysconfig.get_path("platlib"))'],
                                            env=ENV, text=True).strip())
    for name, digest in source['files'].items():
        if name.startswith('python/'):
            assert sha(installed / name.removeprefix('python/')) == digest, name
    STATE['source_and_installed_runtime_bytes_match'] = True
    assert run('runtime', [PYTHON, '-c', 'import torch,platform; import ultrafast_maskops; print(platform.platform()); print(torch.__version__); print(torch.multiprocessing.get_all_sharing_strategies()); print(torch.multiprocessing.get_sharing_strategy()); print(ultrafast_maskops.__file__)']) == 0
    assert run('collect', [PYTHON, '-m', 'pytest', '--collect-only', '-q', 'tests']) == 0
    # Count the node IDs independently of the eventual test result.
    collected = [line.strip() for line in path('collect.log').read_text().splitlines() if line.startswith('tests/') and '::' in line]
    assert len(collected) == len(set(collected)) and len(collected) >= 366
    STATE['collected_tests'] = len(collected)
    code = run('tests', [PYTHON, '-m', 'pytest', '-q', 'tests', '--junitxml=' + str(path('tests.xml'))], timeout=600)
    if path('tests.xml').exists():
        cases = ET.parse(path('tests.xml')).findall('.//testcase')
        bad = [dict(case.attrib, outcomes=[tag for tag in ('failure', 'error', 'skipped') if case.find(tag) is not None])
               for case in cases if any(case.find(tag) is not None for tag in ('failure', 'error', 'skipped'))]
        STATE['tests'] = dict(total=len(cases), bad=bad, xml_sha256=sha(path('tests.xml')), returncode=code)
        save()
        assert len(cases) == len(collected) and not bad and code == 0
    else:
        raise AssertionError('test process did not produce JUnit')
    assert run('installed-audit', [PYTHON, 'bench/audit_wheel.py', '--wheel', wheel, '--installed-root', installed,
                                  '--out', path('installed-audit.json')]) == 0
    verify_sources(source)
    STATE.update(complete=True, passed=True, finished_at_ns=time.time_ns())
    save()


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        STATE.update(complete=True, passed=False, finished_at_ns=time.time_ns(), error=traceback.format_exc())
        save()
        raise
