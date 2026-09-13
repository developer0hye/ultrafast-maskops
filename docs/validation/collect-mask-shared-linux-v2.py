import hashlib
import itertools
import json
from pathlib import Path
import tarfile
import xml.etree.ElementTree as ET

ROOT = Path('/home/yonghye/ultrafast-vision-build')
PREFIX = 'mask-shared-linux-v2'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(suffix):
    return json.loads((ROOT / (PREFIX + '-' + suffix)).read_text())


def main():
    qualification = read('qualification.json')
    source = read('source.json')
    build = read('build.json')
    stress = read('spawn-stress/series.json')
    assert qualification['complete'] and qualification['installed_suite_passed']
    assert qualification['standalone_passed'] and qualification['normalized_dependency_pins_match']
    assert qualification['source_and_installed_runtime_bytes_match']
    assert qualification['source_commit'] == source['source_commit'] == build['source_commit'] == stress['source_commit']
    assert build['complete'] and build['returncode'] == 0
    assert build['source_archive_sha256'] == source['archive_sha256'] == sha(ROOT / (PREFIX + '-source.tar'))
    for command in qualification['commands']:
        assert command['returncode'] == 0
        assert sha(Path(command['log'])) == command['log_sha256']
    cases = ET.parse(ROOT / (PREFIX + '-tests.xml')).findall('.//testcase')
    assert len(cases) == 257
    assert all(c.find(tag) is None for c in cases for tag in ('failure', 'error', 'skipped'))
    assert sha(ROOT / (PREFIX + '-tests.xml')) == qualification['tests']['xml_sha256']
    assert sum('test_shared_collate' in c.attrib['classname'] for c in cases) == 26
    assert stress['complete'] and stress['passed'] and len(stress['records']) == 18
    assert stress['script_sha256'] == sha(ROOT / (PREFIX + '-spawn-stress.py'))
    assert stress['qualification_sha256'] == sha(ROOT / (PREFIX + '-qualification.json'))
    plan = list(itertools.product(range(3), ('setup', 'first-batch', 'full-epoch'), (False, True)))
    for rec, (round_, reset, overlap) in zip(stress['records'], plan):
        assert (rec['round'], rec['reset'], rec['overlap']) == (round_, reset, overlap)
        assert rec['returncode'] == 0 and rec['case_passed'] and not rec.get('timed_out', False)
        log = ROOT / (PREFIX + '-spawn-stress') / (str(rec['index']) + '.log')
        xml = log.with_suffix('.xml')
        assert sha(log) == rec['log_sha256'] and sha(xml) == rec['xml_sha256']
        assert not any(text in log.read_text() for text in ('SIGABRT', 'terminate called', 'Fatal Python error'))
        repeated = ET.parse(xml).findall('.//testcase')
        assert len(repeated) == 1
        assert repeated[0].attrib['name'] == f'test_close_mosaic_retains_format_through_real_worker_reset[{reset}-{overlap}-2]'
        assert all(repeated[0].find(tag) is None for tag in ('failure', 'error', 'skipped'))

    suffixes = [
        'source.tar', 'source.json', 'build.json', 'build.log', 'lint.json',
        'qualify.py', 'qualification.json', 'requirements.txt', 'freeze.txt',
        'standalone-venv.log', 'standalone-numpy.log', 'standalone-wheel.log',
        'standalone-audit.log', 'standalone-wheel-validation.json', 'dependencies.log',
        'pip-check.log', 'tests.log', 'tests.xml', 'spawn-stress.py', 'collect.py',
    ]
    files = [ROOT / (PREFIX + '-' + suffix) for suffix in suffixes]
    files += [ROOT / (PREFIX + '-clean/pyvenv.cfg')]
    files += list((ROOT / (PREFIX + '-dist')).glob('*.whl'))
    files += list((ROOT / (PREFIX + '-dist')).glob('*.tar.gz'))
    files += [p for p in (ROOT / (PREFIX + '-spawn-stress')).iterdir() if p.is_file()]
    for file in files:
        assert file.is_file() and not file.is_symlink()
    manifest_files = {str(file.relative_to(ROOT)): {'bytes': file.stat().st_size, 'sha256': sha(file)} for file in files}
    assert len(manifest_files) == len(files)
    archive = ROOT / (PREFIX + '-qualification-evidence.tar.gz')
    assert not archive.exists()
    with tarfile.open(archive, 'w:gz') as tar:
        for file in files:
            tar.add(file, arcname=str(file.relative_to(ROOT)), recursive=False)
    with tarfile.open(archive) as tar:
        assert set(tar.getnames()) == set(manifest_files)
        for name, info in manifest_files.items():
            data = tar.extractfile(name).read()
            assert len(data) == info['bytes'] and hashlib.sha256(data).hexdigest() == info['sha256']
    result = {'preservation_complete': True, 'installed_suite_passed': True, 'spawn_stress_passed': True,
              'production_qualified': False, 'source_commit': source['source_commit'],
              'tests': {'passed': 257, 'failed': 0, 'skipped': 0, 'shared_collation_cases': 26},
              'fresh_stress_processes': 18, 'archive_bytes': archive.stat().st_size,
              'archive_sha256': sha(archive), 'file_count': len(files), 'files': manifest_files,
              'all_archive_members_read_back': True,
              'limitations': ['One Linux CPU/Python/Torch profile; four-image spawned stress fixture, workers=2.',
                              'Reference transport normalized only in functional reset tests.',
                              'Previous v1 failed suite and GDB evidence retained separately.',
                              'Current real-data loader and GPU lifecycle/resume grid remain unexecuted.',
                              'No packet speed, memory improvement or general shutdown-reliability claim.']}
    (ROOT / (PREFIX + '-qualification-preservation.json')).write_text(json.dumps(result, indent=2) + '\n')
    print({k: v for k, v in result.items() if k != 'files'})


if __name__ == '__main__':
    main()
