"""Read back compiled dependency provenance and the installed Darwin binary."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/Volumes/T7/ultrafast-vision-build')
PREFIX = 'mask-packet-m2-v1'
SOURCE = Path('/Users/yhkwon/Documents/ultrafast-maskops-m2-packet-v1-source')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = ROOT / (PREFIX + '-native-inspection-v2.json')
    assert not out.exists()
    compact = json.loads((SOURCE / 'licenses/manifest.json').read_text())
    provenance = ROOT / (PREFIX + '-notice-provenance.json')
    assert sha(provenance) == compact['provenance_sha256']
    manifest = json.loads(provenance.read_text())
    graphpath = ROOT / (PREFIX + '-compiler-deps.txt.gz')
    graph = gzip.decompress(graphpath.read_bytes())
    selected, omitted = {}, set()
    for line in graph.decode().splitlines():
        if not line.startswith('    '):
            continue
        name = line.strip()
        for marker, prefix in [('opencv-src/', 'opencv-4.13.0'),
                               ('kleidicv-0.7.0/', 'kleidicv-0.7.0'),
                               ('site-packages/pybind11/include/', 'pybind11-3.0.1/include')]:
            if marker in name:
                key = prefix + '/' + name.split(marker, 1)[1]
                file = Path(name)
                value = dict(bytes=file.stat().st_size, sha256=sha(file))
                assert key not in selected or selected[key] == value
                selected[key] = value
                break
        else:
            omitted.add(name)
    missing = [n for n in selected if n not in manifest['source_files']]
    changed = [n for n, v in selected.items() if n in manifest['source_files'] and v != manifest['source_files'][n]]
    binary, = (ROOT / (PREFIX + '-clean/lib/python3.12/site-packages/ultrafast_maskops')).glob('_native*.so')
    commands = {}
    for label, cmd in [('otool-libraries', ['otool', '-L', str(binary)]),
                       ('otool-load-commands', ['otool', '-l', str(binary)]),
                       ('exports', ['nm', '-gU', str(binary)])]:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        log = ROOT / (PREFIX + '-' + label + '.log')
        with log.open('x') as stream:
            stream.write(proc.stdout + proc.stderr)
        commands[label] = dict(command=cmd, returncode=proc.returncode, log=str(log), sha256=sha(log))
        assert proc.returncode == 0
    result = dict(scope='Observed compiled dependency bytes against hash-bound notice source inventory. Excludes generated, system, Python and toolchain files. Darwin load commands and exports retained; no older macOS execution or complete license review claim.',
                  inspector_sha256=sha(Path(__file__)), extension=str(binary), extension_sha256=sha(binary),
                  source_manifest_sha256=sha(SOURCE / 'licenses/manifest.json'), provenance_sha256=sha(provenance),
                  compressed_graph=str(graphpath), compressed_graph_sha256=sha(graphpath), selected_count=len(selected),
                  selected=selected, missing=missing, changed=changed, omitted=sorted(omitted), commands=commands,
                  selected_sources_covered=not missing and not changed)
    with out.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('extension_sha256', 'selected_count', 'missing', 'changed', 'selected_sources_covered')}))
    assert selected and not missing and not changed


if __name__ == '__main__':
    main()
