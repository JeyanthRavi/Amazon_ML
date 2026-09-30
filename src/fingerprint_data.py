"""Record supplied input hashes and the local runtime without copying datasets."""
import argparse
import hashlib
import json
import platform
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = {'python': platform.python_version(), 'implementation': platform.python_implementation(),
                'system': platform.system(), 'machine': platform.machine(), 'dependencies': 'Python standard library only', 'inputs': {}}
    for path in sorted(args.data.glob('*/*.tsv')):
        manifest['inputs'][str(path.relative_to(args.data))] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
        print(f'Hashed {path.name}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
