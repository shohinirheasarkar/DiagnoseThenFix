"""This losslessly compresses finished logs and records hashes so reviewers can check the evidence."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path


def package(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.is_relative_to(source) or source == destination:
        raise ValueError('Put the archive outside the source folder')
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Archive destination must be empty; use a new folder')
    records = []
    for path in sorted(source.rglob('*')):
        if not path.is_file() or path.suffix not in ('.json', '.jsonl'):
            continue
        before = path.stat()
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        if path.suffix == '.jsonl':
            target = target.with_suffix('.jsonl.gz')
            temporary = target.with_suffix('.tmp')
            with path.open('rb') as incoming, temporary.open('wb') as raw:
                with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as outgoing:
                    while chunk := incoming.read(1024*1024):
                        digest.update(chunk)
                        outgoing.write(chunk)
            temporary.replace(target)
        else:
            content = path.read_bytes()
            digest.update(content)
            target.write_bytes(content)
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError(f'Source changed during packaging: {path}')
        records.append({'source': str(relative), 'artifact': str(target.relative_to(destination)),
                        'source_sha256': digest.hexdigest(),
                        'artifact_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                        'source_bytes': path.stat().st_size, 'artifact_bytes': target.stat().st_size})
    manifest = {'created_at': datetime.now(timezone.utc).isoformat(), 'files': records,
                'lossless': True, 'compression': 'gzip, mtime=0'}
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'archive.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    result = package(args.source, args.destination)
    print(f'Archived {len(result["files"])} evidence files.')


if __name__ == '__main__':
    main()
